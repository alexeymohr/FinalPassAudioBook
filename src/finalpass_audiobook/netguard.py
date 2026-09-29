"""Fail-closed network guard for every analysis run.

Once client audio is opened there must be no downloading, update checking,
telemetry or remote inference. While a guard is active this refuses — loopback
included — and counts:

* every socket connect, send (TCP or UDP), bind and name lookup, including the
  low-level `_socket` module and anything else that raises Python's audit events
  (`sys.addaudithook`, which fires inside the interpreter's C code);
* starting another process (subprocess, os.system/exec/spawn/fork), since a child
  process would run without the guard; multiprocessing's "spawn" start method
  raises no audit event, so its launcher is blocked directly;
* looking up C network functions through ctypes (connect, sendto, getaddrinfo…).

Every report records the count, which must be 0. The macOS app adds the OS
sandbox (no network entitlement) on top. An audit hook cannot be removed, so the
hook is installed once and does nothing while no guard is active.
"""
from __future__ import annotations

import os
import sys
import threading

try:
    import _posixsubprocess
except ImportError:                         # not on POSIX
    _posixsubprocess = None


class NetworkAccessDenied(RuntimeError):
    """Raised when guarded code attempts network access or starts a process."""


_NET_EVENTS = {"socket.connect", "socket.sendto", "socket.sendmsg", "socket.bind", "socket.getaddrinfo",
               "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo"}
_PROCESS_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
                   "os.forkpty", "pty.spawn"}
_NET_SYMBOLS = {"connect", "connectx", "send", "sendto", "sendmsg", "sendmsg_x", "socket", "bind",
                "getaddrinfo", "gethostbyname", "gethostbyname2", "gethostbyaddr", "getnameinfo",
                "res_query", "res_search", "res_send"}

_lock = threading.Lock()
_active: list["NetworkGuard"] = []
_hooked = False
_saved_fork_exec = None


def _refuse(detail: str) -> None:
    with _lock:
        guards = list(_active)
    for g in guards:
        g.attempts.append(detail)
    raise NetworkAccessDenied(f"network or process access attempted during analysis: {detail}")


def _audit(event: str, args: tuple) -> None:
    if not _active:
        return
    if event in _NET_EVENTS or event in _PROCESS_EVENTS:
        _refuse(f"{event} {args!r}"[:200])
    if event == "ctypes.dlsym" and len(args) > 1 and str(args[1]) in _NET_SYMBOLS:
        _refuse(f"ctypes symbol {args[1]}")


def _blocked_fork_exec(*a, **k):            # noqa: ANN002, ANN003
    _refuse("process launch (multiprocessing / subprocess)")


class NetworkGuard:
    def __init__(self) -> None:
        self.attempts: list[str] = []

    def install(self) -> "NetworkGuard":
        global _hooked, _saved_fork_exec
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
            os.environ[var] = "1"
        with _lock:
            if not _hooked:
                sys.addaudithook(_audit)
                _hooked = True
            if not _active and _posixsubprocess is not None:
                _saved_fork_exec = _posixsubprocess.fork_exec
                _posixsubprocess.fork_exec = _blocked_fork_exec
            _active.append(self)
        return self

    def uninstall(self) -> None:
        global _saved_fork_exec
        with _lock:
            if self in _active:
                _active.remove(self)
            if not _active and _posixsubprocess is not None and _saved_fork_exec is not None:
                _posixsubprocess.fork_exec = _saved_fork_exec
                _saved_fork_exec = None

    def __enter__(self) -> "NetworkGuard":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.uninstall()
