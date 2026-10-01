"""The network guard refuses and counts every route the audit found open. Loopback only."""
from __future__ import annotations

import _ctypes
import _socket
import ctypes
import multiprocessing as mp
import os
import socket
import subprocess
import sys

import pytest

from finalpass_audiobook.netguard import NetworkAccessDenied, NetworkGuard
from synth import POSIX


def _child() -> None:        # pragma: no cover - never allowed to start
    pass


ROUTES = {
    "create_connection": lambda: socket.create_connection(("127.0.0.1", 9), timeout=0.2),
    "gethostbyname": lambda: socket.gethostbyname("localhost"),
    "getaddrinfo": lambda: socket.getaddrinfo("localhost", 80),
    "getnameinfo": lambda: socket.getnameinfo(("127.0.0.1", 80), 0),
    "udp sendto": lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("127.0.0.1", 9)),
    "_socket connect": lambda: _socket.socket().connect(("127.0.0.1", 9)),
    "bind": lambda: socket.socket().bind(("127.0.0.1", 0)),
    "subprocess": lambda: subprocess.run([sys.executable, "-c", "pass"], check=False),
    "os.system": lambda: os.system("exit 0"),        # fixed literal: proves the guard refuses this route
    "multiprocessing spawn": lambda: mp.get_context("spawn").Process(target=_child).start(),
    "socket creation": lambda: socket.socket(),
    "ctypes libcurl": lambda: ctypes.CDLL("libcurl.4.dylib"),
    "ctypes libresolv": lambda: ctypes.CDLL("/usr/lib/libresolv.dylib"),
    "ctypes winsock": lambda: ctypes.CDLL("ws2_32.dll"),
}
if POSIX:                                            # the C library's own symbols (no such handle on Windows)
    _libc = lambda name: _ctypes.dlsym(_ctypes.dlopen(None), name)   # noqa: E731
    ROUTES.update({
        "ctypes libc connect": lambda: ctypes.CDLL(None).connect,
        "ctypes libc system": lambda: ctypes.CDLL(None).system,
        "ctypes lookup by handle": lambda: _libc("connect"),
        "ctypes curl function": lambda: _libc("curl_easy_perform"),
        "ctypes execvP": lambda: _libc("execvP"),
        "ctypes async lookup": lambda: _libc("getaddrinfo_async_start"),
        "ctypes DNS service": lambda: _libc("DNSServiceGetAddrInfo"),
        "ctypes res_query": lambda: _libc("res_query"),
        "ctypes connectx": lambda: _libc("connectx"),
        "ctypes Network.framework": lambda: _libc("nw_connection_create"),
    })
else:                                                # Windows: what a process or a download would look up
    ROUTES.update({
        "ctypes CreateProcessW": lambda: ctypes.windll.kernel32.CreateProcessW,
        "ctypes WSAConnect": lambda: ctypes.CDLL("kernel32").WSAConnect,
    })


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_every_route_is_refused_and_counted(name: str) -> None:
    with NetworkGuard() as g:
        with pytest.raises(NetworkAccessDenied):
            ROUTES[name]()
    assert len(g.attempts) >= 1


def test_nested_guards_both_count_and_the_outer_stays_active() -> None:
    with NetworkGuard() as outer:
        with NetworkGuard() as inner:
            with pytest.raises(NetworkAccessDenied):
                socket.gethostbyname("localhost")
        with pytest.raises(NetworkAccessDenied):          # inner exited; outer still guards
            socket.gethostbyname("localhost")
    assert len(inner.attempts) == 1 and len(outer.attempts) == 2


def test_nothing_is_blocked_outside_a_guard() -> None:
    with NetworkGuard():
        pass
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.close()
    assert subprocess.run([sys.executable, "-c", "pass"], check=False).returncode == 0


def test_the_outer_guard_still_refuses_a_process_after_an_inner_guard_ends() -> None:
    with NetworkGuard() as outer:
        with NetworkGuard():
            pass
        with pytest.raises(NetworkAccessDenied):
            subprocess.run([sys.executable, "-c", "pass"], check=False)
    assert outer.attempts


def test_every_file_is_analysed_inside_the_guard(tmp_path, monkeypatch) -> None:        # noqa: ANN001
    """A check that reached for the network is refused, counted in the report, and only its file is skipped."""
    from finalpass_audiobook import run as run_mod

    def phone_home(path, opts, model=None, stage=None):        # noqa: ANN001, ANN202
        socket.getaddrinfo("localhost", 80)

    monkeypatch.setattr(run_mod, "analyze_file", phone_home)
    report = run_mod.run([tmp_path / "a.wav", tmp_path / "b.wav"], run_mod.RunOptions(truncation=False))
    assert report.network_attempts == 2
    assert all("NetworkAccessDenied" in fr.notes[0] for fr in report.files)
