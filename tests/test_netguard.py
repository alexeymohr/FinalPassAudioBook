"""The network guard refuses and counts every route the audit found open. Loopback only."""
from __future__ import annotations

import _socket
import ctypes
import multiprocessing as mp
import os
import socket
import subprocess

import pytest

from finalpass_audiobook.netguard import NetworkAccessDenied, NetworkGuard


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
    "subprocess": lambda: subprocess.run(["true"], check=False),
    "os.system": lambda: os.system("true"),          # fixed literal: proves the guard refuses this route
    "multiprocessing spawn": lambda: mp.get_context("spawn").Process(target=_child).start(),
    "ctypes libc connect": lambda: ctypes.CDLL(None).connect,
}


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
    assert subprocess.run(["true"], check=False).returncode == 0
