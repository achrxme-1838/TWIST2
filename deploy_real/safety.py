"""Backend-agnostic safety state shared by sim and real controllers.

Holds PD gain multipliers (kp_scale / kd_scale) that ramp from the chosen
initial_scale up to 1.0 via keypresses.

Key handling is split across two threads to keep the control loop deterministic:
    handle_keycode  — called from the input thread (mujoco viewer event thread,
                      stdin reader, etc). Only enqueues an event; cheap; never
                      touches PD state or stdout. Returns immediately.
    drain           — called from the control thread (main loop) once per tick.
                      Pulls queued events and applies them — this is the only
                      place PD state mutates and the only place that prints,
                      so the input thread never contends with the loop on the
                      GIL or on stdout.
"""
from __future__ import annotations

import select
import sys
import termios
import threading
import tty
from queue import SimpleQueue


class SafetyController:
    KEY_BINDINGS_HELP = (
        "[Safety] Keys (focus the viewer/terminal):\n"
        "         2..9 -> 20%..90% PD gain     X -> 100% (FULL)"
    )

    def __init__(self, initial_scale: float = 0.1):
        self.kp_scale = float(initial_scale)
        self.kd_scale = float(initial_scale)
        self._events: SimpleQueue = SimpleQueue()
        print(f"[Safety] Initial PD gain scale: {self.kp_scale*100:.0f}%")
        print(self.KEY_BINDINGS_HELP)

    def handle_keycode(self, keycode: int):
        """Input-thread entry point. Maps keycode to an event and enqueues it.
        No mutation, no I/O — keeps the input thread tiny."""
        try:
            ch = chr(int(keycode))
        except (ValueError, TypeError):
            return
        if ch in "23456789":
            self._events.put(("set_scale", int(ch) * 0.1, ""))
        elif ch in ("X", "x"):
            self._events.put(("set_scale", 1.0, "(FULL)"))

    def drain(self):
        """Control-thread entry point. Apply queued events. Call once per tick."""
        while not self._events.empty():
            kind, val, label = self._events.get_nowait()
            if kind == "set_scale":
                self.kp_scale = float(val)
                self.kd_scale = float(val)
                suffix = f" {label}" if label else ""
                print(f"[Safety] PD gain scale -> {val*100:.0f}%{suffix}")


class StdinKeyListener:
    """Non-blocking stdin reader for headless (real-robot) deployments.

    Puts the controlling tty into cbreak mode (line buffering off, but Ctrl+C
    still raises SIGINT) and runs a daemon thread that polls stdin via
    `select.select` so the main loop is never blocked. Each keystroke is
    forwarded to `callback(keycode: int)` — wire it to
    `SafetyController.handle_keycode`.

    Always pair `start()` with `stop()` (or use as a context manager); skipping
    `stop()` leaves the shell in cbreak mode and the user has to `reset` it.
    """

    def __init__(self, callback):
        self._callback = callback
        self._stop = threading.Event()
        self._thread = None
        self._old_settings = None
        self._fd = None

    def start(self):
        if not sys.stdin.isatty():
            print("[Safety] stdin is not a tty; keyboard input disabled.")
            return
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="safety-key"
        )
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                ch = sys.stdin.read(1)
                if ch:
                    self._callback(ord(ch))

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self._old_settings is not None and self._fd is not None:
            termios.tcsetattr(
                self._fd, termios.TCSADRAIN, self._old_settings
            )
            self._old_settings = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
