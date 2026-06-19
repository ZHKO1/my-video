"""CLI output formatting — progress display, status messages, error formatting."""

import sys
import threading


_lock = threading.Lock()


def info(msg: str) -> None:
    with _lock:
        print(f"  {msg}", file=sys.stderr)


def success(msg: str) -> None:
    with _lock:
        print(f"\u2713 {msg}", file=sys.stderr)


def error(msg: str) -> None:
    with _lock:
        print(f"\u2717 Error: {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    with _lock:
        print(f"! Warning: {msg}", file=sys.stderr)


def hint(msg: str) -> None:
    """Print a hint message (e.g. how to fix a config issue)."""
    with _lock:
        print(f"  {msg}", file=sys.stderr)


class ProgressLine:
    """Simple single-line progress indicator for CLI."""

    SPINNER = [
        "\u280b",
        "\u2819",
        "\u2839",
        "\u2838",
        "\u283c",
        "\u2834",
        "\u2826",
        "\u2827",
        "\u2807",
        "\u280f",
    ]

    def __init__(self, message: str = ""):
        self.message = message
        self.percent: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = 0
        self._is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def start(self) -> "ProgressLine":
        self._stop.clear()
        if self._is_tty:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def update(self, percent: int, message: str = "") -> None:
        self.percent = percent
        if message:
            self.message = message

    def _cleanup(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._is_tty:
            sys.stderr.write("\r\033[K")

    def finish(self, message: str = "") -> None:
        self._cleanup()
        if message:
            success(message)

    def fail(self, message: str = "") -> None:
        self._cleanup()
        if message:
            error(message)

    def _spin(self) -> None:
        while not self._stop.is_set():
            char = self.SPINNER[self._frame % len(self.SPINNER)]
            if self.percent is not None:
                line = f"\r{char} {self.message} [{self.percent}%]"
            else:
                line = f"\r{char} {self.message}"
            sys.stderr.write(f"{line}\033[K")
            sys.stderr.flush()
            self._frame += 1
            self._stop.wait(0.1)
