from __future__ import annotations

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TextIO

import pyte


class Tee:
    def __init__(self, terminal_stream: TextIO, log_stream: TextIO) -> None:
        self.terminal_stream = terminal_stream
        self.log_stream = log_stream
        self.log_renderer = TerminalLogRenderer()

    def write(self, text: str) -> int:
        self.terminal_stream.write(text)
        self.terminal_stream.flush()

        self.log_stream.write(self.log_renderer.feed(text))
        self.log_stream.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal_stream.flush()
        self.log_stream.flush()

    def flush_pending(self) -> None:
        self.log_stream.write(self.log_renderer.flush())
        self.log_stream.flush()


class TerminalLogRenderer:
    def __init__(self, columns: int = 4096) -> None:
        self.columns = columns
        self.screen = pyte.Screen(columns, 1)
        self.stream = pyte.Stream(self.screen)

    def feed(self, text: str) -> str:
        rendered_lines: list[str] = []
        parts = text.split("\n")

        for part in parts[:-1]:
            self.stream.feed(part)
            rendered_lines.append(self.current_line())
            self.reset_line()

        self.stream.feed(parts[-1])
        return "".join(f"{line}\n" for line in rendered_lines)

    def flush(self) -> str:
        if self.screen.cursor.x == 0 and not self.current_line():
            return ""

        line = self.current_line()
        self.reset_line()
        return f"{line}\n"

    def current_line(self) -> str:
        return self.screen.display[0].rstrip()

    def reset_line(self) -> None:
        self.screen.reset()
        self.stream = pyte.Stream(self.screen)


def clean_log_text(text: str) -> str:
    renderer = TerminalLogRenderer()
    return renderer.feed(text) + renderer.flush()


@contextmanager
def tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            try:
                yield
            finally:
                tee.flush_pending()
