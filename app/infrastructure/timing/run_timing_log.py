"""
Un fichero de log por ejecución con prefijo y timestamp en el nombre.
Registra duraciones en ms de pasos anidadas (time.perf_counter).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.core.config import get_settings


def _utc_stamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]


class RunTimingLog:
    """Escribe `TIMING_LOGS_PATH/<prefix>_<utc>.log` y acumula líneas por paso."""

    def __init__(self, prefix: str) -> None:
        settings = get_settings()
        base = Path(settings.timing_logs_path)
        base.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix)[:64]
        self.path = base / f"{safe}_{_utc_stamp_for_filename()}.log"
        self._t0 = time.perf_counter()
        self.path.write_text(
            "\n".join(
                [
                    f"# run={prefix}",
                    f"# started_utc={datetime.now(timezone.utc).isoformat()}",
                    f"# log_file={self.path.as_posix()}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def append_line(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

    def meta(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self.append_line(f"meta {k}={v}")

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.append_line(f"step {name} | {dt_ms:.2f} ms")

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def finish(self, **summary: Any) -> None:
        for k, v in summary.items():
            self.append_line(f"summary {k}={v}")
        self.append_line(f"total_wall_ms | {self.total_ms():.2f} ms")
