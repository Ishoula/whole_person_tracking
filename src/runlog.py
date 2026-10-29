"""CSV run log shared by the webcam script, the simulator and the evaluator."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from .centering import CenterCommand
from .whole_person import FrameResult

COLUMNS = [
    "t", "frame", "scenario", "wrong", "state", "conf", "target_track",
    "n_tracks", "face_visible", "verified", "ambiguous",
    "tx", "ty", "err_x", "err_y", "horizontal", "vertical",
    "pan_speed", "tilt_speed", "gain", "active", "dead_zone",
]


class RunLogger:
    def __init__(self, path: str, dead_zone: float):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNS)
        self.dead_zone = dead_zone
        self.path = path

    def write(self, res: FrameResult, cmd: CenterCommand, scenario: str, wrong: int) -> None:
        tx, ty = res.target_point if res.target_point is not None else ("", "")
        self._w.writerow([
            f"{res.t:.4f}", res.frame_index, scenario, int(wrong), res.state.value,
            f"{res.conf:.4f}", "" if res.target_track is None else res.target_track,
            res.n_tracks, int(res.face_visible), int(res.verified), int(res.ambiguous),
            tx if tx == "" else f"{tx:.1f}", ty if ty == "" else f"{ty:.1f}",
            f"{cmd.error_x:.4f}", f"{cmd.error_y:.4f}", cmd.horizontal, cmd.vertical,
            f"{cmd.pan_speed:.4f}", f"{cmd.tilt_speed:.4f}", f"{cmd.gain:.3f}",
            int(cmd.active), self.dead_zone,
        ])

    def close(self) -> None:
        self._f.close()
