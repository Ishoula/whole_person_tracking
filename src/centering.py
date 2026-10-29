"""
Target centring logic and a virtual pan/tilt camera (no hardware needed).

CenteringController turns the target's position error into pan/tilt commands,
respecting the identity state:
  CONFIRMED  full gain
  TEMPORARY  gain scaled down with confidence (we are less sure who this is)
  UNKNOWN    no motion at all

Command words match the MQTT controller from the previous weeks
(LEFT/RIGHT/CENTER, UP/DOWN/CENTER): error_x < 0 means the target is left of
centre, error_y < 0 means it is above centre (camera must tilt UP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .identity_state import IdState


@dataclass
class CenterCommand:
    error_x: float = 0.0          # normalised, -1..1 (relative to viewport half-size)
    error_y: float = 0.0
    horizontal: str = "CENTER"    # LEFT / RIGHT / CENTER
    vertical: str = "CENTER"      # UP / DOWN / CENTER
    pan_speed: float = 0.0        # -1..1 (negative = left)
    tilt_speed: float = 0.0       # -1..1 (negative = up)
    gain: float = 0.0
    active: bool = False


def normalized_error(point, center, half_size) -> Tuple[float, float]:
    return (
        float((point[0] - center[0]) / half_size[0]),
        float((point[1] - center[1]) / half_size[1]),
    )


class CenteringController:
    def __init__(
        self,
        dead_zone: float = 0.07,
        exit_ratio: float = 0.6,
        kp: float = 1.2,
        ema_alpha: float = 0.35,
        confirmed_thresh: float = 0.80,
    ):
        self.dead_zone = dead_zone
        self.exit_ratio = exit_ratio
        self.kp = kp
        self.ema_alpha = ema_alpha
        self.confirmed_thresh = confirmed_thresh
        self._ex: Optional[float] = None
        self._ey: Optional[float] = None
        self._moving = [False, False]

    def reset(self) -> None:
        self._ex = self._ey = None
        self._moving = [False, False]

    def _axis(self, e: float, idx: int) -> float:
        """Dead zone with hysteresis: start moving at dz, stop at dz*exit_ratio."""
        thr = self.dead_zone * (self.exit_ratio if self._moving[idx] else 1.0)
        if abs(e) <= thr:
            self._moving[idx] = False
            return 0.0
        self._moving[idx] = True
        return float(np.clip(self.kp * e, -1.0, 1.0))

    def step(self, error: Optional[Tuple[float, float]], state: IdState, conf: float) -> CenterCommand:
        if error is None or state == IdState.UNKNOWN:
            self.reset()
            return CenterCommand()

        ex, ey = error
        a = self.ema_alpha
        self._ex = ex if self._ex is None else a * ex + (1 - a) * self._ex
        self._ey = ey if self._ey is None else a * ey + (1 - a) * self._ey

        if state == IdState.CONFIRMED:
            gain = 1.0
        else:  # TEMPORARY: less sure -> move less
            gain = float(np.clip(conf / self.confirmed_thresh, 0.0, 1.0))

        pan = self._axis(self._ex, 0) * gain
        tilt = self._axis(self._ey, 1) * gain

        horizontal = "CENTER" if pan == 0.0 else ("LEFT" if pan < 0 else "RIGHT")
        vertical = "CENTER" if tilt == 0.0 else ("UP" if tilt < 0 else "DOWN")
        return CenterCommand(self._ex, self._ey, horizontal, vertical, pan, tilt, gain, True)


class VirtualCamera:
    """A digital pan/tilt window inside the real frame.

    Stands in for the stepper-driven camera: the viewport moves by
    pan_speed/tilt_speed and saturates at the frame edge, just like a motor
    hitting its soft limit.
    """

    def __init__(self, frame_w: int, frame_h: int, zoom: float = 1.6, max_speed: float = 0.9):
        self.W, self.H = frame_w, frame_h
        self.vw, self.vh = frame_w / zoom, frame_h / zoom
        self.cx, self.cy = frame_w / 2.0, frame_h / 2.0
        self.max_speed = max_speed       # frame-widths (heights) per second at speed 1.0

    @property
    def half_size(self) -> Tuple[float, float]:
        return (self.vw / 2.0, self.vh / 2.0)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    def error_to(self, point) -> Tuple[float, float]:
        return normalized_error(point, self.center, self.half_size)

    def apply(self, cmd: CenterCommand, dt: float) -> None:
        self.cx += cmd.pan_speed * self.max_speed * self.W * dt
        self.cy += cmd.tilt_speed * self.max_speed * self.H * dt
        self.cx = float(np.clip(self.cx, self.vw / 2.0, self.W - self.vw / 2.0))
        self.cy = float(np.clip(self.cy, self.vh / 2.0, self.H - self.vh / 2.0))

    def at_limit(self) -> Tuple[bool, bool]:
        return (
            self.cx <= self.vw / 2.0 + 1e-3 or self.cx >= self.W - self.vw / 2.0 - 1e-3,
            self.cy <= self.vh / 2.0 + 1e-3 or self.cy >= self.H - self.vh / 2.0 - 1e-3,
        )

    def rect(self) -> Tuple[int, int, int, int]:
        return (
            int(self.cx - self.vw / 2.0),
            int(self.cy - self.vh / 2.0),
            int(self.cx + self.vw / 2.0),
            int(self.cy + self.vh / 2.0),
        )

    def render(self, frame: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.rect()
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        return cv2.resize(crop, (frame.shape[1], frame.shape[0]))
