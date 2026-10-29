"""
Three-state identity model with exponential confidence decay.

States
------
CONFIRMED  A face verified as the target very recently, on the body we are
           following. Confidence is high.
TEMPORARY  The face is no longer verifiable (head turn, occlusion, crossing...)
           but the body we were following is still around. We *believe* it is
           still the target, and our belief fades exponentially with time.
UNKNOWN    We have no body anchor, or the belief has decayed too far. We do
           not claim to know who the target is. Only a fresh face
           verification can leave this state.

Honesty rules
-------------
* Only a face verification can raise confidence to 1.0 (=> CONFIRMED).
  Body appearance may slow or speed up the decay, but it can never confirm.
* While the followed body is entangled with another person (path crossing),
  confidence is capped, because the track may have jumped to the wrong body.
* A visible face that does NOT match the target is negative evidence and
  cuts the confidence immediately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple


class IdState(Enum):
    CONFIRMED = "CONFIRMED"
    TEMPORARY = "TEMPORARY"
    UNKNOWN = "UNKNOWN"


@dataclass
class IdentityConfig:
    # Exponential decay: conf(t) = conf0 * exp(-(t - t0) / tau_eff)
    tau: float = 4.0                  # seconds
    confirmed_thresh: float = 0.80    # conf >= this  -> CONFIRMED
    unknown_thresh: float = 0.30      # conf <  this  -> UNKNOWN

    ambiguity_cap: float = 0.50       # max conf while entangled with another person
    # A face verified this recently is not 'stale' and escapes the overlap cap.
    # Must exceed the re-verification interval (verify_every / fps ~ 0.27 s at
    # 30 fps), otherwise the state flickers CONFIRMED <-> TEMPORARY during
    # overlaps. Longer = smoother but a longer blind spot if bodies swap.
    fresh_window: float = 0.35
    ambiguity_tau_scale: float = 0.50 # decay twice as fast while entangled
    coasting_tau_scale: float = 0.60  # decay faster when the body is not seen
    mismatch_factor: float = 0.55     # conf *= this per visible non-matching face check

    # Appearance can modulate decay speed: tau_eff = tau * (1 + appearance_gain * g)
    # where g in [-1, 1] comes from the similarity (neutral similarity -> g = 0).
    appearance_gain: float = 0.5
    appearance_neutral: float = 0.70


@dataclass
class Evidence:
    """What the vision pipeline observed in one frame."""

    verified_track: Optional[int] = None       # track whose face matched the target
    verified_sim: float = 0.0
    mismatch_tracks: Set[int] = field(default_factory=set)  # faces checked, not the target
    live_tracks: Set[int] = field(default_factory=set)      # tracks that still exist
    anchor_visible: bool = True                # bound track matched a detection/face now
    ambiguous: bool = False                    # bound track entangled with another person
    appearance_sim: Optional[float] = None     # body-appearance similarity to the target


@dataclass
class Transition:
    t: float
    old: IdState
    new: IdState
    reason: str


class IdentityManager:
    """Maintains who the target is, and how sure we are."""

    def __init__(self, cfg: Optional[IdentityConfig] = None):
        self.cfg = cfg or IdentityConfig()
        self.state: IdState = IdState.UNKNOWN
        self.conf: float = 0.0
        self.track_id: Optional[int] = None
        self.transitions: List[Transition] = []
        self.last_verified_t: Optional[float] = None
        self._t_prev: Optional[float] = None

    # ------------------------------------------------------------------ helpers
    def _appearance_g(self, sim: Optional[float]) -> float:
        if sim is None:
            return 0.0
        n = self.cfg.appearance_neutral
        if sim >= n:
            g = (sim - n) / max(1e-6, 1.0 - n)
        else:
            g = (sim - n) / max(1e-6, n)
        return max(-1.0, min(1.0, g))

    def effective_tau(self, ev: Evidence) -> float:
        cfg = self.cfg
        tau = cfg.tau * (1.0 + cfg.appearance_gain * self._appearance_g(ev.appearance_sim))
        if ev.ambiguous:
            tau *= cfg.ambiguity_tau_scale
        if not ev.anchor_visible:
            tau *= cfg.coasting_tau_scale
        return max(1e-3, tau)

    def _set_state(self, t: float, new: IdState, reason: str) -> None:
        if new != self.state:
            self.transitions.append(Transition(t, self.state, new, reason))
            self.state = new

    # ------------------------------------------------------------------- update
    def update(self, t: float, ev: Evidence) -> Tuple[IdState, float]:
        cfg = self.cfg
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t
        reason = "decay"

        # 1) Exponential decay (only meaningful if we are bound to a body).
        if self.track_id is not None:
            self.conf *= math.exp(-dt / self.effective_tau(ev))

            # 2) Entangled with another person: the track may have jumped.
            #    Only *stale* belief is capped; a face verified moments ago on
            #    this very body is still direct evidence (avoids state flicker).
            stale = self.last_verified_t is None or (t - self.last_verified_t) > cfg.fresh_window
            if ev.ambiguous and stale and self.conf > cfg.ambiguity_cap:
                self.conf = cfg.ambiguity_cap
                reason = "ambiguous (path crossing)"

            # 3) The body we follow vanished entirely.
            if self.track_id not in ev.live_tracks:
                self.track_id = None
                self.conf = 0.0
                reason = "body track lost"

            # 4) A visible face on our body that is NOT the target.
            elif self.track_id in ev.mismatch_tracks:
                self.conf *= cfg.mismatch_factor
                reason = "visible face does not match target"

        # 5) Positive evidence: the only way to reach full confidence.
        if ev.verified_track is not None:
            if self.track_id != ev.verified_track:
                reason = "face verified (re-bound to new body)"
            else:
                reason = "face verified"
            self.track_id = ev.verified_track
            self.conf = 1.0
            self.last_verified_t = t

        # 6) Map confidence to a state.
        if self.track_id is None:
            self.conf = 0.0
            self._set_state(t, IdState.UNKNOWN, reason)
        elif self.conf >= cfg.confirmed_thresh:
            self._set_state(t, IdState.CONFIRMED, reason)
        elif self.conf >= cfg.unknown_thresh:
            self._set_state(t, IdState.TEMPORARY, reason)
        else:
            # Belief decayed too far: unbind, we no longer claim to know.
            self.track_id = None
            self.conf = 0.0
            self._set_state(t, IdState.UNKNOWN, "confidence decayed below threshold")

        return self.state, self.conf

    def reset(self) -> None:
        self.state = IdState.UNKNOWN
        self.conf = 0.0
        self.track_id = None
        self._t_prev = None
        self.last_verified_t = None
