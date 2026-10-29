"""
Whole-person pipeline: detection -> body tracks -> face-to-body mapping ->
identity (CONFIRMED / TEMPORARY / UNKNOWN) -> centring signal.

The pipeline only depends on three small interfaces, so the real backends
(OpenCV/ArcFace) and the scripted simulation backends are interchangeable:

    person_detector.detect(frame)            -> List[Detection]   (or None)
    face_detector.detect(frame)              -> List[FaceObs]
    identifier.identify(frame, face_obs)     -> IdMatch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .face_body_map import FaceBodyLink, map_faces_to_bodies
from .identity_state import Evidence, IdentityConfig, IdentityManager, IdState
from .person_tracker import PersonTrack, PersonTracker, hist_similarity


# ------------------------------------------------------------------ interfaces
@dataclass
class FaceObs:
    box: Tuple[float, float, float, float]
    raw: Any = None                       # backend-specific payload (e.g. 5-pt landmarks)


@dataclass
class IdMatch:
    name: Optional[str]
    similarity: float
    accepted: bool


# ---------------------------------------------------------------------- output
@dataclass
class TrackView:
    id: int
    box: Tuple[float, float, float, float]
    head: Tuple[float, float]
    is_target: bool
    visible: bool
    entangled: bool
    source: str


@dataclass
class FrameResult:
    t: float
    frame_index: int
    state: IdState
    conf: float
    target_track: Optional[int]
    target_point: Optional[Tuple[float, float]]    # head point of the bound body
    target_box: Optional[Tuple[float, float, float, float]]
    face_visible: bool                             # bound body has a face detected now
    verified: bool                                 # face verification succeeded this frame
    ambiguous: bool
    n_tracks: int
    tracks: List[TrackView] = field(default_factory=list)
    face_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class PipelineConfig:
    det_every: int = 3               # run the (slow) person detector every N frames
    verify_every: int = 8            # re-verify linked faces every N frames
    acquire_every: int = 2           # while UNKNOWN, try to identify this often
    mismatch_sim_ceiling: float = 0.20   # not accepted AND sim below this = clear mismatch
    identity: IdentityConfig = field(default_factory=IdentityConfig)


# -------------------------------------------------------------------- pipeline
class WholePersonPipeline:
    def __init__(
        self,
        target_name: str,
        face_detector,
        identifier,
        person_detector=None,
        cfg: Optional[PipelineConfig] = None,
        tracker: Optional[PersonTracker] = None,
    ):
        self.target_name = target_name
        self.face_detector = face_detector
        self.identifier = identifier
        self.person_detector = person_detector
        self.cfg = cfg or PipelineConfig()
        self.tracker = tracker or PersonTracker()
        self.identity = IdentityManager(self.cfg.identity)
        self.target_hist: Optional[np.ndarray] = None
        self.frame_index = 0

    # ------------------------------------------------------------------ helpers
    def _due(self, tr: PersonTrack) -> bool:
        cfg = self.cfg
        if self.identity.state == IdState.UNKNOWN:
            interval = cfg.acquire_every
        elif tr.id == self.identity.track_id and self.identity.state == IdState.TEMPORARY:
            interval = max(1, cfg.verify_every // 2)      # try harder to re-confirm
        else:
            interval = cfg.verify_every
        return (self.frame_index - tr.last_id_frame) >= interval

    def _is_clear_mismatch(self, m: IdMatch) -> bool:
        if m.accepted:
            return m.name != self.target_name
        return m.similarity < self.cfg.mismatch_sim_ceiling

    # ------------------------------------------------------------------ process
    def process(self, frame: np.ndarray, t: float) -> FrameResult:
        self.frame_index += 1
        tracker = self.tracker

        # 1) detect
        faces: List[FaceObs] = self.face_detector.detect(frame)
        tracker.predict(t)
        if self.person_detector is not None and (self.frame_index - 1) % self.cfg.det_every == 0:
            tracker.update(frame, self.person_detector.detect(frame), t)
        else:
            tracker.expire(t)

        # 2) map faces onto bodies; unmatched faces get a face-estimated body
        bodies = {tid: tuple(tr.box) for tid, tr in tracker.tracks.items()}
        links, unmatched = map_faces_to_bodies([f.box for f in faces], bodies)
        for ln in links:
            tracker.apply_face(frame, tracker.tracks[ln.track_id], faces[ln.face_index].box, t)
        for fi in unmatched:
            tr = tracker.spawn_from_face(frame, faces[fi].box, t)
            links.append(FaceBodyLink(fi, tr.id, 1.0))

        # 3) identify linked faces (rate limited)
        best_verified: Optional[Tuple[int, float]] = None
        mismatch: Set[int] = set()
        for ln in links:
            tr = tracker.tracks[ln.track_id]
            if not self._due(tr):
                continue
            m: IdMatch = self.identifier.identify(frame, faces[ln.face_index])
            tr.last_id_frame, tr.last_id_t = self.frame_index, t
            tr.last_id_name, tr.last_id_sim = m.name, m.similarity
            if m.accepted and m.name == self.target_name:
                if best_verified is None or m.similarity > best_verified[1]:
                    best_verified = (tr.id, m.similarity)
            elif self._is_clear_mismatch(m):
                mismatch.add(tr.id)

        # 4) feed the identity manager
        bound = self.identity.track_id
        bound_tr = tracker.tracks.get(bound) if bound is not None else None
        ambiguous = tracker.is_entangled(bound) if bound_tr is not None else False
        ev = Evidence(
            verified_track=best_verified[0] if best_verified else None,
            verified_sim=best_verified[1] if best_verified else 0.0,
            mismatch_tracks=mismatch,
            live_tracks=set(tracker.tracks.keys()),
            anchor_visible=bool(bound_tr.visible) if bound_tr is not None else True,
            ambiguous=ambiguous,
            appearance_sim=hist_similarity(self.target_hist, bound_tr.hist) if bound_tr is not None else None,
        )
        state, conf = self.identity.update(t, ev)

        # 5) learn the target's appearance only from clean, confirmed views
        tid = self.identity.track_id
        tgt = tracker.tracks.get(tid) if tid is not None else None
        if tgt is not None and state == IdState.CONFIRMED and tgt.hist is not None:
            if not tracker.is_entangled(tid):
                self.target_hist = (
                    tgt.hist.copy() if self.target_hist is None else 0.95 * self.target_hist + 0.05 * tgt.hist
                )

        # 6) package the result
        face_on_target = any(ln.track_id == tid for ln in links) if tid is not None else False
        views = [
            TrackView(
                id=k,
                box=tuple(float(x) for x in v.box),
                head=(float(v.head[0]), float(v.head[1])),
                is_target=(k == tid),
                visible=v.visible,
                entangled=tracker.is_entangled(k),
                source=v.source,
            )
            for k, v in tracker.tracks.items()
        ]
        return FrameResult(
            t=t,
            frame_index=self.frame_index,
            state=state,
            conf=float(conf),
            target_track=tid,
            target_point=(float(tgt.head[0]), float(tgt.head[1])) if tgt is not None else None,
            target_box=tuple(float(x) for x in tgt.box) if tgt is not None else None,
            face_visible=face_on_target,
            verified=best_verified is not None and best_verified[0] == tid,
            ambiguous=ambiguous,
            n_tracks=len(tracker.tracks),
            tracks=views,
            face_boxes=[tuple(float(x) for x in f.box) for f in faces],
        )
