"""
Scripted scenario simulator.

Renders synthetic frames (coloured rectangles + head circles) and supplies
scripted person/face/identity backends, so the *logic* (identity states,
decay, crossing handling, centring) can be tested without a webcam.

This validates the logic. It says nothing about how well HOG/Haar/ArcFace
perform on a real webcam; that is what the live evaluation is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .person_tracker import Detection, iou
from .whole_person import FaceObs, IdMatch

W, H = 640, 480
BODY_W, BODY_H = 130, 380
Y_OFFSET = 90.0


@dataclass
class SimPerson:
    name: str
    shirt: Tuple[int, int, int]                       # BGR
    depth: int                                        # higher = closer to camera (drawn last)
    pos: Callable[[float], Tuple[float, float]]       # t -> (cx, top_y)
    face_visible: Callable[[float], bool] = lambda t: True
    hidden: Callable[[float], bool] = lambda t: False  # fully hidden (behind a pillar)


@dataclass
class Scenario:
    name: str
    duration: float
    persons: List[SimPerson]
    label: Callable[[float], str]
    pillar: Optional[Tuple[int, int]] = None          # (x1, x2) drawn as an obstacle


def body_box(cx: float, top: float) -> Tuple[float, float, float, float]:
    return (cx - BODY_W / 2, top, cx + BODY_W / 2, top + BODY_H)


def face_box(cx: float, top: float) -> Tuple[float, float, float, float]:
    return (cx - 26, top + 6, cx + 26, top + 62)


class World:
    """Holds the current simulated time and answers 'what is visible?'."""

    def __init__(self, scenario: Scenario, seed: int = 0, body_recall: float = 0.92, face_recall: float = 0.95):
        self.sc = scenario
        self.t = 0.0
        self.rng = np.random.default_rng(seed)
        self.body_recall = body_recall
        self.face_recall = face_recall

    def person_state(self, p: SimPerson):
        cx, top = p.pos(self.t)
        return cx, top

    def occluded_fraction(self, p: SimPerson) -> float:
        """Fraction of p's body box covered by a closer person's body box."""
        cx, top = p.pos(self.t)
        b = body_box(cx, top)
        area = (b[2] - b[0]) * (b[3] - b[1])
        covered = 0.0
        for q in self.sc.persons:
            if q is p or q.depth <= p.depth or q.hidden(self.t):
                continue
            qb = body_box(*q.pos(self.t))
            ix = max(0.0, min(b[2], qb[2]) - max(b[0], qb[0]))
            iy = max(0.0, min(b[3], qb[3]) - max(b[1], qb[1]))
            covered = max(covered, ix * iy / area)
        return covered

    def body_detected(self, p: SimPerson) -> bool:
        if p.hidden(self.t):
            return False
        if self.occluded_fraction(p) > 0.55:
            return False
        return bool(self.rng.random() < self.body_recall)

    def face_detected(self, p: SimPerson) -> bool:
        if p.hidden(self.t) or not p.face_visible(self.t):
            return False
        if self.occluded_fraction(p) > 0.25:
            return False
        return bool(self.rng.random() < self.face_recall)

    def render(self) -> np.ndarray:
        img = np.full((H, W, 3), (70, 70, 70), np.uint8)
        for p in sorted(self.sc.persons, key=lambda q: q.depth):
            if p.hidden(self.t):
                continue
            cx, top = p.pos(self.t)
            x1, y1, x2, y2 = (int(v) for v in body_box(cx, top))
            cv2.rectangle(img, (x1, y1 + 60), (x2, y1 + 240), p.shirt, -1)       # torso
            cv2.rectangle(img, (x1 + 15, y1 + 240), (x2 - 15, y2), (90, 60, 40), -1)  # legs
            cv2.circle(img, (int(cx), int(top + 34)), 28, (150, 170, 220), -1)   # head
            if not p.face_visible(self.t):
                cv2.circle(img, (int(cx), int(top + 34)), 28, (30, 30, 40), -1)  # back of head
        if self.sc.pillar is not None:
            cv2.rectangle(img, (self.sc.pillar[0], 0), (self.sc.pillar[1], H), (40, 90, 40), -1)
        return img


# ------------------------------------------------------------ scripted backends
class ScriptedPersonDetector:
    def __init__(self, world: World, noise: float = 3.0):
        self.w, self.noise = world, noise

    def detect(self, frame) -> List[Detection]:
        out = []
        for p in self.w.sc.persons:
            if not self.w.body_detected(p):
                continue
            b = np.array(body_box(*p.pos(self.w.t))) + self.w.rng.normal(0, self.noise, 4)
            out.append(Detection(tuple(float(x) for x in b), 1.0, "sim"))
        return out


class ScriptedFaceDetector:
    def __init__(self, world: World, noise: float = 2.0):
        self.w, self.noise = world, noise

    def detect(self, frame) -> List[FaceObs]:
        out = []
        for p in self.w.sc.persons:
            if not self.w.face_detected(p):
                continue
            b = np.array(face_box(*p.pos(self.w.t))) + self.w.rng.normal(0, self.noise, 4)
            out.append(FaceObs(tuple(float(x) for x in b), raw=p.name))
        return out


class ScriptedIdentifier:
    """Target scores ~0.8 and is accepted; everyone else ~0.1 and is rejected."""

    def __init__(self, world: World, target: str):
        self.w, self.target = world, target

    def identify(self, frame, face: FaceObs) -> IdMatch:
        rng = self.w.rng
        if face.raw == self.target:
            sim = float(np.clip(rng.normal(0.80, 0.05), 0, 1))
            return IdMatch(self.target, sim, sim > 0.5)
        sim = float(np.clip(rng.normal(0.10, 0.05), 0, 1))
        return IdMatch(None, sim, False)


# ------------------------------------------------------------------- scenarios
def _static(cx: float, top: float = 60.0):
    return lambda t: (cx, top)


def _linear(x0: float, x1: float, t0: float, t1: float, top: float = 60.0):
    def f(t):
        a = np.clip((t - t0) / max(1e-6, t1 - t0), 0.0, 1.0)
        return (x0 + a * (x1 - x0), top)
    return f


RED, BLUE, GREEN = (40, 40, 200), (200, 80, 40), (60, 170, 60)
TARGET = "A"


def make_scenarios() -> Dict[str, Scenario]:
    sc: Dict[str, Scenario] = {}

    # 1) head turn, short: face hidden 3-6 s, body still visible
    sc["head_turn_short"] = Scenario(
        "head_turn_short", 10.0,
        [SimPerson("A", RED, 1, _static(300, 40), face_visible=lambda t: not (3.0 <= t < 6.0)),
         SimPerson("B", BLUE, 0, _static(540, 70))],
        label=lambda t: "head_turn" if 3.0 <= t < 6.0 else "normal",
    )

    # 2) head turn, long: face hidden 3-11 s (long enough for belief to decay away)
    sc["head_turn_long"] = Scenario(
        "head_turn_long", 14.0,
        [SimPerson("A", RED, 1, _static(300, 40), face_visible=lambda t: not (3.0 <= t < 11.0)),
         SimPerson("B", BLUE, 0, _static(540, 70))],
        label=lambda t: "head_turn" if 3.0 <= t < 11.0 else "normal",
    )

    # 3) occlusion: A walks behind a pillar and out the other side
    sc["occlusion"] = Scenario(
        "occlusion", 12.0,
        [SimPerson("A", RED, 1, _linear(200, 560, 1.0, 11.0, 40),
                   hidden=lambda t: 4.2 <= t < 6.4),
         SimPerson("B", BLUE, 0, _static(60, 90))],
        label=lambda t: "occlusion" if 4.2 <= t < 6.4 else "normal",
        pillar=(330, 400),
    )

    # 4) path crossing, different clothes: B passes in front of A
    sc["crossing"] = Scenario(
        "crossing", 12.0,
        [SimPerson("A", RED, 0, _linear(160, 480, 2.0, 10.0, 50)),
         SimPerson("B", BLUE, 1, _linear(480, 160, 2.0, 10.0, 50))],
        label=lambda t: "crossing" if 4.5 <= t < 7.5 else "normal",
    )

    # 5) path crossing, similar clothes: appearance cannot help
    sc["crossing_similar"] = Scenario(
        "crossing_similar", 12.0,
        [SimPerson("A", RED, 0, _linear(160, 480, 2.0, 10.0, 50)),
         SimPerson("B", (45, 45, 190), 1, _linear(480, 160, 2.0, 10.0, 50))],
        label=lambda t: "crossing" if 4.5 <= t < 7.5 else "normal",
    )

    # 5b) swap stress: same depth, both always detected, similar clothes, faces
    #     hidden while they pass through each other (nothing to tell them apart)
    both_hidden = lambda t: not (5.2 <= t < 7.0)
    sc["crossing_swap"] = Scenario(
        "crossing_swap", 12.0,
        [SimPerson("A", RED, 0, _linear(160, 480, 2.0, 10.0, 50), face_visible=both_hidden),
         SimPerson("B", (45, 45, 190), 0, _linear(480, 160, 2.0, 10.0, 50), face_visible=both_hidden)],
        label=lambda t: "crossing" if 4.5 <= t < 7.5 else "normal",
    )

    # 5c) bounce: they walk into each other, overlap, then both reverse; with
    #     similar clothes and no faces the bodies are indistinguishable.
    def a_path(t):
        if t < 2.0: return (160, 50)
        if t < 5.0: return (160 + (t - 2.0) / 3.0 * 150, 50)
        if t < 7.0: return (310, 50)
        return (310 - min(1.0, (t - 7.0) / 3.0) * 150, 50)
    def b_path(t):
        if t < 2.0: return (480, 50)
        if t < 5.0: return (480 - (t - 2.0) / 3.0 * 150, 50)
        if t < 7.0: return (330, 50)
        return (330 + min(1.0, (t - 7.0) / 3.0) * 150, 50)
    sc["crossing_bounce"] = Scenario(
        "crossing_bounce", 12.0,
        [SimPerson("A", RED, 0, a_path, face_visible=both_hidden),
         SimPerson("B", (45, 45, 190), 0, b_path, face_visible=both_hidden)],
        label=lambda t: "crossing" if 4.5 <= t < 7.5 else "normal",
    )

    # 6) vertical motion: A sits, then stands (head goes up) then sits again
    def vert(t):
        a = 0.0
        if 2.0 <= t < 4.0:
            a = (t - 2.0) / 2.0
        elif 4.0 <= t < 7.0:
            a = 1.0
        elif 7.0 <= t < 9.0:
            a = 1.0 - (t - 7.0) / 2.0
        return (320, 140 - 100 * a)
    sc["vertical"] = Scenario(
        "vertical", 11.0,
        [SimPerson("A", RED, 0, vert)],
        label=lambda t: "normal",
    )
    # lower everybody so the head sits where the virtual camera can reach it
    # (its window can only travel between y=150 and y=330 of the frame)
    for scen in sc.values():
        for p in scen.persons:
            p.pos = (lambda f: (lambda t: (f(t)[0], f(t)[1] + Y_OFFSET)))(p.pos)
    return sc


@dataclass
class SimOutcome:
    scenario: str
    log_path: str
    frames: int


def run_scenario(
    name: str,
    log_path: str,
    seed: int = 0,
    fps: float = 30.0,
    pipeline_cfg=None,
    save_video: Optional[str] = None,
    wrong_dist: float = 0.8 * BODY_W,
    body_recall: float = 0.92,
    det_noise: float = 3.0,
):
    """Run one scenario headless and write a run log. Returns the log path."""
    from .centering import CenteringController, VirtualCamera
    from .identity_state import IdState
    from .runlog import RunLogger
    from .whole_person import PipelineConfig, WholePersonPipeline

    scenario = make_scenarios()[name]
    world = World(scenario, seed=seed, body_recall=body_recall)
    pipe = WholePersonPipeline(
        TARGET,
        ScriptedFaceDetector(world),
        ScriptedIdentifier(world, TARGET),
        person_detector=ScriptedPersonDetector(world, noise=det_noise),
        cfg=pipeline_cfg or PipelineConfig(),
    )
    ctrl = CenteringController()
    cam = VirtualCamera(W, H)
    log = RunLogger(log_path, ctrl.dead_zone)
    writer = None
    if save_video:
        writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 2, H))

    target_p = next(p for p in scenario.persons if p.name == TARGET)
    dt = 1.0 / fps
    n = int(scenario.duration * fps)
    for i in range(n):
        world.t = i * dt
        frame = world.render()
        res = pipe.process(frame, world.t)

        err = cam.error_to(res.target_point) if res.target_point is not None else None
        cmd = ctrl.step(err, res.state, res.conf)
        cam.apply(cmd, dt)

        # ground truth: is the body we follow actually the target?
        wrong = 0
        if res.target_point is not None:
            gx, gy = target_p.pos(world.t)
            true_head = np.array([gx, gy + 34.0])
            wrong = int(np.linalg.norm(np.array(res.target_point) - true_head) > wrong_dist)
        log.write(res, cmd, scenario.label(world.t), wrong)

        if writer is not None:
            left = frame.copy()
            for tv in res.tracks:
                col = (0, 200, 0) if tv.is_target else (150, 150, 150)
                cv2.rectangle(left, (int(tv.box[0]), int(tv.box[1])), (int(tv.box[2]), int(tv.box[3])), col, 2)
            cv2.putText(left, f"{res.state.value} {res.conf:.2f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(np.hstack([left, cam.render(frame)]))
    log.close()
    if writer is not None:
        writer.release()
    return SimOutcome(name, log_path, n)
