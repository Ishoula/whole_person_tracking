"""
Logic tests (no webcam, no models). Run:  python -m unittest test_whole_person -v
"""

import csv
import math
import os
import tempfile
import unittest

import numpy as np

from src.centering import CenteringController, VirtualCamera
from src.face_body_map import face_body_score, map_faces_to_bodies
from src.identity_state import Evidence, IdentityConfig, IdentityManager, IdState
from src.person_tracker import Detection, PersonTracker, decode_yolov8, decode_yolox, face_to_body_box, iou
from src.sim import run_scenario


def ev(**kw):
    base = dict(live_tracks={1}, anchor_visible=True)
    base.update(kw)
    return Evidence(**base)


def states_over_time(rows):
    seq = []
    for r in rows:
        if not seq or seq[-1] != r["state"]:
            seq.append(r["state"])
    return seq


class IdentityTests(unittest.TestCase):
    def verified(self, m, t=0.0):
        return m.update(t, ev(verified_track=1))

    def test_starts_unknown_and_face_confirms(self):
        m = IdentityManager()
        self.assertEqual(m.update(0.0, ev())[0], IdState.UNKNOWN)
        state, conf = self.verified(m, 0.1)
        self.assertEqual((state, conf), (IdState.CONFIRMED, 1.0))

    def test_decay_is_exponential(self):
        m = IdentityManager(IdentityConfig(tau=4.0))
        self.verified(m, 0.0)
        for k in range(1, 21):                       # 2 s in 0.1 s steps
            _, conf = m.update(k * 0.1, ev())
        self.assertAlmostEqual(conf, math.exp(-2.0 / 4.0), places=6)

    def test_confirmed_to_temporary_to_unknown(self):
        m = IdentityManager(IdentityConfig(tau=4.0))
        self.verified(m, 0.0)
        seen, t = [], 0.0
        while t < 8.0:
            t += 0.05
            seen.append(m.update(t, ev())[0])
        order = [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]]
        self.assertEqual(order, [IdState.CONFIRMED, IdState.TEMPORARY, IdState.UNKNOWN])

    def test_appearance_never_confirms(self):
        m = IdentityManager()
        self.verified(m, 0.0)
        t = 0.0
        for _ in range(400):                          # 20 s of perfect appearance match
            t += 0.05
            m.update(t, ev(appearance_sim=1.0))
        self.assertNotEqual(m.state, IdState.CONFIRMED)

    def test_appearance_modulates_decay_speed(self):
        def conf_after(sim):
            m = IdentityManager()
            self.verified(m, 0.0)
            c = 1.0
            for k in range(1, 31):
                _, c = m.update(k * 0.1, ev(appearance_sim=sim))
            return c
        self.assertGreater(conf_after(0.95), conf_after(None))
        self.assertLess(conf_after(0.30), conf_after(None))

    def test_stale_belief_is_capped_while_entangled(self):
        m = IdentityManager()
        self.verified(m, 0.0)
        for k in range(1, 4):                         # 0.3 s: still fresh, not capped
            _, c = m.update(k * 0.1, ev(ambiguous=True))
        self.assertGreater(c, 0.7)                    # decaying, but not slammed down to the 0.5 cap
        for k in range(4, 30):                        # stale now
            _, c = m.update(k * 0.1, ev(ambiguous=True))
        self.assertLessEqual(c, 0.5)
        self.assertNotEqual(m.state, IdState.CONFIRMED)

    def test_visible_non_matching_face_cuts_confidence(self):
        m = IdentityManager()
        self.verified(m, 0.0)
        _, c = m.update(0.1, ev(mismatch_tracks={1}))
        self.assertLess(c, 0.6)

    def test_lost_body_means_unknown(self):
        m = IdentityManager()
        self.verified(m, 0.0)
        state, conf = m.update(0.1, ev(live_tracks=set()))
        self.assertEqual((state, conf, m.track_id), (IdState.UNKNOWN, 0.0, None))

    def test_face_verified_on_other_body_rebinds(self):
        m = IdentityManager()
        self.verified(m, 0.0)
        m.update(1.0, Evidence(verified_track=2, live_tracks={1, 2}))
        self.assertEqual((m.track_id, m.state), (2, IdState.CONFIRMED))


class MappingTests(unittest.TestCase):
    BODY = (100, 50, 230, 430)

    def test_head_zone_face_maps(self):
        self.assertGreater(face_body_score((140, 56, 192, 112), self.BODY), 0.7)

    def test_face_in_lower_body_rejected(self):
        self.assertEqual(face_body_score((140, 300, 192, 356), self.BODY), 0.0)

    def test_face_outside_rejected(self):
        self.assertEqual(face_body_score((400, 56, 452, 112), self.BODY), 0.0)

    def test_one_to_one_assignment(self):
        bodies = {1: (100, 50, 230, 430), 2: (300, 50, 430, 430)}
        faces = [(330, 56, 382, 112), (140, 56, 192, 112)]
        links, unmatched = map_faces_to_bodies(faces, bodies)
        self.assertEqual({(l.face_index, l.track_id) for l in links}, {(0, 2), (1, 1)})
        self.assertEqual(unmatched, [])

    def test_extra_face_is_unmatched(self):
        links, unmatched = map_faces_to_bodies([(140, 56, 192, 112), (150, 60, 200, 114)], {1: self.BODY})
        self.assertEqual(len(links), 1)
        self.assertEqual(len(unmatched), 1)

    def test_face_estimated_body_contains_face(self):
        face = (300, 100, 360, 170)
        body = face_to_body_box(face, (480, 640, 3))
        self.assertGreater(face_body_score(face, body), 0.7)


class TrackerTests(unittest.TestCase):
    def frame(self):
        return np.full((480, 640, 3), 70, np.uint8)

    def test_two_people_keep_their_ids(self):
        tr, f = PersonTracker(), self.frame()
        ids = None
        for k in range(30):
            t = k / 30.0
            a = Detection((100 + 3 * k, 50, 230 + 3 * k, 430))
            b = Detection((400 - 3 * k, 50, 530 - 3 * k, 430))
            tr.predict(t)
            tr.update(f, [a, b], t)
            cur = sorted((tid, round(float(t_.center[0]) / 100)) for tid, t_ in tr.tracks.items())
            ids = ids or cur
        self.assertEqual(len(tr.tracks), 2)
        left = min(tr.tracks.values(), key=lambda x: x.center[0])
        self.assertEqual(left.id, 1)

    def test_coasting_then_expiry(self):
        tr, f = PersonTracker(max_coast_s=2.0), self.frame()
        tr.predict(0.0)
        tr.update(f, [Detection((100, 50, 230, 430))], 0.0)
        tr.predict(1.0); tr.expire(1.0)
        self.assertEqual(len(tr.tracks), 1)
        tr.predict(2.5); tr.expire(2.5)
        self.assertEqual(len(tr.tracks), 0)

    def test_entanglement_flag(self):
        tr, f = PersonTracker(), self.frame()
        tr.predict(0.0)
        tr.update(f, [Detection((100, 50, 230, 430)), Detection((400, 50, 530, 430))], 0.0)
        self.assertFalse(any(tr.is_entangled(t) for t in tr.tracks))
        for k in range(1, 16):                        # B walks into A over half a second
            t = k / 30.0
            tr.predict(t)
            tr.update(f, [Detection((100, 50, 230, 430)), Detection((400 - 14 * k, 50, 530 - 14 * k, 430))], t)
        self.assertEqual(len(tr.tracks), 2)
        self.assertTrue(any(tr.is_entangled(t) for t in tr.tracks))

    def test_yolo_decode(self):
        out = np.zeros((1, 84, 3), np.float32)
        out[0, :4, 0] = [320, 320, 100, 200]; out[0, 4, 0] = 0.9     # strong person
        out[0, :4, 1] = [322, 322, 100, 200]; out[0, 4, 1] = 0.6     # duplicate -> NMS
        out[0, :4, 2] = [50, 50, 20, 20];    out[0, 4, 2] = 0.1      # below threshold
        dets = decode_yolov8(out, 640, 640, 640)
        self.assertEqual(len(dets), 1)
        self.assertAlmostEqual(dets[0].box[0], 270.0, places=3)

    def test_yolox_decode(self):
        out = np.zeros((1, 3, 85), np.float32)
        out[0, 0, :4] = [208, 208, 100, 200]; out[0, 0, 4] = 0.9; out[0, 0, 5] = 0.9  # strong person
        out[0, 1, :4] = [210, 210, 100, 200]; out[0, 1, 4] = 0.8; out[0, 1, 5] = 0.8  # duplicate -> NMS
        out[0, 2, :4] = [50, 50, 20, 20];    out[0, 2, 4] = 0.2; out[0, 2, 5] = 0.2  # below threshold
        dets = decode_yolox(out, 640, 480, input_size=(416, 416))
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].source, "yolox")
        self.assertAlmostEqual(dets[0].box[0], (208 - 50) * (640 / 416.0), places=3)

    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)


class CenteringTests(unittest.TestCase):
    def test_directions_match_mqtt_words(self):
        c = CenteringController()
        cmd = c.step((-0.5, -0.5), IdState.CONFIRMED, 1.0)
        self.assertEqual((cmd.horizontal, cmd.vertical), ("LEFT", "UP"))
        c = CenteringController()
        cmd = c.step((0.5, 0.5), IdState.CONFIRMED, 1.0)
        self.assertEqual((cmd.horizontal, cmd.vertical), ("RIGHT", "DOWN"))

    def test_dead_zone_and_hysteresis(self):
        c = CenteringController(dead_zone=0.10, ema_alpha=1.0)
        self.assertEqual(c.step((0.05, 0.0), IdState.CONFIRMED, 1.0).horizontal, "CENTER")
        self.assertEqual(c.step((0.15, 0.0), IdState.CONFIRMED, 1.0).horizontal, "RIGHT")
        self.assertEqual(c.step((0.08, 0.0), IdState.CONFIRMED, 1.0).horizontal, "RIGHT")   # still moving
        self.assertEqual(c.step((0.05, 0.0), IdState.CONFIRMED, 1.0).horizontal, "CENTER")

    def test_unknown_never_moves(self):
        cmd = CenteringController().step((0.9, 0.9), IdState.UNKNOWN, 0.0)
        self.assertEqual((cmd.horizontal, cmd.vertical, cmd.active), ("CENTER", "CENTER", False))

    def test_temporary_moves_more_gently(self):
        full = CenteringController(ema_alpha=1.0).step((0.4, 0.0), IdState.CONFIRMED, 1.0)
        soft = CenteringController(ema_alpha=1.0).step((0.4, 0.0), IdState.TEMPORARY, 0.45)
        self.assertLess(abs(soft.pan_speed), abs(full.pan_speed))
        self.assertGreater(abs(soft.pan_speed), 0.0)

    def test_virtual_camera_converges_and_saturates(self):
        cam, ctrl = VirtualCamera(640, 480), CenteringController()
        target = (450.0, 300.0)
        for _ in range(300):
            cmd = ctrl.step(cam.error_to(target), IdState.CONFIRMED, 1.0)
            cam.apply(cmd, 1 / 30)
        ex, ey = cam.error_to(target)
        self.assertLess(abs(ex), 0.07); self.assertLess(abs(ey), 0.07)
        # unreachable target: window stops at the frame edge (like a soft limit)
        for _ in range(300):
            cam.apply(ctrl.step(cam.error_to((10.0, 10.0)), IdState.CONFIRMED, 1.0), 1 / 30)
        self.assertEqual(cam.at_limit(), (True, True))


class SimulationTests(unittest.TestCase):
    def run_sim(self, name, **kw):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.csv")
            run_scenario(name, path, seed=3, **kw)
            with open(path) as f:
                return list(csv.DictReader(f))

    def test_head_turn_decays_then_reconfirms(self):
        rows = self.run_sim("head_turn_short")
        self.assertEqual(states_over_time(rows), ["CONFIRMED", "TEMPORARY", "CONFIRMED"])

    def test_long_head_turn_goes_unknown_then_reacquires(self):
        rows = self.run_sim("head_turn_long")
        self.assertEqual(states_over_time(rows), ["CONFIRMED", "TEMPORARY", "UNKNOWN", "CONFIRMED"])

    def test_occlusion_never_loses_identity(self):
        rows = self.run_sim("occlusion")
        self.assertNotIn("UNKNOWN", {r["state"] for r in rows})

    def test_vertical_tracking_reaches_dead_zone(self):
        rows = self.run_sim("vertical")
        late = [abs(float(r["err_y"])) for r in rows[60:]]
        self.assertLess(float(np.mean(late)), 0.12)
        self.assertIn("UP", {r["vertical"] for r in rows})
        self.assertIn("DOWN", {r["vertical"] for r in rows})

    def test_crossing_never_confidently_wrong(self):
        for name in ("crossing", "crossing_similar", "crossing_swap"):
            for r in self.run_sim(name, body_recall=0.7, det_noise=12.0):
                self.assertFalse(r["wrong"] == "1" and r["state"] == "CONFIRMED", name)


if __name__ == "__main__":
    unittest.main()
