"""
Evaluate one or more run logs written by track_person.py (or the simulator).

    python evaluate_run.py logs/run_20261022_101500.csv
    python evaluate_run.py runs/sim_*.csv --out reports

Produces, per log: <name>_metrics.json, <name>_report.md, <name>_timeline.png
and prints a summary table.

What is measured
  honesty      Of the frames you flagged as "tracker is on the WRONG person",
               how many did the system still call CONFIRMED? (target: 0)
  continuity   For each labelled event (head turn / occlusion / crossing):
               did identity survive (never UNKNOWN), longest TEMPORARY streak,
               and how long after the event until CONFIRMED again.
  decay        Fitted time constant of confidence while no face is verified,
               a sanity check that the decay really is exponential.
  centring     Mean |error| and % of time inside the dead zone while locked,
               plus how often the pan/tilt command changes (jitter).
"""

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# reference palette (light mode): categorical slots + chrome
C_CONF, C_TEMP, C_UNK = "#2a78d6", "#eb6834", "#898781"
C_ERRX, C_ERRY, C_WRONG = "#1baf7a", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
EVENTS = ("head_turn", "occlusion", "crossing")


# ----------------------------------------------------------------------- load
def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": float(r["t"]), "scenario": r["scenario"], "wrong": int(r["wrong"]),
                "state": r["state"], "conf": float(r["conf"]),
                "track": r["target_track"], "verified": int(r["verified"]),
                "face": int(r["face_visible"]), "amb": int(r["ambiguous"]),
                "ex": float(r["err_x"]), "ey": float(r["err_y"]),
                "hor": r["horizontal"], "ver": r["vertical"],
                "active": int(r["active"]), "dz": float(r["dead_zone"]),
            })
    return rows


def runs_of(rows, key):
    """Maximal runs of equal key(row): yields (value, i_start, i_end_inclusive)."""
    out, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or key(rows[i]) != key(rows[start]):
            out.append((key(rows[start]), start, i - 1))
            start = i
    return out


# -------------------------------------------------------------------- metrics
def honesty(rows):
    wrong = [r for r in rows if r["wrong"]]
    by = Counter(r["state"] for r in wrong)
    n = len(wrong)
    return {
        "wrong_frames": n,
        "wrong_confirmed": by.get("CONFIRMED", 0),
        "wrong_temporary": by.get("TEMPORARY", 0),
        "wrong_unknown": by.get("UNKNOWN", 0),
        "confidently_wrong_rate": (by.get("CONFIRMED", 0) / n) if n else None,
    }


def events(rows):
    out = []
    for label, i0, i1 in runs_of(rows, lambda r: r["scenario"]):
        if label not in EVENTS:
            continue
        seg = rows[i0:i1 + 1]
        t0, t1 = seg[0]["t"], seg[-1]["t"]
        kept = sum(r["state"] != "UNKNOWN" for r in seg) / len(seg)

        longest = cur = 0.0
        for k, r in enumerate(seg):
            if r["state"] == "TEMPORARY" and k > 0:
                cur += r["t"] - seg[k - 1]["t"]
                longest = max(longest, cur)
            else:
                cur = 0.0

        reconf = None
        for r in rows[i1:]:
            if r["state"] == "CONFIRMED":
                reconf = max(0.0, r["t"] - t1)
                break
        out.append({
            "event": label, "start": round(t0, 2), "end": round(t1, 2),
            "duration_s": round(t1 - t0, 2),
            "identity_kept_frac": round(kept, 3),
            "went_unknown": any(r["state"] == "UNKNOWN" for r in seg),
            "longest_temporary_s": round(longest, 2),
            "reconfirm_after_end_s": None if reconf is None else round(reconf, 2),
            "wrong_frames": sum(r["wrong"] for r in seg),
            "wrong_confirmed": sum(r["wrong"] and r["state"] == "CONFIRMED" for r in seg),
        })
    return out


def decay_fit(rows):
    """Fit ln(conf) = a - t/tau on stretches with no verification."""
    taus = []
    i = 0
    while i < len(rows):
        if rows[i]["verified"]:
            j = i + 1
            while j < len(rows) and not rows[j]["verified"] and rows[j]["state"] != "UNKNOWN" and rows[j]["conf"] > 0.05:
                j += 1
            seg = rows[i + 1:j]
            if len(seg) >= 8 and seg[-1]["t"] - seg[0]["t"] > 0.6:
                t = np.array([r["t"] for r in seg])
                y = np.log([r["conf"] for r in seg])
                slope = np.polyfit(t, y, 1)[0]
                if slope < -1e-3:
                    taus.append(-1.0 / slope)
            i = max(j, i + 1)
        else:
            i += 1
    if not taus:
        return {"n_stretches": 0, "median_tau_s": None}
    return {"n_stretches": len(taus), "median_tau_s": round(float(np.median(taus)), 2)}


def centring(rows):
    locked = [r for r in rows if r["state"] != "UNKNOWN" and r["active"]]
    if not locked:
        return {"locked_frames": 0}
    dz = locked[0]["dz"]
    ex = np.array([abs(r["ex"]) for r in locked])
    ey = np.array([abs(r["ey"]) for r in locked])
    inside = np.mean((ex <= dz) & (ey <= dz))
    flips = sum(1 for a, b in zip(rows, rows[1:]) if (a["hor"], a["ver"]) != (b["hor"], b["ver"]))
    minutes = max(1e-6, (rows[-1]["t"] - rows[0]["t"]) / 60.0)
    return {
        "locked_frames": len(locked),
        "mean_abs_err_x": round(float(ex.mean()), 3),
        "mean_abs_err_y": round(float(ey.mean()), 3),
        "pct_inside_dead_zone": round(float(inside) * 100, 1),
        "command_changes_per_min": round(flips / minutes, 1),
    }


def evaluate(rows):
    n = len(rows)
    occ = Counter(r["state"] for r in rows)
    transitions = sum(1 for a, b in zip(rows, rows[1:]) if a["state"] != b["state"])
    return {
        "frames": n,
        "duration_s": round(rows[-1]["t"] - rows[0]["t"], 2),
        "state_time_pct": {s: round(100.0 * occ.get(s, 0) / n, 1) for s in ("CONFIRMED", "TEMPORARY", "UNKNOWN")},
        "state_transitions": transitions,
        "honesty": honesty(rows),
        "events": events(rows),
        "decay": decay_fit(rows),
        "centring": centring(rows),
    }


# ----------------------------------------------------------------------- plot
def plot(rows, title, out_png):
    t = np.array([r["t"] for r in rows])
    conf = np.array([r["conf"] for r in rows])
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(11, 6.6), sharex=True, facecolor=SURF,
        gridspec_kw={"height_ratios": [0.7, 3, 2], "hspace": 0.12},
    )
    for ax in (ax0, ax1, ax2):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # scenario labels + wrong flags
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.grid(False)
    ax0.spines["left"].set_visible(False)
    for label, i0, i1 in runs_of(rows, lambda r: r["scenario"]):
        if label == "normal":
            continue
        ax0.axvspan(t[i0], t[i1], ymin=0.05, ymax=0.55, facecolor="none", edgecolor=INK2, hatch="///", linewidth=0.8)
        ax0.text((t[i0] + t[i1]) / 2, 0.8, label.replace("_", " "), ha="center", va="center", fontsize=9, color=INK)
    for _, i0, i1 in [x for x in runs_of(rows, lambda r: r["wrong"]) if x[0] == 1]:
        ax0.plot([t[i0], t[i1]], [0.12, 0.12], color=C_WRONG, linewidth=4, solid_capstyle="butt")

    # state bands + confidence
    col = {"CONFIRMED": C_CONF, "TEMPORARY": C_TEMP, "UNKNOWN": C_UNK}
    for st, i0, i1 in runs_of(rows, lambda r: r["state"]):
        x1 = t[min(i1 + 1, len(t) - 1)]
        ax1.axvspan(t[i0], x1, facecolor=col[st], alpha=0.16, linewidth=0)
    ax1.plot(t, conf, color=INK, linewidth=2)
    for v, lab in ((0.8, "confirmed ≥ 0.80"), (0.3, "unknown < 0.30")):
        ax1.axhline(v, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
        ax1.text(t[-1], v + 0.02, lab, ha="right", va="bottom", fontsize=8, color=INK2)
    ax1.set_ylim(-0.02, 1.08)
    ax1.set_ylabel("identity confidence", color=INK2, fontsize=10)

    # centring error
    ex = np.array([r["ex"] for r in rows])
    ey = np.array([r["ey"] for r in rows])
    live = np.array([r["state"] != "UNKNOWN" and r["active"] == 1 for r in rows])
    dz = rows[0]["dz"]
    ax2.axhspan(-dz, dz, color=GRID, alpha=0.9, linewidth=0)
    ax2.plot(t, np.where(live, ex, np.nan), color=C_ERRX, linewidth=2)
    ax2.plot(t, np.where(live, ey, np.nan), color=C_ERRY, linewidth=2)
    ax2.axhline(0, color=MUTED, linewidth=0.8)
    ax2.set_ylabel("centring error\n(− left/up, + right/down)", color=INK2, fontsize=10)
    ax2.set_xlabel("time (s)", color=INK2, fontsize=10)
    lim = max(0.3, float(np.nanmax(np.abs(np.where(live, np.stack([ex, ey]), 0)))) * 1.1)
    ax2.set_ylim(-lim, lim)

    handles = [
        Patch(facecolor=C_CONF, alpha=0.3, label="CONFIRMED"),
        Patch(facecolor=C_TEMP, alpha=0.3, label="TEMPORARY"),
        Patch(facecolor=C_UNK, alpha=0.3, label="UNKNOWN"),
        Line2D([0], [0], color=INK, linewidth=2, label="confidence"),
        Line2D([0], [0], color=C_ERRX, linewidth=2, label="horizontal error"),
        Line2D([0], [0], color=C_ERRY, linewidth=2, label="vertical error"),
        Line2D([0], [0], color=C_WRONG, linewidth=4, label="flagged wrong person"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(title, x=0.07, ha="left", fontsize=13, color=INK, y=0.985)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.08)
    fig.savefig(out_png, dpi=140, facecolor=SURF)
    plt.close(fig)


# --------------------------------------------------------------------- report
def fmt(v, none="n/a"):
    return none if v is None else str(v)


def report_md(name, m):
    h, c, d = m["honesty"], m["centring"], m["decay"]
    L = [f"# Evaluation: {name}", "",
         f"{m['frames']} frames, {m['duration_s']} s, {m['state_transitions']} state transitions.", "",
         "## Time in each state", "",
         "| CONFIRMED | TEMPORARY | UNKNOWN |", "|---|---|---|",
         f"| {m['state_time_pct']['CONFIRMED']}% | {m['state_time_pct']['TEMPORARY']}% | {m['state_time_pct']['UNKNOWN']}% |", "",
         "## Honesty (needs the W flag, or the simulator's ground truth)", ""]
    if h["wrong_frames"] == 0:
        L.append("No frames were flagged as 'wrong person', so honesty is untested in this run.")
    else:
        L += [f"- Frames on the wrong person: **{h['wrong_frames']}**",
              f"- …labelled CONFIRMED (confidently wrong): **{h['wrong_confirmed']}** "
              f"({h['confidently_wrong_rate'] * 100:.1f}%)",
              f"- …labelled TEMPORARY: {h['wrong_temporary']}, UNKNOWN: {h['wrong_unknown']}"]
    L += ["", "## Events", ""]
    if not m["events"]:
        L.append("No labelled events (use keys 2/3/4 while they happen).")
    else:
        L += ["| event | start–end (s) | identity kept | went UNKNOWN | longest TEMPORARY (s) | re-confirm after end (s) | wrong frames (CONFIRMED) |",
              "|---|---|---|---|---|---|---|"]
        for e in m["events"]:
            L.append(f"| {e['event']} | {e['start']}–{e['end']} | {e['identity_kept_frac'] * 100:.0f}% | "
                     f"{'yes' if e['went_unknown'] else 'no'} | {e['longest_temporary_s']} | "
                     f"{fmt(e['reconfirm_after_end_s'])} | {e['wrong_frames']} ({e['wrong_confirmed']}) |")
    L += ["", "## Confidence decay", "",
          f"Fitted time constant while no face was verified: **{fmt(d['median_tau_s'])} s** "
          f"over {d['n_stretches']} stretch(es). It differs from the configured tau when appearance, "
          "coasting or overlap speed up or slow down the decay.", "",
          "## Centring", ""]
    if c.get("locked_frames", 0) == 0:
        L.append("No locked frames.")
    else:
        L += [f"- Mean |error|: x = {c['mean_abs_err_x']}, y = {c['mean_abs_err_y']} (frame half-size = 1.0)",
              f"- Inside dead zone: {c['pct_inside_dead_zone']}% of locked frames",
              f"- Pan/tilt command changes: {c['command_changes_per_min']} per minute"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None, help="output folder (default: next to each log)")
    args = ap.parse_args()

    summary = []
    for path in args.logs:
        rows = load(path)
        if len(rows) < 2:
            print(f"skip {path}: too few rows")
            continue
        name = Path(path).stem
        out_dir = Path(args.out) if args.out else Path(path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        m = evaluate(rows)
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(m, indent=2))
        (out_dir / f"{name}_report.md").write_text(report_md(name, m))
        plot(rows, name, str(out_dir / f"{name}_timeline.png"))
        h = m["honesty"]
        summary.append((name, m, h))

    print(f"\n{'run':28s} {'CONF%':>6s} {'TEMP%':>6s} {'UNK%':>6s} {'wrong':>6s} {'wrong&CONF':>10s} {'in-dz%':>7s}")
    for name, m, h in summary:
        s, c = m["state_time_pct"], m["centring"]
        print(f"{name:28s} {s['CONFIRMED']:6.1f} {s['TEMPORARY']:6.1f} {s['UNKNOWN']:6.1f} "
              f"{h['wrong_frames']:6d} {h['wrong_confirmed']:10d} {c.get('pct_inside_dead_zone', float('nan')):7.1f}")


if __name__ == "__main__":
    main()
