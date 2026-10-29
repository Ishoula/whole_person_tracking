# Evaluation: sim_occlusion

360 frames, 11.97 s, 2 state transitions.

## Time in each state

| CONFIRMED | TEMPORARY | UNKNOWN |
|---|---|---|
| 86.7% | 13.3% | 0.0% |

## Honesty (needs the W flag, or the simulator's ground truth)

No frames were flagged as 'wrong person', so honesty is untested in this run.

## Events

| event | start–end (s) | identity kept | went UNKNOWN | longest TEMPORARY (s) | re-confirm after end (s) | wrong frames (CONFIRMED) |
|---|---|---|---|---|---|---|
| occlusion | 4.2–6.37 | 100% | no | 1.6 | 0.03 | 0 (0) |

## Confidence decay

Fitted time constant while no face was verified: **3.4 s** over 1 stretch(es). It differs from the configured tau when appearance, coasting or overlap speed up or slow down the decay.

## Centring

- Mean |error|: x = 0.184, y = 0.033 (frame half-size = 1.0)
- Inside dead zone: 55.6% of locked frames
- Pan/tilt command changes: 25.1 per minute
