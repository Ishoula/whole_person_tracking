# Evaluation: sim_head_turn_long

420 frames, 13.97 s, 3 state transitions.

## Time in each state

| CONFIRMED | TEMPORARY | UNKNOWN |
|---|---|---|
| 49.3% | 28.8% | 21.9% |

## Honesty (needs the W flag, or the simulator's ground truth)

No frames were flagged as 'wrong person', so honesty is untested in this run.

## Events

| event | start–end (s) | identity kept | went UNKNOWN | longest TEMPORARY (s) | re-confirm after end (s) | wrong frames (CONFIRMED) |
|---|---|---|---|---|---|---|
| head_turn | 3.0–10.97 | 62% | yes | 4.03 | 0.03 | 0 (0) |

## Confidence decay

Fitted time constant while no face was verified: **4.13 s** over 1 stretch(es). It differs from the configured tau when appearance, coasting or overlap speed up or slow down the decay.

## Centring

- Mean |error|: x = 0.028, y = 0.03 (frame half-size = 1.0)
- Inside dead zone: 95.4% of locked frames
- Pan/tilt command changes: 8.6 per minute
