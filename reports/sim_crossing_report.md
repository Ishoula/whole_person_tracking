# Evaluation: sim_crossing

360 frames, 11.97 s, 3 state transitions.

## Time in each state

| CONFIRMED | TEMPORARY | UNKNOWN |
|---|---|---|
| 80.6% | 8.1% | 11.4% |

## Honesty (needs the W flag, or the simulator's ground truth)

No frames were flagged as 'wrong person', so honesty is untested in this run.

## Events

| event | start–end (s) | identity kept | went UNKNOWN | longest TEMPORARY (s) | re-confirm after end (s) | wrong frames (CONFIRMED) |
|---|---|---|---|---|---|---|
| crossing | 4.5–7.47 | 54% | yes | 0.97 | 0.0 | 0 (0) |

## Confidence decay

Fitted time constant while no face was verified: **0.98 s** over 1 stretch(es). It differs from the configured tau when appearance, coasting or overlap speed up or slow down the decay.

## Centring

- Mean |error|: x = 0.144, y = 0.033 (frame half-size = 1.0)
- Inside dead zone: 34.2% of locked frames
- Pan/tilt command changes: 35.1 per minute
