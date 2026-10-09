# tetris-gc binary format, version 2

All integers and IEEE-754 binary64 doubles are little-endian, serialized field by
field without C++ padding. Files are bounded to 1 GiB of payload. The included
`tools/read_tgc.py` is a runnable reference decoder. Game records cost 164 bytes;
there are no per-piece traces.

## Header (32 bytes)

| Field | Type |
| --- | --- |
| Magic `TGC2BIN\0` | 8 bytes |
| Version, currently 2 | u32 |
| Kind: training=1, endurance/evaluation=2, statistics=3 | u32 |
| Payload byte length | u64 |
| FNV-1a 64-bit checksum of payload | u64 |

Checksum offset basis is 14695981039346656037; multiplier is 1099511628211.
Arithmetic wraps modulo 2^64. This detects corruption, not malicious modification.

## Payload

The following structures occur in order:

1. Configuration (68 bytes).
2. `u64 training_run_count`, then that many training runs.
3. `u64 policy_count`, then that many policies.
4. `u64 game_count`, then that many game states.
5. For statistics files only: `u64 summary_count`, then that many summaries.

Configuration fields, in order:

```text
u32 rules                   # article=0, battleblocks=1, article-drop=2
u64 seed, pieces, validation_pieces, checkpoint_pieces
u32 population, games, generations, runs, validation_games
f64 mutation
u32 sampling                # common=0, independent=1, refresh=2
```

Training run:

```text
u64 seed, optimizer_random_state, next_candidate_id
i32 generation              # -1 = initial population not yet evaluated
u64 population_count
  candidate[population_count]
u64 history_count
  history[history_count]
u64 validation_game_count
  game[validation_game_count]
```

Candidate (72 bytes): `u64 id`, `f64 weights[4]`, then `u64 lines, pieces,
topouts, games`. Populations are ranked by descending fitness, with candidate IDs
breaking ties. Weight order is height, holes, bumpiness, lines; costs are minimized.

History entry (60 bytes): `u32 generation`, `u64 best_lines, evaluated_pieces`,
`f64 mean_fitness`, `f64 best_weights[4]`. No timestamps are stored, making
identical experiments produce identical bytes across thread counts and resume.

Policy (44 bytes): `u64 training_seed`, `f64 weights[4]`, `u32 baseline` (0 or 1). Policies occur in
training-run order, with a final published baseline unless it was explicitly omitted.
For validation reports, each run contributes a winner followed by its paired baseline.

Game state (164 bytes):

```text
u64 seed, lines, pieces, bag_random_state
u32 status                  # 0 active, 1 topout, 2 administrative cap
u32 current_piece, next_piece, bag_index
u32 bag[7]
u32 board_rows[22]          # 10-bit row masks, bottom row first
```

Piece IDs: I=0, O=1, T=2, S=3, Z=4, J=5, L=6. Bag index is the index of the next
unconsumed bag entry; 7 triggers reshuffling. Cached board heights and occupancy
are rebuilt on loading. Game records are policy-major: `games` records per policy.
Validation records in a training run are winner-major, then baseline-major.
`status=2` means alive at the configured cap; extending the cap reopens that state.

Summary (304 bytes):

```text
u64 games, topouts, censored, active
u32 mean_ci_available
f64 mean_lines, median_lines, p05_lines, p95_lines, sd_lines, ci_low, ci_high
f64 mean_pieces, median_pieces, p05_pieces, p95_pieces
survival[5]
```

Each survival entry (36 bytes): `u64 milestone`, `u32 identifiable`, `f64 probability,
ci_low, ci_high`. Horizons are 1,000, 10,000, 100,000, 1,000,000 and the configured
cap, in that order (duplicates are permitted). When `identifiable=0`, the horizon
lies beyond the data's support; the stored Kaplan-Meier tail value must not be
interpreted as an established survival probability there.

Random state belongs to SplitMix64; bag shuffling uses unbiased bounded sampling
and descending Fisher-Yates. Exact resumption requires a compatible simulator
version, rules and arithmetic. The snapshot contains process state, not compiler
or operating-system provenance.
