#!/usr/bin/env python3
"""Read tetris-gc v2 portable binary snapshots. No third-party packages required."""
import argparse
import json
import struct
from pathlib import Path


class Reader:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def unpack(self, fmt):
        size = struct.calcsize('<' + fmt)
        if self.offset + size > len(self.data):
            raise ValueError('Truncated binary payload')
        result = struct.unpack_from('<' + fmt, self.data, self.offset)
        self.offset += size
        return result

    def u32(self): return self.unpack('I')[0]
    def u64(self): return self.unpack('Q')[0]
    def f64(self): return self.unpack('d')[0]
    def weights(self): return list(self.unpack('4d'))

    def game(self):
        seed, lines, pieces, random_state = self.unpack('4Q')
        status, current, following, index = self.unpack('4I')
        bag, rows = list(self.unpack('7I')), list(self.unpack('22I'))
        return dict(seed=seed, lines=lines, pieces=pieces, random_state=random_state,
                    status=status, current=current, next=following, bag_index=index, bag=bag, rows=rows)

    def games(self): return [self.game() for _ in range(self.u64())]


def read(path):
    data = Path(path).read_bytes()
    if len(data) < 32 or data[:8] != b'TGC2BIN\0':
        raise ValueError('Not a tetris-gc binary snapshot')
    version, kind, size, expected = struct.unpack_from('<IIQQ', data, 8)
    if version != 2 or kind not in (1, 2, 3) or size != len(data) - 32:
        raise ValueError('Unsupported version/kind or invalid size')
    h = 14695981039346656037
    for byte in data[32:]:
        h = ((h ^ byte) * 1099511628211) & ((1 << 64) - 1)
    if h != expected:
        raise ValueError('Checksum mismatch')
    r = Reader(data[32:])
    config = dict(rules=r.u32(), seed=r.u64(), pieces=r.u64(), validation_pieces=r.u64(), checkpoint_pieces=r.u64(),
                  population=r.u32(), games=r.u32(), generations=r.u32(), runs=r.u32(), validation_games=r.u32(),
                  mutation=r.f64(), sampling=r.u32())
    runs = []
    for _ in range(r.u64()):
        run = dict(seed=r.u64(), random_state=r.u64(), next_id=r.u64(), generation=r.unpack('i')[0])
        population = []
        for _ in range(r.u64()):
            candidate = dict(id=r.u64(), weights=r.weights())
            candidate.update(zip(('lines', 'pieces', 'topouts', 'games'), r.unpack('4Q')))
            population.append(candidate)
        run['population'] = population
        history = []
        for _ in range(r.u64()):
            history.append(dict(generation=r.u32(), best=r.u64(), evaluated_pieces=r.u64(), mean=r.f64(), weights=r.weights()))
        run.update(history=history, validation=r.games())
        runs.append(run)
    policies = [dict(training_seed=r.u64(), weights=r.weights(), baseline=bool(r.u32())) for _ in range(r.u64())]
    games = r.games()
    summaries = []
    if kind == 3:
        for _ in range(r.u64()):
            summary = dict(zip(('games', 'topouts', 'censored', 'active'), r.unpack('4Q')))
            summary['mean_ci_available'] = bool(r.u32())
            summary.update(zip(('mean_lines', 'median_lines', 'p05_lines', 'p95_lines', 'sd_lines', 'ci_low', 'ci_high',
                                'mean_pieces', 'median_pieces', 'p05_pieces', 'p95_pieces'), r.unpack('11d')))
            summary['survival'] = [dict(milestone=r.u64(), identifiable=bool(r.u32()), probability=r.f64(),
                                        ci_low=r.f64(), ci_high=r.f64()) for _ in range(5)]
            summaries.append(summary)
    if r.offset != len(r.data):
        raise ValueError('Unexpected trailing payload')
    return dict(version=version, kind=kind, config=config, training=runs, policies=policies, games=games, summaries=summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--json', action='store_true', help='Explicitly decode all records to stdout')
    args = parser.parse_args()
    result = read(args.path)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"kind={result['kind']} rules={result['config']['rules']} runs={len(result['training'])} game-records={len(result['games'])}")
        for run in result['training']:
            best = run['population'][0]
            print(f"seed={run['seed']} generation={run['generation']} best-lines={best['lines']} weights={best['weights']}")
        for i, summary in enumerate(result['summaries']):
            print(f"policy={i} mean-lines={summary['mean_lines']:.3f} median={summary['median_lines']:.3f} "
                  f"topouts={summary['topouts']} censored={summary['censored']} active={summary['active']}")
