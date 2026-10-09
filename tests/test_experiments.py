import importlib.util
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

APP = sys.argv[1]
spec = importlib.util.spec_from_file_location('read_tgc', Path(__file__).parents[1]/'tools/read_tgc.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def run(*args, ok=True):
    result = subprocess.run([APP, *map(str, args)], capture_output=True, text=True, timeout=60)
    assert (result.returncode == 0) == ok, (result.returncode, result.stdout, result.stderr)
    return result


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    common = ['--population', 30, '--games', 4, '--pieces', 100, '--validation-games', 4,
              '--validation-pieces', 120, '--checkpoint-pieces', 37, '--runs', 2, '--seed', 1729]
    for sampling in ('common', 'independent', 'refresh'):
        a, b, resumed = [root/f'{sampling}-{suffix}.tgc' for suffix in ('one', 'four', 'resume')]
        run('train', *common, '--sampling', sampling, '--generations', 4, '--threads', 1, '--output', a)
        run('train', *common, '--sampling', sampling, '--generations', 4, '--threads', 4, '--output', b)
        assert a.read_bytes() == b.read_bytes(), f'{sampling}: thread count changed binary output'
        run('train', *common, '--sampling', sampling, '--generations', 1, '--threads', 2, '--output', resumed)
        run('train', '--resume', resumed, '--generations', 4, '--threads', 4)
        assert a.read_bytes() == resumed.read_bytes(), f'{sampling}: resume differs from uninterrupted run'
        decoded = module.read(a)
        assert len(decoded['training']) == 2
        for training in decoded['training']:
            assert len(training['history']) == 5
            assert len(training['validation']) == 8
            assert all(game['status'] in (1, 2) for game in training['validation'])
            assert [g['seed'] for g in training['validation'][:4]] == [g['seed'] for g in training['validation'][4:]]
    trained = a
    interrupted_training = root/'interrupted-training.tgc'
    process = subprocess.Popen([APP, 'train', *map(str, common), '--generations', '4', '--threads', '1',
                                '--output', str(interrupted_training)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not interrupted_training.exists() and time.monotonic() < deadline:
            time.sleep(.001)
        assert interrupted_training.exists()
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 130, (stdout, stderr)
        run('train', '--resume', interrupted_training, '--threads', 4)
        assert interrupted_training.read_bytes() == (root/'common-one.tgc').read_bytes(), 'Interrupted generation changed optimizer RNG'
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
    full, split = root/'full.tgc', root/'split.tgc'
    args = ['--weights-file', trained, '--games', 5, '--seed', 82, '--checkpoint-pieces', 19]
    run('endurance', *args, '--pieces', 250, '--threads', 1, '--output', full)
    run('endurance', *args, '--pieces', 70, '--threads', 4, '--output', split)
    run('endurance', '--resume', split, '--pieces', 250, '--threads', 2)
    assert full.read_bytes() == split.read_bytes(), 'Endurance continuation changed game states'
    decoded = module.read(full)
    assert len(decoded['policies']) == 3 and len(decoded['games']) == 15
    assert [p['baseline'] for p in decoded['policies']] == [False, False, True]
    for p in range(3):
        assert [g['seed'] for g in decoded['games'][p*5:(p+1)*5]] == [g['seed'] for g in decoded['games'][:5]]
    stats = root/'stats.tgc'
    run('report', '--input', full, '--output', stats)
    summaries = module.read(stats)['summaries']
    assert len(summaries) == 3
    for summary in summaries:
        assert summary['games'] == summary['topouts'] + summary['censored'] + summary['active']
        assert summary['active'] == 0 and summary['p05_lines'] <= summary['median_lines'] <= summary['p95_lines']
        assert summary['ci_low'] <= summary['mean_lines'] <= summary['ci_high']
        assert summary['survival'][-1]['identifiable']
    run('report', '--input', stats, '--output', root/'roundtrip.tgc')
    assert stats.read_bytes() == (root/'roundtrip.tgc').read_bytes(), 'Statistics roundtrip changed bytes'
    run('report', '--input', trained, '--output', root/'validation-stats.tgc')
    assert len(module.read(root/'validation-stats.tgc')['summaries']) == 4
    bad = root/'bad.tgc'
    raw = bytearray(full.read_bytes()); raw[-1] ^= 1; bad.write_bytes(raw)
    assert 'checksum' in run('report', '--input', bad, '--output', stats, ok=False).stderr.lower()
    bad.write_bytes(full.read_bytes()[:-8]); run('report', '--input', bad, '--output', stats, ok=False)
    run('endurance', '--resume', split, '--pieces', 1, ok=False)
    run('train', '--resume', trained, '--seed', 2, ok=False)
    run('evaluate', '--weights-file', trained, '--output', trained, ok=False)
    huge = root/'huge-cap.tgc'
    run('endurance', '--games', 2, '--pieces', 4294967296, '--weights', '-1,0,0,0',
        '--no-baseline', '--threads', 2, '--output', huge)
    large = module.read(huge)
    assert large['config']['pieces'] == 4294967296 and len(large['games']) == 2
    assert all(g['status'] == 1 for g in large['games']) and huge.stat().st_size < 1024
    # An interrupted long game saves its live board, bag and counters, rather than a capped result.
    paused = root/'paused.tgc'
    process = subprocess.Popen([APP, 'endurance', '--games', '2', '--pieces', '1000000000',
                                '--checkpoint-pieces', '1000', '--threads', '2', '--output', str(paused)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not paused.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert paused.exists(), 'No initial endurance checkpoint'
        time.sleep(.05)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 130, (stdout, stderr)
        state = module.read(paused)
        assert any(g['status'] == 0 for g in state['games'])
        assert sum(g['pieces'] for g in state['games']) > 0
        run('report', '--input', paused, '--output', root/'paused-stats.tgc')
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
print('Binary reproducibility, training/endurance resume, paired validation, statistics and interrupt checks passed')
