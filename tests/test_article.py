import random
import subprocess
import sys
from pathlib import Path

probe, node = sys.argv[1:3]
rng = random.Random(8229)
cases = []
for trial in range(50):
    height = trial % 22
    rows = [rng.randrange(1024) if y < height else 0 for y in range(22)]
    weights = (0.510066, 0.35663, 0.184483, -0.760666) if trial % 2 else (1, 2, 3, -4)
    for piece in range(7):
        for next_piece in (-1, (piece + 3) % 7):
            cases.append(' '.join(map(str, (piece, next_piece, *weights, *rows))))
text = '\n'.join(cases) + '\n'
actual = subprocess.run([probe], input=text, text=True, capture_output=True, check=True).stdout.splitlines()
expected = subprocess.run([node, str(Path(__file__).parent/'reference/verify.js')], input=text, text=True, capture_output=True, check=True).stdout.splitlines()
assert len(actual) == len(expected) == len(cases)
for i, (a, b) in enumerate(zip(actual, expected)):
    assert a == b, f'Case {i} differs:\n{cases[i]}\nC++: {a}\nJS:  {b}'
print(f'{len(cases)} placements match the original JavaScript')
