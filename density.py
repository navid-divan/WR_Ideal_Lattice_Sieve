import csv
import math
import os
from math import isqrt

import numpy as np

from wrlattices import nearsquare_sieve, squarefree_sieve

DELTA = 1.0 - (1.0 + math.log(math.log(2.0))) / math.log(2.0)
NMAX = 10 ** 8
CHECKPOINTS = [10 ** 3, 3 * 10 ** 3, 10 ** 4, 3 * 10 ** 4, 10 ** 5, 3 * 10 ** 5,
               10 ** 6, 3 * 10 ** 6, 10 ** 7, 3 * 10 ** 7, 10 ** 8]


def pair_total(N):
    total = 0
    d = 1
    while d * d <= N:
        top = min(3 * d, N // d)
        if top > d:
            total += top - d
        d += 1
    return total


def prime_sum_table():
    rows = []
    for x in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7):
        sieve = np.ones(x + 1, dtype=bool)
        sieve[:2] = False
        for p in range(2, isqrt(x) + 1):
            if sieve[p]:
                sieve[p * p::p] = False
        ps = np.nonzero(sieve)[0]
        s = int(ps.astype(np.int64).sum())
        M = len(ps)
        pnt = x * x / (2 * math.log(x))
        wrong = (M * M / 2) * math.log(M) ** 2
        rows.append((x, M, s, pnt, wrong))
    return rows


def main():
    os.makedirs("results", exist_ok=True)
    near = nearsquare_sieve(NMAX)
    sf = squarefree_sieve(NMAX)
    ind = near & sf
    ind[0::2] = False
    del near
    sfodd = sf.copy()
    sfodd[0::2] = False
    del sf
    with open("results/density.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "wr_fields", "sf_odd", "sf_ratio", "pair_total", "pair_ratio", "ford_shape"])
        for N in CHECKPOINTS:
            cnt = int(ind[:N + 1].sum())
            base = int(sfodd[:N + 1].sum())
            pt = pair_total(N)
            shape = (math.log(N)) ** (-DELTA) * (math.log(math.log(N))) ** (-1.5)
            w.writerow([N, cnt, base, cnt / base, pt, pt / N, shape])
            print(N, cnt, base, round(cnt / base, 5), pt, round(pt / N, 5), flush=True)
    with open("results/primesums.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "M", "sum_p", "pnt_approx", "claimed_bound"])
        for row in prime_sum_table():
            w.writerow(row)
            print(row, flush=True)


if __name__ == "__main__":
    main()
