import csv
import gc
import os
import time
import tracemalloc
from math import gcd, isqrt

import numpy as np

from wrlattices import (wr_field_indicator, is_squarefree, nearsquare_divscan,
                        nearsquare_factor, classify_field, divisor_pairs3,
                        wr_ideals_bruteforce, is_wr_form, is_wr_form_enum,
                        ideal_gram, reduce_form)


def range_sieve(N):
    ind = wr_field_indicator(N)
    return int(ind.sum())


def range_divscan(N):
    cnt = 0
    for D in range(3, N + 1, 2):
        if is_squarefree(D) and nearsquare_divscan(D):
            cnt += 1
    return cnt


def range_spf(N):
    spf = np.zeros(N + 1, dtype=np.int32)
    for p in range(2, isqrt(N) + 1):
        mask = spf[p::p] == 0
        spf[p::p][mask] = p
    cnt = 0
    for D in range(3, N + 1, 2):
        n = D
        fac = []
        sq = False
        while n > 1:
            p = spf[n] if spf[n] else n
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e > 1:
                sq = True
                break
            fac.append(p)
        if sq:
            continue
        divs = [1]
        for p in fac:
            divs += [d * p for d in divs]
        ok = False
        for d in divs:
            if d < D // d <= 3 * d:
                ok = True
                break
        if ok:
            cnt += 1
    return cnt


def field_classify(D):
    return classify_field(D, 1)


def field_brute(D):
    Amax = 3 * isqrt(3 * D) + 6
    return wr_ideals_bruteforce(D, 1, Amax)


def tester_reduce(forms):
    out = 0
    for A, B, C in forms:
        if is_wr_form(A, B, C):
            out += 1
    return out


def tester_enum(forms):
    out = 0
    for A, B, C in forms:
        if is_wr_form_enum(A, B, C):
            out += 1
    return out


def make_test_forms(D, eps, count):
    forms = []
    a = 1
    while len(forms) < count:
        a += 1
        for b in range(a):
            G = ideal_gram(D, eps, a, b)
            if G is not None:
                A, B, C = G
                g = gcd(gcd(A, B), C)
                forms.append((A // g, B // g, C // g))
                if len(forms) >= count:
                    break
    return forms


RANGE_GRIDS = {
    "sieve": [10 ** 5, 10 ** 6, 10 ** 7, 3 * 10 ** 7, 10 ** 8],
    "spf": [10 ** 5, 3 * 10 ** 5, 10 ** 6, 3 * 10 ** 6],
    "divscan": [10 ** 3, 3 * 10 ** 3, 10 ** 4, 3 * 10 ** 4, 10 ** 5],
}

FIELD_GRID = [10 ** 4 + 5, 10 ** 5 + 3, 10 ** 6 + 5, 10 ** 7 + 5, 10 ** 8 + 7, 10 ** 9 + 5]


def nearest_wr_D(x):
    D = x if x % 2 == 1 else x + 1
    while not (is_squarefree(D) and nearsquare_factor(D)):
        D += 2
    return D


def run():
    os.makedirs("results", exist_ok=True)
    rows = []
    fns = {"sieve": range_sieve, "spf": range_spf, "divscan": range_divscan}
    for method, grid in RANGE_GRIDS.items():
        for N in grid:
            gc.collect()
            t0 = time.perf_counter()
            val = fns[method](N)
            t1 = time.perf_counter()
            rows.append({"task": "range", "method": method, "size": N,
                         "seconds": t1 - t0, "value": val})
            print("range", method, N, round(t1 - t0, 3), val, flush=True)
    for x in FIELD_GRID:
        D = nearest_wr_D(x)
        gc.collect()
        t0 = time.perf_counter()
        prs, ideals = field_classify(D)
        t1 = time.perf_counter()
        rows.append({"task": "field", "method": "classify", "size": D,
                     "seconds": t1 - t0, "value": len(ideals)})
        print("field classify", D, round(t1 - t0, 5), len(ideals), flush=True)
        if D <= 10 ** 6 + 10:
            t0 = time.perf_counter()
            brute = field_brute(D)
            t1 = time.perf_counter()
            rows.append({"task": "field", "method": "brute", "size": D,
                         "seconds": t1 - t0, "value": len(brute)})
            print("field brute", D, round(t1 - t0, 3), len(brute), flush=True)
            assert sorted(brute) == sorted(ideals), D
    for k in (4, 6, 8, 10):
        D = nearest_wr_D(10 ** k + 1)
        forms = make_test_forms(D, 1, 60)
        for method, fn in (("reduce", tester_reduce), ("enum", tester_enum)):
            gc.collect()
            t0 = time.perf_counter()
            v = fn(forms)
            t1 = time.perf_counter()
            rows.append({"task": "tester", "method": method, "size": D,
                         "seconds": t1 - t0, "value": v})
            print("tester", method, D, round(t1 - t0, 4), v, flush=True)
    with open("results/benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "method", "size", "seconds", "value"])
        w.writeheader()
        w.writerows(rows)
    mem = []
    for method, N in (("sieve", 10 ** 7), ("spf", 10 ** 6), ("divscan", 10 ** 4)):
        gc.collect()
        tracemalloc.start()
        fns[method](N)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        mem.append({"task": "range", "method": method, "size": N, "peak_bytes": peak})
        print("memory", method, N, peak, flush=True)
    with open("results/memory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "method", "size", "peak_bytes"])
        w.writeheader()
        w.writerows(mem)


if __name__ == "__main__":
    run()
