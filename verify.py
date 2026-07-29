from math import isqrt

import numpy as np

from wrlattices import (is_squarefree, reduce_form, is_wr_form, is_wr_form_enum,
                        is_wr_ideal, wr_ideals_bruteforce, divisor_pairs3,
                        nearsquare_divscan, nearsquare_factor, wr_field_indicator,
                        classify_field, ideal_gram)
from math import gcd


def check_reduction():
    import random
    random.seed(11)
    for _ in range(4000):
        A = random.randint(1, 40)
        B = random.randint(-80, 80)
        C = random.randint(1, 60)
        if 4 * A * C - B * B <= 0:
            continue
        a, b, c, _ = reduce_form(A, B, C)
        assert -a < b <= a <= c
        assert b * b - 4 * a * c == B * B - 4 * A * C
        assert is_wr_form(A, B, C) == is_wr_form_enum(A, B, C)
    print("reduction and WR test ok")


def check_classification():
    mism = 0
    for D in range(2, 200):
        if not is_squarefree(D):
            continue
        ns = nearsquare_divscan(D)
        assert ns == nearsquare_factor(D)
        for eps in (-1, 1):
            Amax = 3 * isqrt(3 * D) + 6
            wr = wr_ideals_bruteforce(D, eps, Amax)
            has = len(wr) > 0
            pred = (D % 2 == 1) and ns
            if has != pred:
                mism += 1
                print("MISMATCH", D, eps, has, pred, wr[:5])
    assert mism == 0
    print("classification iff verified for all squarefree D < 200, both signs")


def check_real_ideals():
    for D in (15, 21, 35, 91, 105, 1155, 15015):
        prs = divisor_pairs3(D)
        _, real_ideals = classify_field(D, 1)
        assert len(real_ideals) == 2 * len(prs)
        for a, b in real_ideals:
            assert is_wr_ideal(D, 1, a, b), (D, a, b)
        Amax = 3 * isqrt(3 * D) + 6
        if D <= 200:
            brute = wr_ideals_bruteforce(D, 1, Amax)
            assert sorted(brute) == sorted(real_ideals), (D, brute, real_ideals)
        _, im_ideals = classify_field(D, -1)
        for a, b in im_ideals:
            assert is_wr_ideal(D, -1, a, b), (D, -1, a, b)
    print("explicit WR ideal lists verified, real count equals 2*rho(D)")


def check_simclasses():
    for D in (15, 21, 1155, 15015):
        prs = divisor_pairs3(D)
        for eps in (-1, 1):
            Amax = 3 * isqrt(3 * D) + 6 if D < 3000 else 800
            forms = set()
            for a in range(1, Amax + 1):
                for b in range(a):
                    G = ideal_gram(D, eps, a, b)
                    if G is None:
                        continue
                    A, B, C = G
                    g = gcd(gcd(A, B), C)
                    rf = reduce_form(A // g, B // g, C // g)
                    if rf[0] == rf[2]:
                        forms.add((rf[0], abs(rf[1]), rf[2]))
            assert len(forms) == len(prs), (D, eps, forms, prs)
    print("similarity classes equal number of divisor pairs")


def check_sieve():
    N = 5000
    ind = wr_field_indicator(N)
    for D in range(1, N + 1):
        expected = is_squarefree(D) and D % 2 == 1 and nearsquare_divscan(D)
        assert bool(ind[D]) == expected, D
    print("range sieve agrees with per-D scan up to", N)


def check_hand_values():
    assert nearsquare_divscan(3) and nearsquare_divscan(15) and nearsquare_divscan(21)
    assert not nearsquare_divscan(5) and not nearsquare_divscan(7) and not nearsquare_divscan(11)
    assert is_wr_ideal(21, -1, 5, 2)
    assert is_wr_ideal(21, -1, 6, 3)
    assert is_wr_ideal(21, -1, 14, 7)
    assert is_wr_ideal(21, 1, 3, 1)
    assert is_wr_ideal(21, 1, 7, 3)
    assert is_wr_ideal(15, -1, 3, 1)
    assert not is_wr_ideal(5, -1, 2, 1)
    print("hand values ok")


if __name__ == "__main__":
    check_reduction()
    check_hand_values()
    check_classification()
    check_real_ideals()
    check_simclasses()
    check_sieve()
    print("all checks passed")
