import math
import random
from math import gcd, isqrt

import numpy as np


def is_squarefree(n):
    if n <= 0:
        return False
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True


def reduce_form(A, B, C):
    steps = 0
    while True:
        steps += 1
        if C < A:
            A, B, C = C, -B, A
            continue
        if B <= -A or B > A:
            k = (B + A) // (2 * A)
            B2 = B - 2 * A * k
            C2 = A * k * k - B * k + C
            if B2 == -A:
                B2 = A
            B, C = B2, C2
            continue
        return A, B, C, steps


def is_wr_form(A, B, C):
    a, b, c, _ = reduce_form(A, B, C)
    return a == c


def is_wr_form_enum(A, B, C):
    disc = 4 * A * C - B * B
    ybound = isqrt(4 * A * A // disc + 1) + 2
    best = None
    for y in range(-ybound, ybound + 1):
        xc = -B * y / (2 * A)
        for x in range(int(math.floor(xc)) - 2, int(math.ceil(xc)) + 3):
            if x == 0 and y == 0:
                continue
            v = A * x * x + B * x * y + C * y * y
            if best is None or v < best:
                best = v
    mins = []
    for y in range(-ybound, ybound + 1):
        xc = -B * y / (2 * A)
        for x in range(int(math.floor(xc)) - 2, int(math.ceil(xc)) + 3):
            if x == 0 and y == 0:
                continue
            v = A * x * x + B * x * y + C * y * y
            if v == best:
                mins.append((x, y))
    for x1, y1 in mins:
        for x2, y2 in mins:
            if x1 * y2 - x2 * y1 != 0:
                return True
    return False


def ideal_gram(D, eps, a, b):
    if (eps * D) % 4 == 1:
        Bo = 2 * b + 1
        num = Bo * Bo - eps * D
        if num % 4 != 0 or (num // 4) % a != 0:
            return None
        return 4 * a * a, 4 * a * Bo, Bo * Bo + D
    num = b * b - eps * D
    if num % a != 0:
        return None
    return a * a, 2 * a * b, b * b + D


def is_wr_ideal(D, eps, a, b):
    G = ideal_gram(D, eps, a, b)
    if G is None:
        return None
    A, B, C = G
    g = gcd(gcd(A, B), C)
    return is_wr_form(A // g, B // g, C // g)


def wr_ideals_bruteforce(D, eps, Amax):
    out = []
    for a in range(1, Amax + 1):
        for b in range(a):
            r = is_wr_ideal(D, eps, a, b)
            if r:
                out.append((a, b))
    return out


def divisor_pairs3(D):
    out = []
    d = 1
    while d * d <= D:
        if D % d == 0:
            e = D // d
            if d < e <= 3 * d:
                out.append((d, e))
        d += 1
    return out


def nearsquare_divscan(D):
    d = 1
    while d * d <= D:
        if D % d == 0 and d < D // d <= 3 * d:
            return True
        d += 1
    return False


def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = 128
        g = r = q = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = gcd(q, n)
                k += m
            r <<= 1
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
        if g != n:
            return g


def _is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def factorize(n):
    if n == 1:
        return {}
    if _is_prime(n):
        return {n: 1}
    d = _pollard_brent(n)
    f1 = factorize(d)
    f2 = factorize(n // d)
    for p, e in f2.items():
        f1[p] = f1.get(p, 0) + e
    return f1


def divisors_from_factors(fac):
    divs = [1]
    for p, e in fac.items():
        divs = [d * p ** k for d in divs for k in range(e + 1)]
    return divs


def nearsquare_factor(D):
    divs = divisors_from_factors(factorize(D))
    for d in divs:
        e = D // d
        if d < e <= 3 * d:
            return True
    return False


def nearsquare_sieve(N):
    near = np.zeros(N + 1, dtype=bool)
    for d in range(1, isqrt(N) + 1):
        top = min(3 * d, N // d)
        if top > d:
            near[d * (d + 1):d * top + 1:d] = True
    return near


def squarefree_sieve(N):
    sf = np.ones(N + 1, dtype=bool)
    sf[0] = False
    for p in range(2, isqrt(N) + 1):
        sf[p * p::p * p] = False
    return sf


def wr_field_indicator(N):
    near = nearsquare_sieve(N)
    sf = squarefree_sieve(N)
    odd = np.zeros(N + 1, dtype=bool)
    odd[1::2] = True
    return near & sf & odd


def classify_field(D, eps):
    if D % 2 == 0 or not nearsquare_factor(D):
        return [], []
    prs = divisor_pairs3(D)
    ideals = []
    half = (eps * D) % 4 == 1
    for d1, d2 in prs:
        for d in (d1, d2):
            if half:
                ideals.append((d, (d - 1) // 2))
            else:
                ideals.append((2 * d, d))
    return prs, ideals


def pair_count_sieve(N):
    total = np.zeros(N + 1, dtype=np.int32)
    for d in range(1, isqrt(N) + 1):
        top = min(3 * d, N // d)
        if top > d:
            total[d * (d + 1):d * top + 1:d] += 1
    return total
