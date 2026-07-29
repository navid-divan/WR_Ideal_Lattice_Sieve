# Well-rounded Ideal Lattices of Quadratic Fields

This is the implementation that classifies the well-rounded (WR) ideal lattices of real and imaginary quadratic fields, enumerates the fields that contain them, and reproduces the experimental findings of our research including tables and figures.

## Requirements
Python 3.10, NumPy, and Matplotlib. All arithmetic is exact integer arithmetic, and no floating point enters any reported count.

## Files
`wrlattices.py` is the core library (including Lagrange--Gauss reduction of positive definite binary forms), `verify.py` contains the correctness checks, `benchmark.py` is the wall-time and peak-memory benchmarks for the three range methods (pair sieve, factor-table scan, divisor scan), `density.py` is the density experiment, and `makeplots.py` generates figures from CSV with the distribution of the pair-counting function.

To reproduce the results, you can run in this order:
```
python3 verify.py
python3 benchmark.py
python3 density.py
python3 makeplots.py
```

## Method overview

A primitive ideal of the quadratic order is written as `a Z + (b + delta) Z`.
Our research shows that the associated planar lattice is well rounded exactly
when `D` is odd and `D = d e` for a divisor pair with `d < e <= 3 d`, for
both real and imaginary fields, and that the WR ideals themselves are given
by explicit formulas in terms of these divisor pairs. The classification
converts the search for WR ideals of a field into a divisor computation, and
the search for all admissible fields up to `N` into a sieve that marks the
products `d m` with `d < m <= 3 d`, in `O(N)` marks. The density of the
admissible fields tends to zero at the rate of the Erdos-Ford theory of
divisors in short intervals, which the density experiment represents.
