Seconds to simulate one million paths (best of 3). 12 logical CPU cores. Speedups are relative to the NumPy version.

| Option | Pure Python | NumPy | C++, 1 thread | C++, 12 threads | Fastest vs pure Python |
|:--|--:|--:|--:|--:|--:|
| European call | 0.339 s (0.1×) | 0.033 s (1.0×) | 0.021 s (1.5×) | 0.009 s (3.8×) | 39× |
| Asian call (52 fixings) | 15.112 s (0.1×) | 1.407 s (1.0×) | 1.239 s (1.1×) | 0.329 s (4.3×) | 46× |
| Barrier call (252 checks) | 70.809 s (0.1×) | 5.830 s (1.0×) | 4.163 s (1.4×) | 0.820 s (7.1×) | 86× |
