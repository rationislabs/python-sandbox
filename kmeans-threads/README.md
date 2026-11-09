# kmeans-threads

One k-means written five ways and timed on Python 3.15t with the GIL off and on. The question:
for real numeric work, when do pure-Python threads without the GIL beat processes, and how far
behind numpy do they stay?

Lloyd's k-means, a fixed 10 iterations, 200,000 points in 4 dimensions, 8 clusters. The start
centres come from a seeded k-means++ and are shared, so every variant does the same work and
must end at the same centres (relative 1e-9) and the same labels as `loop`; the bench refuses
to write numbers otherwise.

| Variant | How |
|---|---|
| `loop` | Pure Python, one thread. The baseline. |
| `threads` | Pure Python, one chunk of points per worker in a `ThreadPoolExecutor`; each returns per-centre sums and counts, the main thread adds them up. |
| `processes` | The same chunks in a `ProcessPoolExecutor`. Each worker gets every point once (pool initializer); a task is a chunk index plus the centres. |
| `numpy` | Vectorised, one thread. |
| `numpy-threads` | numpy per chunk in threads. numpy releases the GIL in its loops, so this scales with the GIL on too: the control case. |

## Results

Machine: AMD Ryzen 5 2500U with Radeon Vega Mobile Gfx, 8 CPUs, Python 3.15.0, numpy 2.5.4

Median of 3 runs. "pool start-up" is timed apart from the fit: starting the workers and, for
processes, sending each one the 200,000 points. Load average was 4.4 just before the run (other
programs were open); outside load hurts the many-worker rows more than the one-worker rows, so
the parallel speedups here are, if anything, on the low side.

| variant | workers | GIL | seconds | pool start-up s | speedup vs loop |
|---|---:|---|---:|---:|---:|
| loop | 1 | off | 14.041 | 0.000 | 1.00 |
| threads | 1 | off | 17.981 | 0.000 | 0.78 |
| threads | 2 | off | 12.519 | 0.001 | 1.12 |
| threads | 4 | off | 9.227 | 0.001 | 1.52 |
| threads | 8 | off | 7.857 | 0.002 | 1.79 |
| processes | 1 | off | 14.253 | 0.180 | 0.99 |
| processes | 2 | off | 9.274 | 0.296 | 1.51 |
| processes | 4 | off | 6.593 | 0.566 | 2.13 |
| processes | 8 | off | 5.976 | 1.133 | 2.35 |
| numpy | 1 | off | 1.007 | 0.000 | 13.94 |
| numpy-threads | 1 | off | 1.031 | 0.000 | 13.61 |
| numpy-threads | 2 | off | 0.692 | 0.001 | 20.30 |
| numpy-threads | 4 | off | 0.564 | 0.001 | 24.91 |
| numpy-threads | 8 | off | 0.499 | 0.002 | 28.13 |
| loop | 1 | on | 14.268 | 0.000 | 1.00 |
| threads | 1 | on | 17.911 | 0.001 | 0.80 |
| threads | 2 | on | 33.716 | 0.001 | 0.42 |
| threads | 4 | on | 34.613 | 0.002 | 0.41 |
| threads | 8 | on | 35.275 | 0.005 | 0.40 |
| processes | 1 | on | 14.652 | 0.308 | 0.97 |
| processes | 2 | on | 9.116 | 0.544 | 1.57 |
| processes | 4 | on | 6.152 | 1.142 | 2.32 |
| processes | 8 | on | 5.628 | 2.255 | 2.54 |
| numpy | 1 | on | 1.010 | 0.000 | 14.13 |
| numpy-threads | 1 | on | 0.960 | 0.001 | 14.87 |
| numpy-threads | 2 | on | 0.652 | 0.001 | 21.90 |
| numpy-threads | 4 | on | 0.540 | 0.002 | 26.40 |
| numpy-threads | 8 | on | 0.439 | 0.003 | 32.48 |

The CPU has 4 cores and 8 threads, so the 8-worker rows run on SMT siblings.

What the table shows:

- **Free threads work, but modestly.** With the GIL off, pure-Python threads reach 1.8× at 8
  workers. With the GIL on, the same code gets slower as threads are added (0.4×): they only
  fight over the lock.
- **One free thread costs about 28%.** `threads` with 1 worker takes 18.0 s against `loop`'s
  14.0 s for the same calls. Every point tuple belongs to the main thread, and in the
  free-threaded build a thread touching objects it does not own takes the slower, atomic
  reference-count path. Measured on 50,000 points: 0.32 s in the main thread, 0.41 s in another
  thread on the main thread's data, 0.36 s when that thread built its own copy. Process workers
  unpickle their own copy, which is part of why `processes` with 1 worker matches `loop`.
- **Processes still win among the pure-Python variants** here: 2.35× at 8 workers, the GIL
  makes no difference to them, and shipping the data to the workers costs 0.2 to 2.3 s up front
  (more workers, more copies).
- **numpy beats all of it on one core**: 14× faster than `loop`, and 28 to 32× with threads.
  The numpy rows look the same with the GIL on, because numpy drops the GIL in its loops.

So on 3.15t, threads are now a real option for pure-Python CPU work without the cost of copying
data to processes, but on this machine they do not yet beat processes, and vectorising stays the
bigger win by an order of magnitude.

## Run it

```sh
cd kmeans-threads
uv run kmeans bench                 # full table, GIL off then GIL on (re-runs itself with PYTHON_GIL=1)
uv run kmeans bench --points 20000  # quick run
uv run kmeans check                 # every variant agrees with loop on a small data set
uv run pytest
```

`bench` writes `results.csv` with the machine, Python and numpy versions in every row. The tests
check that every variant and worker count gives `loop`'s centres and labels, that an empty
cluster keeps its centre, and that the bench refuses to label a run "GIL off" when the GIL is on.
