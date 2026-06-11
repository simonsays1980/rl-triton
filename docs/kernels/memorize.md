# Triton Kernel Concepts — Mental Models

## Memory layout: how a 2D tensor lives in RAM

PyTorch stores a 2D tensor as a single flat block of memory, row by row.

```
Tensor shape [3 envs, 4 timesteps]:

  t=0   t=1   t=2   t=3
  1.0   2.0   3.0   4.0   ← env 0
  5.0   6.0   7.0   8.0   ← env 1
  9.0  10.0  11.0  12.0   ← env 2

In memory (one flat line):
  [1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,  9.0, 10.0, 11.0, 12.0]
   ^--- env 0            ^--- env 1             ^--- env 2
   index 0               index 4                index 8
```

---

## Stride

**Stride** is the answer to: "how many slots do I skip in memory to move from one row to the next?"

For the tensor above, stride = 4 (the number of timesteps).  
For a normal contiguous tensor, `stride(0) == seq_len`.

```python
base = env_idx * stride_env
# env 0: base = 0 * 4 = 0   → points at 1.0
# env 1: base = 1 * 4 = 4   → points at 5.0
# env 2: base = 2 * 4 = 8   → points at 9.0
```

Within a row, elements are next to each other in memory (stride 1), so
`base + t` gives the element at timestep `t` for that environment.

---

## Lanes

A GPU executes the **same instruction on many values at once**.

Imagine 8 workers standing in a line, each holding one value. You shout one
instruction and all 8 execute it simultaneously. Each worker is a **lane**.
The group of workers is the **vector**.

```
worker:   0     1     2     3     4     5     6     7
value:   [δ7,  δ6,  δ5,  δ4,  δ3,  δ2,  δ1,  δ0]
```

This maps directly to GPU hardware: a group of threads (called a **warp** on
NVIDIA GPUs) executes the same instruction at the same time, each thread
working on its own lane.

---

## `tl.arange(0, BLOCK_SIZE)` — giving each worker a unique index

```python
offsets = tl.arange(0, BLOCK_SIZE)
# → [0, 1, 2, 3, 4, 5, 6, 7]
```

This hands each worker their own index number. All subsequent operations
happen across all workers at once. Worker 0 holds 0, worker 1 holds 1, etc.
Each worker uses its index to fetch a **different** element from memory —
all at the same time.

---

## `rev_offsets` — loading the sequence backwards

GAE is a backward recurrence: `A[t] = δ[t] + decay[t] * A[t+1]`.
We need to load the sequence from the last timestep down to the first.

```python
rev_offsets = seq_len - 1 - offsets
```

```
seq_len = 5,  BLOCK_SIZE = 8:

worker:        0    1    2    3    4    5    6    7
offsets:      [0,   1,   2,   3,   4,   5,   6,   7]
rev_offsets:  [4,   3,   2,   1,   0,  -1,  -2,  -3]
```

- Worker 0 fetches timestep 4 (the last one).
- Worker 4 fetches timestep 0 (the first one).
- Workers 5–7 get negative indices — out of bounds — and are masked out.

All 8 fetches happen simultaneously.

---

## Mask — handling sequences shorter than BLOCK_SIZE

```python
mask = offsets < seq_len
# → [T, T, T, T, T, F, F, F]  for seq_len=5, BLOCK_SIZE=8
```

Workers whose `rev_offsets` go negative are inactive. `tl.load(..., mask=mask, other=0.0)`
fills those lanes with 0 (the identity for the scan), so they contribute
nothing to the result.

---

## BLOCK_SIZE — how many workers to hire

`BLOCK_SIZE` must be:
1. **A power of 2** — hardware requirement on NVIDIA GPUs.
2. **At least `seq_len`** — so every timestep gets a worker.

```python
BLOCK_SIZE = triton.next_power_of_2(seq_len)
# seq_len=333  →  BLOCK_SIZE=512
# seq_len=512  →  BLOCK_SIZE=512
# seq_len=513  →  BLOCK_SIZE=1024
```

Workers covering positions beyond `seq_len` are masked out and do no useful
work — they are the inevitable padding cost of requiring a power-of-2 size.

---

## Putting it all together (GAE flat kernel)

```
seq_len=5, BLOCK_SIZE=8, env_idx=1, stride_env=5

base = 1 * 5 = 5   ← start of env 1's row in memory

worker:        0    1    2    3    4    5    6    7
rev_offsets:  [4,   3,   2,   1,   0,  -1,  -2,  -3]
ptr:          [9,   8,   7,   6,   5,   ✗,   ✗,   ✗]   (base + rev_offsets)
loads:        [δ4, δ3,  δ2,  δ1,  δ0,  0,   0,   0]   (masked lanes get 0)
```

The associative scan then runs left-to-right across the 8 workers —
which is right-to-left in time — computing the GAE recurrence in parallel.
Results are written back using the same `rev_offsets`, restoring time order.
