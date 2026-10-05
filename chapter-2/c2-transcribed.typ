#import "../template/layout.typ": with-layout
#show: with-layout

= Scaling Book Exercises — Chapter 2

Transcribed from handwritten solutions.
#link("https://github.com/kevinlopezandrade/scaling-book-solutions/blob/main/chapter-2/c2-handwritten.pdf")[Source]

This document contains only the handwritten solution content from the scan, with exercise statements copied from the Chapter 2 book markdown and kept separate from the solutions.

== Exercise 1 — Bounding LLM latency

=== Exercise statement

Say you want to sample from a 200B parameter model in bf16 that's split across 32 TPU v4p. How long would it take to load all the parameters from HBM into the systolic array? Hint: use the numbers above.

=== Solution

Scan pages: 1-2

v4p $=>$
  - HBM BW = $1.2 e 12$
  - FLOPs/s bf16 = $2.75 e 14$
  - HBM capacity = $32 "GB"$

$200 "Billion" = 200 dot 10^9$ (bf16). Split across 32 TPUs v4p $=>$ Bytes per TPU:
$
  frac(200 dot 10^9 dot 2, 32)
    = frac(400 dot 10^9, 32)
    = 12.5 dot 10^9 " bytes"
    = 12.5 "GB"
$

$12.5 "GB" <= 32 "GB".$ So it does fit.

Each TPU HBM has $12.5 "GB"$ of weights to send to the Tensor Cores:
$
  T_("comms")
    = frac(12.5 dot 10^9 " bytes", 1.2 e 12 " bytes/s")
    approx 10 dot 10^(-3) " s"
    = 10 " ms".
$

== Exercise 2 — TPU details

=== Exercise statement

Consider a full TPU v5e pod. How many total CPU hosts are there? How many TPU TensorCores? What is the total FLOPs/s for the whole pod? What is the total HBM? Do the same exercise for TPU v5p pod.

=== Solution

Scan pages: 2-5

*TPU v5e pod*

- Pod size: $16 times 16$
- Host size: $4 times 2$
- HBM per chip: $16 "GB"$
- FLOPs bf16 per chip = $1.97 e 14$
- Host size: $2 times 4 = 8$

$16 times 16$: how many blocks of $2 times 4$ fit:

```text
One host (schematic):
+----+----+----+----+
|    |    |    |    |
+----+----+----+----+
|    |    |    |    |
+----+----+----+----+
```

4 hosts to cover one "row", 8 hosts to cover one "column" $=> 32 " hosts".$ I could also just flatten:
$16 times 16 = 256$

```text
[ ][ ][ ][ ][ ][ ][ ][ ] ...
<---------- 8 ---------> is a host
```
and then divide i.e $256/8 = 32$ hosts.

TPU v5e has only one Tensor Core per chip $=> 256 "Tensor Cores".$ Also you
can derive by cores per host and from the book you see using previous result:
$32 times 8 = 256 "Tensor Cores"$ where 32 is hosts in the pod and 8 is cores per host.

Total FLOPs/s = $256 times 1.97 e 14 approx 500 e 14$

Total HBM = $256 times 16 dot 10^9 = 4096 e 9 = 4096 " GB".$

*TPU v5p pod*

- Pod Size: $16 times 20 times 28$
- Host Size: $2 times 2 times 1$

Flatten $=>$ $16 times 20 times 28 = 8960 " chips".$

Total Hosts = $ frac(8960, 4) = 2240$

Tensor Cores = $8960 times 2 = 17920$

Total HBM:
$
  8960 times 96 dot 10^9
    = 860160 dot 10^9
    approx 860 dot 10^3 dot 10^9
    = 860 " TB".
$

Total FLOPs/s:
$
  8960 times 4.59 e 14
    approx 41000 times 10^14
    approx 4.1 "Exa FLOPs/s".
$


== Exercise 3 — PCIe operational intensity

=== Exercise statement

Imagine we're forced to store a big weight matrix $A$ of type `bf16[D, F]`, and a batch of activations $x$ of type `bf16[B, D]` in host DRAM and want to do a matrix multiplication on them. This is running on a single host, and we're using a single TPU v6e chip attached to it. You can assume $B << D$, and $F = 4D$. What is the smallest batch size $B$ we need to remain FLOPs bound over PCIe? Assume PCIe bandwidth of `1.6e10` bytes / second.

=== Solution

Scan pages: 6-7

```text
[Host] -- PCIe --> [TPU]
```
Assume:
- TPU v6e.
- $"bf16" [B, D] dot_D "bf16" [D, F] -> "bf16" [B, F]$
- $B << D$, $F = 4D$.
- PCIe BW = $1.6 e 10 " bytes/s"$.
- FLOPs/s = $9.20 e 14$.

Assuming I can overlap communication with compute:
$
  "AI" := frac(2 dot B D F, 2 B D + 2 D F + 2 B F)
    >= frac(9.20 dot 10^14, 1.6 dot 10^10)
$

Using $F = 4D$ and $B << D$:

$
  "AI"
    approx frac(2 dot B dot D dot 4D, 2 dot D dot 4D)
    = frac(B dot D^2, D^2)
$

$=> B >= 5.75 dot 10^4 = 57.5 dot 10^3$

Like 57 sequences of 1024 seq len.

== Exercise 4 — General matmul latency

=== Exercise statement

Let's say we want to multiply a weight matrix `int8[16384, 4096]` by an activation matrix of size `int8[B, 4096]` where $B$ is some unknown batch size. Let's say we're on 1 TPU v5e to start.

1. How long will this multiplication take as a function of $B$? Hint: it may help to calculate how long it will take to load the arrays from HBM and how long the multiplication will actually take. Which is bottlenecking you?
2. What if we wanted to run this operation out of VMEM? How long would it take as a function of $B$?

=== Solution

Scan pages: 7-13

v5e:
  - HBM BW = $8.2 e 11$
  - FLOPs/s int8 = $3.94 e 14$

VMEM BW $approx$ HBM BW $times 22$ $approx 8.2 dot 10^11 times 22 = 180.4 times 10^11 " bytes/s".$

$"int8" [B, 4096] dot "int8" [4096, 16384] -> "int8" [B, 16384]$

$T = max(T_("comms"), T_("math"))$

Note: $16384 = 4 dot 4096$, so let $D = 4096$ and $F = 4096 dot 4 => F = 4 dot D$ as before.

$[B, D] dot [D, 4D] -> [B, 4D]$

$
  T_("comms")
    = frac(B dot D + 4 dot D^2 + B dot 4 dot D, 8.2 e 11)
    = frac(B dot (D + 4 dot D) + 4 dot D^2, 8.2 e 11)
    = frac(B dot (5D) + 4 dot D^2, 8.2 e 11)
$

$
  T_("math")
    = frac(2 dot B dot D dot F, 3.94 e 14)
    = frac(8 dot B dot D^2, 3.94 e 14)
$

$
  T_("operation")
    = max(T_("comms")(B), T_("math")(B))
    = max(
      frac(B dot 5 dot D + 4 dot D^2, 8.2 e 11),
      frac(8 dot B dot D^2, 3.94 e 14)
    )
$

$
  => T_("operation")(B, "BW")
    = max(
      frac(B dot 5 dot D + 4 dot D^2, "BW"),
      frac(8 dot B dot D^2, 3.94 e 14)
    )
$

Since: VMEM = $128 " MiB" = 128 dot 10^6$.

$4096 times (4 dot 4096) approx 68 dot 10^6$. At least the weights do fit in VMEM.

Bottleneck: compute bound when?
$
  T_("comms")(B, "BW") <= T_("math")(B)
  <=> frac(B dot 5 dot D + 4 dot D^2, "BW")
    <= frac(8 dot B dot D^2, 3.94 e 14 " FLOPs/s") \
  <=> frac(B dot 5 dot D + 4 dot D^2, 8 dot B dot D^2)
    <= frac("BW", 3.94 e 14)
    = "AI"^(-1)
  <=> frac(B dot 5 dot D, 8 dot B dot D^2)
    + frac(4 dot D^2, 8 dot B dot D^2)
    <= "AI"^(-1) \
  <=> 0 <= frac(5, 8 dot D) + frac(4, 8 dot B) <= "AI"^(-1)
  <=> frac(4, 8 dot B) <= "AI"^(-1) - frac(5, 8 dot D) \
  <=> frac(1, 2 dot B) <= "AI"^(-1) - frac(5, 8 dot D)
  <=> frac(1, B) <= 2 dot "AI"^(-1) - frac(10, 8 dot D) \
  <=> B >= frac(1, 2 dot (1 / "AI") - frac(10, 8 dot D))
  <=> B >= frac(1, 2 dot (frac("BW", 3.94 e 14)) - 3 dot 10^(-4))
$

$"BW"_("HBM") = 8.2 e 11$
$
  B >= frac(1, 2 dot frac(8.2 e 11, 3.94 e 14) - 3 dot 10^(-4))
    approx frac(1, 4 dot 10^(-3))
    = (4 dot 10^(-3))^(-1)
    = frac(1, 4) dot 10^3
    = 250
$

$"BW"_("VMEM") = 180.4 e 11$
$
  B >= frac(1, 2 dot frac(180.4 e 11, 3.94 e 14) - 3 dot 10^(-4))
    approx 11.
$


== Exercise 5 — ICI bandwidth

=== Exercise statement

Let's say we have a TPU v5e `4x4` slice. Let's say we want to send an array of type `bf16[8, 128, 8192]` from `TPU{0,0}` to `TPU{3, 3}`. Let's say the per-hop latency for TPU v5e is $1 mu s$.

1. How soon will the first byte arrive at its destination?
2. How long will the total transfer take?

=== Solution

Scan pages: 14-17

4x4 slice, no wraparound.\
Source = [00] = TPU{0,0}; receiver = [33] = TPU{3,3}.

P1: across the top row, then down the right column.\
P2: down the left column, then across the bottom row.

```text
[00]--P1-->[01]--P1-->[02]--P1-->[03]
  |                                |
  P2                               P1
  v                                v
[10]       [11]       [12]       [13]
  |                                |
  P2                               P1
  v                                v
[20]       [21]       [22]       [23]
  |                                |
  P2                               P1
  v                                v
[30]--P2-->[31]--P2-->[32]--P2-->[33]
```
v5e $=> (4,4)$ slice has no wraparound.

Assume $T_("comms, hop") = alpha + frac(D, "BW")$ for one hop.
Then, assuming pipelining:

$
  T_("comms") = "#Hops" dot alpha + frac(D, "BW")
  quad "where" alpha = 1 mu s
$
$"#Hops" = 6 =>$ first byte arrives after $6 mu s$.

Different algos, but a lower bound on time is:
$
  T >= frac(D, 2 dot "BW")
$
since it needs to receive $D$ bytes at $2*"BW"$:
```text
           |
           |
           v
----> [TPU[3, 3]]
```



*Algo 1*

Data Split = $frac(2 times 8 times 128 times 8192, 2)$

path1 = P1 arrows: [00] -> [01] -> [02] -> [03] -> [13] -> [23] -> [33]\
path2 = P2 arrows: [00] -> [10] -> [20] -> [30] -> [31] -> [32] -> [33]

$
  => max(
    frac(D " (bytes)", 2 dot "BW"_("path1")),
    frac(D " (bytes)", 2 dot "BW"_("path2"))
  )
  = frac(D, 2 dot "BW")
  >= " lower bound"
$

*Algo 2*

Everything through path 1:

$
  max(frac(D, "BW"_("path1")), 0)
    = frac(D, "BW"_("path1"))
    > " lower bound"
$

BW = $4.5 e 10$. Therefore:
$
  frac(D, 2)
    = frac(2 times 8 times 128 times 8192, 2)
    = 2^3 dot 2^7 dot 2^13
    = 2^23
$

$
  => frac(2^23, 4.5 dot 10^10)
    approx 0.00018
    = 0.18 dot 10^(-3) " s"
$

$T_("comms") = 6 mu s + 0.18 " ms".$

== Exercise 6 — Pulling it all together

=== Exercise statement

Imagine you have a big matrix $A$: `int8[128 * 1024, 128 * 1024]` sharded evenly across a TPU v5e 4x4 slice but offloaded to host DRAM on each chip. Let's say you want to copy the entire array to `TPU{0, 0}` and multiply it by a vector `bf16[8, 128 * 1024]`. How long will this take? Hint: use the numbers above.

=== Solution

Scan pages: 18-20

```text
Question 6 setup, schematic:

Host 0 (4x2)             Host 1 (4x2)
+----+----+              +----+----+
|    |----|--------------|    |    |
+----+----+              +----+----+
|    |    |              |    |    |
+----+----+              +----+----+
|    |    |              |    |    |
+----+----+              +----+----+
|    |    |              |    |    |
+----+----+              +----+----+
```

No wraparound links.

$"int8" A[128 dot 1024, 128 dot 1024]$. Assume each host DRAM has an even part of the array $A => frac(128^2 dot 1024^2, 2) approx 8.6 "GB"$
per host, so it fits in v5e HBM, which is $16 "GB"$.

*Path*

```text
[Host_B] --PCIe--> [TPU] --ICI--> [TPU[0,0]]

[Host_A] --PCIe--> [TPU[0,0]]
```

Assuming pipelining:
$
  T_("comms")(A)
    := frac(8.6 "GB", "BW"_("PCIe"))
    = frac(8.6 dot 10^9, 1.6 dot 10^10)
    approx 5.3 dot 10^(-1) "s"
    = 0.53 " s"
$

$"Bytes in" "bf16"[8, 128 times 1024] = 8 times 128 times 1024 times 2 approx 2 "GB"$. So let's assume it's in HBM already.
Assuming intended operation is:
$
[128 times 1024, 128 times 1024] @ [128 times 1024, 8]
$
$
  => T
    = max(T_("comms"), T_("math"))
    = max(0.53 "s", frac((128 dot 1024)^2 dot 8 dot 2, 1.97 e 14))
    approx max(0.53 "s", 0.001 "s")
    = 0.53 "s".
$
