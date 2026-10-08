#import "../template/layout.typ": with-layout
#show: with-layout

= Scaling Book Exercises -- Chapter 8

#block[
  #link("https://github.com/kevinlopezandrade/scaling-book-solutions/blob/main/chapter-8/c8-handwritten.pdf")[Handwritten solutions].
  Watch me solve:
  #link("https://www.youtube.com/watch?v=RYWXjw-Vp48")[Part~0]
] <chapter-resources>

*Remark:* The following questions don't appear numbered in the Scaling Book. I will number them in order of appearance as 0.X, where $X in NN$ and $X >= 1$.

== Exercise 0.1 -- LLaMA 3-70B KV cache size

=== Exercise statement

How large are LLaMA 3-70B's KV caches per token? Assume we store them in int8. This determines how large our batch size can be on a given topology.

=== Solution

Scan pages: 1

Per-token KV cache size is:

$
  frac(2 S L K H, S) = 2 L K H
$

where $S$ is sequence length. Hence, in int8:

$
  2 dot 80 dot 8 dot 128 = 163840 "Bytes" approx 164 "KB"
$

== Exercise 0.2 -- Memory use and smallest slice

=== Exercise statement

Let's say we want to serve L3 70B at batch size 32 and 8192 sequence length with everything (params and KVs) in int8. How much total memory will this use? What's the smallest slice we could serve this on?

=== Solution

Scan pages: 1--2

Let $M$ be the number of parameters of the model, let $B = 32$ and $S = 8192$. Assume int8.

$
  "Total Memory"
    &= M + B S dot 2 L K H \
    &= 70 e 9 + B S dot 164 e 3 \
    &approx 113 "GB"
$

Assuming we serve this model in TPU v5e, the smallest possible slice is a $4 times 2$ since it gives us $128 "GB"$ of aggregated HBM.

== Exercise 0.3 -- Decode latency and throughput

=== Exercise statement

At this batch size and quantization on a TPU v5e 4x2, roughly what latency would we expect per decode step? What throughput (tokens / sec / chip)? What about a 4x4? Assume we perform our FLOPs in bfloat16 and everything is fully sharded.

=== Solution

Scan pages: 2--3

At $B = 32$ the MLP block is not compute bound. Therefore, using the general equation from chapter 6:

$
  "Step Time"
    = frac(B times "KV cache size", "BW" dot N)
      + frac(M, "BW" dot N)
$

where $N$ is the total number of devices in the slice. Hence:

$
  "Step Time"
    &= frac(32 dot 8192 dot 164 e 3, 8.2 e 11 dot 8)
      + frac(70 e 9, 8.2 e 11 dot 8) \
    &approx frac(113 e 9, 8.2 e 11 dot 8) \
    &approx 0.017 "s" = 17 "ms"
$

$
  "Tokens/s/chip"
    &= B dot frac(1, 0.017) dot frac(1, N) \
    &= frac(B, 0.017 dot N) \
    &= frac(32, 0.017 dot 8) \
    &= 235 "tokens/s/chip"
$

If instead we use a $4 times 4$ slice, we get:

$
  "Step Time"
    = frac(113 e 9, 8.2 e 11 dot 16)
    approx 0.008 "s"
    = 8 "ms"
$

$
  "Tokens/s/chip"
    = frac(32, 0.008 dot 16)
    = 250 "tokens/s/chip"
$

== Exercise 0.4 -- Compute-bound batch sizes

=== Exercise statement

On TPU v5e, using bfloat16 weights and activations, how large do our batch sizes need to be for us to be compute-bound in our matmuls? What if we do int8 weights but perform our FLOPs in bfloat16? What about int8 weights with int8 FLOPs?

=== Solution

Scan pages: 3--4

For bfloat16 weights and activations:

$
  "Intensity(Matmul)"
    = B
    >= frac(1.97 e 14, 8.2 e 11)
    approx 240
$

If int8 weights but FLOPs in bfloat16, note that:

$
  "Intensity(Matmul)"
    &= frac(2 B D F, B D + D F + B F) \
    &= 2 dot frac(B D F, B D + D F + B F) \
    &approx 2 B >= 240
    <=> B >= 120
$

If int8 weights and int8 FLOPs:

$
  "Intensity(Matmul)"
    = 2 B
    >= frac(2 dot 1.97 e 14, 8.2 e 11)
    <=> B >= 240
$

== Exercise 0.5 -- Smallest serving topology by precision

=== Exercise statement

What is the smallest TPU v5e topology we could serve LLaMA 3-70B on using bfloat16, int8, and int4 (both KVs and parameters) with 8k context? You can think of KV caches as negligibly small for this one.

=== Solution

Scan pages: 4--5

Assume $B = 1$.

For bfloat16:

$
  "Total Memory"
    &= 2 M + 2 B S dot 2 L K H \
    &= 2 dot 70 e 9 + 2 dot 8192 dot 164 e 3 \
    &approx 143 "GB"
$

So we need at least a $4 times 4$ TPU v5e slice.

For int8:

$
  "Total Memory"
    &= M + B S dot 2 L K H \
    &= 70 e 9 + 8192 dot 164 e 3 \
    &approx 71 "GB"
$

So we need at least a $4 times 2$ TPU v5e slice.

For int4:

$
  "Total Memory"
    = frac(M, 2) + frac(B S dot 2 L K H, 2)
    approx 35.5 "GB"
$

So we need at least a $2 times 2$ TPU v5e slice.

== Exercise 0.6 -- Generate-step latency at maximum batch size

=== Exercise statement

Assume we use the largest batch size that fits on these topologies. What latency could we expect for each generate step?

=== Solution

Scan pages: 5--6

For bfloat16 in a $4 times 4$ slice:

$
  frac(2 M + 2 B S dot 2 L K H, 4 times 4) <= 16 "GB"
$

Note that $B$ cannot be $B >= 240$, for otherwise our KV cache size will not fit in this topology. Hence we are memory bound, and choosing the max $B$ that satisfies the conditions just makes us closer to our upper bound. Therefore:

$
  "bfloat16 Step Time"
    = frac(16 e 9, 8.2 e 11)
    approx 0.019 "s"
    = 19 "ms"
$

For int8, $B$ cannot be $B >= 240$, hence as in the bfloat16 case:

$
  "int8 step time with max B" = 19 "ms"
$

For int4, the same argument as above holds, therefore:

$
  "int4 step time with max B" = 19 "ms"
$

== Exercise 0.7 -- Throughput per chip

=== Exercise statement

For each of these, what throughput per chip does this give us (in terms of queries / chip)? Assume our median decode length is 512 tokens.

=== Solution

Scan pages: 6--7

For each case we can use the condition we derived in Exercise 2 of chapter 7 to obtain the max $B$ that fits in each topology.

For int8 and a $2 times 4$ slice:

$
  B
    <= frac(abs(X) dot "HBM" - M, 2 S L K H)
    = frac(8 dot 16 - 70 e 9, 8192 dot 164 e 3)
    approx 42
$

For bfloat16 and a $4 times 4$ slice:

$
  B
    <= frac(abs(X) dot "HBM" - 2 M, 2 S dot 2 L K H)
    approx 43
$

For int4 and a $2 times 2$ slice:

$
  B
    <= frac(abs(X) dot "HBM" - frac(M, 2), frac(S dot 2 L K H, 2))
    approx 43
$

Therefore:

$
  "Queries/chip bfloat16, 4x4 slice"
    = frac(1, 0.014 dot 512) dot 42 dot frac(1, 16)
    approx 0.26
$

$
  "Queries/chip int8, 4x2 slice"
    = frac(1, 0.014 dot 512) dot 42 dot frac(1, 8)
    approx 0.53
$

$
  "Queries/chip int4, 2x2 slice"
    = frac(1, 0.014 dot 512) dot 42 dot frac(1, 4)
    approx 1.07
$

== Exercise 0.8 -- Doubling the topology

=== Exercise statement

How would our peak throughput change if we doubled our topology for each of the above examples?

=== Solution

Scan pages: 8

If we double our topology, we can see that:

$
  B
    <= frac(2 dot abs(X) dot "HBM" - M, S dot 2 L K H)
    approx 138
$

for all cases. Therefore, since:

$
  frac(138, 42) approx 3.3
$

we $1.65 times$ our peak throughput in each case.

== Exercise 0.9 -- Sharding on a TPU v5e 4x8

=== Exercise statement

Now let's dig into the question of sharding. Let's say we wanted to serve in bfloat16 on a TPU v5e 4x8. What sharding would we use for our model on a TPU v5e 4x8 during generation? Can we avoid being communication bound?

=== Solution

Scan pages: 8--9

The only sharding option we have is model parallelism. From chapter 7 we know that we will not be communication bound if:

$
  frac(2 D F, N W_("HBM"))
    > frac(2 B D, 2 W_("ICI"))
  <=> frac(2 F, B beta) > N
$

where:

$
  beta = frac(W_("HBM"), W_("ICI"))
$

Therefore, for our setting:

$
  N < frac(2 F, 138 dot 8) approx 52
$

So we can avoid being communication bound since $N = 32$ in a $4 times 8$ TPU v5e slice.

The KV cache will be sharded along the head dimension and among the batch dimension as well:

$
  "KV"[B_X, 2, S, L, K_Y, H]
$

where `Mesh = {X: 4, Y: 8}`.

== Exercise 0.10 -- Prefill time

=== Exercise statement

Assume we achieve a 40% FLOPs utilization during prefill. How long will a prefill of length 8192 take on 16 TPU v5e chips?

=== Solution

Scan pages: 9

We can use the formula:

$
  "Total FLOPs" = 2 M B
$

Hence:

$
  t
    = frac(2 M B, 16) dot frac(1, C dot 0.4)
    approx 0.9
$

where we assumed bfloat16 FLOPs.

== Exercise 0.11 -- Decode completions and KV eviction

=== Exercise statement

Assume we have a median prefill length of 8192 tokens and a median decode length of 4096 tokens. Say we have a generate batch size of 32. On average how many sequences finish decoding per step? On average how many tokens are evicted from our KV cache each step?

=== Solution

Scan pages: 10

At batch size 32 we are memory bound during decoding, therefore:

$
  "Step Time"
    = frac(32 dot 8192 dot 164 e 3 + 70 e 9, N W_("HBM"))
    approx 0.008 "s"
    = 8 "ms"
$

$
  "Sequence Time"
    = "Step Time" times 4096
    approx 35 "s"
$

$
  "Decodings Per Second"
    = frac(1, "StepTime" dot 4096) dot 32
    approx 0.9
$

$
  "Prefills Per Second"
    = frac(1, 0.9)
    approx 1.1
$

Per prefill step I do:

$
  frac(0.9 dot 32, 35) = 0.82 "decodings"
$

So per prefill step I evict:

$
  0.82 dot 4096 approx 3358 "tokens"
$

== Exercise 0.12 -- Prefill-to-generate server ratio

=== Exercise statement

Assume we do disaggregated serving with a median prefill length of 8192 and a median decode length of 512. Assume the prefill and generate latencies calculated above in bfloat16. What ratio of prefill:generate servers will you need to keep both fully saturated?

=== Solution

Scan pages: 11

Assuming each server produces:

$
  "Prefills/s" = frac(1, 0.9)
$

$
  "Decodings/s"
    = frac(32, 0.008 dot 512)
    = 7.81
$

Therefore we need to produce prefills at the same rate as the decoding servers:

$
  N dot frac(1, 0.9) = 7.81
  <=> N = 7.029
$

So we need 7 more prefill servers than decoding servers.

== Exercise 1 -- LLaMA 3-405B forward-pass bounds

=== Exercise statement

How many FLOPs does each forward pass for LLaMA 3-405B use per-token? Assuming we're FLOPs bound, what is a lower bound on a single forward pass on $N$ chips on TPU v5e? What if we're comms bound? Ignore the fact that the model does not fit on a single chip.

=== Solution

Scan pages: 12

We can use the formula from this chapter:

$
  "Total FLOPs" = 2 M B
$

Hence, per token:

$
  "FLOPs/Token"
    = 2 M
    = 2 dot 405 e 9
    = 810 e 9
$

If we are FLOPs bound, a lower bound on $N$ TPU v5e chips is:

$
  t
    >= frac(2 M, N C)
    = frac(810 e 9, N dot 1.97 e 14)
$

If we are communication bound, a lower bound is:

$
  t
    >= frac(2 B D, M_X dot W_("ICI"))
    = frac(2 dot B dot 16384, M_X dot 9.0 e 10)
$

== Exercise 2 -- LLaMA 3-8B serving memory

=== Exercise statement

Assume we want to serve LLaMA 3-8B with BS240 using int8 weights and int8 KV caches. How many bytes are used by (a) model parameters, (b) KV caches, and (c) peak working activations, roughly? What's the smallest topology we can run this on?

=== Solution

Scan pages: 13

For LLaMA 3-8B we have $L = 32$, $D = 4096$, $F = 14336$, $N = 32$, $K = 8$, $H = 128$, and $V = 128256$.

Assuming int8 for everything, we have:

$
  M approx 8 e 9 "Bytes"
$

$
  frac("KV caches Size", S)
    = frac(B S dot 2 L K H, S)
    = 240 dot 2 dot 32 dot 8 dot 128
    approx 15.7 "MB"
$

For peak working activations we will assume:

$
  "In"[B, S, D] dot_D W_("in")[D, F]
    -> "Out"[B, S, F]
$

Hence per token:

$
  B F
    = 240 dot 14336 "Bytes"
    approx 3.4 "MB"
$

are used. The smallest topology is a single TPU v5e.

== Exercise 3 -- LLaMA 3-405B serving under a 15 ms limit

=== Exercise statement

How would you serve LLaMA 3-405B on TPU v5e? Assume int8 weights and bfloat16 FLOPs. Let's say we have a firm limit of 15ms / token. What's the highest-throughput configuration we could achieve? What is the theoretical minimum step time?

=== Solution

Scan pages: 14--17

For LLaMA 3-405B we have $L = 126$, $D = 16384$, $F = 53248$, $N = 128$, $K = 8$, $H = 128$, and $V = 128256$. Assume int8 weights and bfloat16 FLOPs. Then we know that:

$
  "KV cache size/token"
    = 2 L K H
    = 2 dot 126 dot 8 dot 128
    approx 258 e 3 "Bytes"
    = 258 "KB"
$

and:

$
  "Total Memory"
    &= M + B S dot 2 L K H \
    &= 405 e 9 + B S dot 258 e 3
$

Hence we need a TPU v5e slice topology with at least $408 "GB"$ of aggregated HBM, assuming $B = 1$. Since HBM of TPU v5e is $16 "GB"$, we need:

$
  abs(X) dot 16 "GB" >= 408 "GB"
  <=> abs(X) >= 25.5
$

Note that a lower bound in latency is:

$
  frac(405 e 9, abs(X) dot W_("HBM")) <= "Step Time"
$

Hence, if we need to have a latency of $<= 15 "ms"$, we have to satisfy:

$
  frac(405 e 9, 0.015 dot 8.2 e 11) <= abs(X)
  <=> abs(X) >= 32.9
$

So we need more than 32 devices. We can consider an $8 times 8$ TPU v5e slice, and use model parallelism. During prefill we will be memory bound since:

$
  frac(F, 2188) = 24 gt.eq.not abs(X) = 64
$

However, we can improve our latency during generation.

Note that for a choice of $B$ we are memory bound if:

$
  T_("HBM comms") ("int8")
    = frac(D F, abs(X) dot W_("HBM"))
$

$
  T_("ICI comms") ("bfloat16")
    = frac(2 B D, W_("ICI"))
$

$
  frac(D F, abs(X) dot W_("HBM"))
    > frac(2 B D, W_("ICI"))
  <=> frac(F dot W_("ICI"), 2 dot abs(X) dot W_("HBM")) > B
$

So then, in order to be memory bound, we need:

$
  B <= 52.0
$

At this batch size we're memory bound in the MLP. We can solve now for our max $S$ length that still fits:

$
  S
    <= frac(abs(X) dot "HBM" - M, B dot 2 L K H)
    approx 46130
$

However, we need $S$ to be smaller, for otherwise we will not hit the $15 "ms"$ requirement.

$
  "Step time"
    = frac(52 dot S dot 258 e 3 + 405 e 9, 64 dot 8.2 e 11)
    <= 0.015
$

$
  S
    <= frac(64 dot 8.2 e 11 dot 0.015 - 405 e 9, 52 dot 258 e 3)
    approx 28488
$

Therefore, with $B = 52$ and $S = 28488$ we have:

$
  "Step Time" = 15 "ms"
$

and a throughput of:

$
  "Tokens/s"
    = frac(B, 0.015)
    = frac(52, 0.015)
    = 3466
$
