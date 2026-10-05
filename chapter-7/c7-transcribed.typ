#import "../template/layout.typ": with-layout
#show: with-layout

= Scaling Book Exercises -- Chapter 7

Transcribed from handwritten solutions.
#link("https://github.com/kevinlopezandrade/scaling-book-solutions/blob/main/chapter-7/c7-handwritten.pdf")[Source]

== Shared model setup for Exercises 1-7

The chapter uses an invented model based on LLaMA-2 13B with
$L = 64$ (`num_layers`), $D = 4096$ (`d_model`), $F = 16384$
(`ffw_dimension`), $N = 32$ (`num_heads`), $K = 8$
(`num_kv_heads`), $H = 256$ (`qkv_dim`), and $V = 32128$
(`num_embeddings`).

== Exercise 1 -- Parameter count and int8 KV cache

=== Exercise statement

How many parameters does the above model have? How large are its KV caches per token in int8? _You can assume we share the input and output projection matrices._

=== Solution

Scan pages: 1

From previous chapters we know that number of params is:

$
  "Params"
    &= (3 D F + 2 D (N + K) dot H) dot L + D V \
    &= (3 dot 4096 dot (4 dot 4096)
       + 2 dot 4096 dot 40 dot 256) dot 64
       + 4096 dot 32128 \
    &= "18 385 207 246"
       approx 18 dot 10^9
       = 18 "Billion".
$

Per-token KV cache size is:

$
  frac(2 dot S dot L dot K dot H, S) = 2 dot L dot K dot H.
$

Hence in int8:

$
  2 dot 64 dot 8 dot 256
    = 262144 "bytes"
    approx 262 "KB".
$

== Exercise 2 -- Largest batch size at 128K context

=== Exercise statement

Say we want to serve this model on a TPUv5e 4x4 slice and can fully shard our KV cache over this topology. What's the largest batch size we can fit, assuming we use int8 for everything and want to support 128k sequences? What if we dropped the number of KV heads to 1?

=== Solution

Scan pages: 2-3

Assume TPU v5e 4x4 slice where v5e HBM = $16 "GB"$.

Note that:

$
  "Space per shard" = frac(M + B 2 S L K H, |X|).
$

$
  frac(M, |X|)
    + frac(B dot 2 dot S dot L dot K dot H, |X|)
    <= "HBM".
$

where $M$ is total params in int8. Therefore the largest batch size we can fit in is:

$
  frac(M, |X|)
    + frac(B dot 2 dot S dot L dot K dot H, |X|)
    &<= "HBM" \
  <=> frac(M, |X|) - "HBM"
    &<= -frac(B dot 2 dot S dot L dot K dot H, |X|) \
  <=> frac(|X| dot "HBM" - M, |X|)
    &>= frac(B dot 2 dot S dot L dot K dot H, |X|) \
  <=> frac(|X| dot "HBM" - M, 2 dot S dot L dot K dot H)
    &>= B.
$

For $S = 128 "K"$ we get:

$
  B
    <= frac(
      16 dot (16 dot 10^9) - 18 dot 10^9,
      2 dot 128 dot 10^3 dot 64 dot 8 dot 256,
    )
    = 7.04
    approx 7.
$

And if we let $K = 1$ we get:

$
  B
    <= frac(
      16 dot 16 dot 10^9 - 18 dot 10^9,
      2 dot 128 dot 10^3 dot 64 dot 1 dot 256,
    )
    approx 56.
$

== Exercise 3 -- Parameter-loading latency

=== Exercise statement

How long does it take to load all the parameters into the MXU from HBM assuming they're fully sharded on a TPU v5e 4x4 slice? Assume int8 parameters. _This is a good lower bound on the per-step latency._

=== Solution

Scan pages: 3

Assume TPU v5e 4x4 slice where v5e $"BW" = 8.2 dot 10^11$.
Therefore to load $frac(M, |X|)$ into the MXU it takes:

$
  frac(M, |X|) dot frac(1, "BW")
    = frac(18 dot 10^9, 16) dot frac(1, 8.2 dot 10^11)
    approx 0.0013 "s"
    = 1.3 "ms".
$

== Exercise 4 -- Prefill and generation sharding

=== Exercise statement

Let's say we want to serve this model on a TPUv5e 4x4 slice using int8 FLOPs and parameters/activations. How would we shard it for both prefill and decode? _Hint: maybe answer these questions first:_

1. What does ICI look like on a 4x4?
2. What's the roofline bound on tensor parallelism?
3. How can we shard the KV caches?

For this sharding, what is the rough per-step latency for generation?

=== Solution

Scan pages: 4-6

Assume TPU v5e 4x4 slice. With this slice we don't have wraparound links. To shard it both for prefill and generation, since during generation FSDP does not help, we could use tensor parallelism sharded over all the devices for both prefill and generation.

For prefill we are compute bound when:

$
  Y < F dot frac(W_("ICI"), C) dot 2.
$

Let $F = 4 dot 4096$, $W_("ICI") = 9 dot 10^10$, and
$C = 3.94 dot 10^14$. Hence:

$
  Y
    < 16384 dot frac(9 dot 10^10, 3.94 dot 10^14) dot 2
    approx 7.5.
$

So we cannot do tensor parallelism with 16 devices and be compute bound.

We could instead do TP and CP with 4 way TP and 4 way CP, to be compute bound. So then a roofline bound will be:

$
  t = frac(4 dot B D F, X Y C),
  quad "where" quad B = 7 dot 128 "K".
$

During generation we will be memory bound instead of communication bound when using tensor parallelism when:

$
  Y < F dot frac(1, B) dot frac(W_("ICI"), W_("HBM")).
$

where $W_("ICI") / W_("HBM") approx 1 / 8$ in a TPU v5e.
Hence:

$
  Y < 4 dot 4096 dot frac(1, B) dot frac(1, 8)
    = frac(2048, B).
$

Since for $S = 128 "K"$ the maximum batch size is 7, then we get:

$
  Y < frac(2048, 7) = 292.
$

Therefore during generation we will be memory bound and a roofline bound will be:

$
  t = frac(2 D F, X dot W_("HBM")).
$

We can shard the KV caches as:

$
  "KV"[2, B_X, S, K_Y, H]
$

where the mesh is defined as `{X: 2, Y: 8}`.

A rough per-step latency under this sharding is:

$
  frac(B times "KV cache size" + "Parameter size", 16 dot W_("HBM"))
    &= frac(
      7 times (2 dot 128 dot 10^3 dot 64 dot 8 dot 256)
        + 18 dot 10^9,
      16 dot 8.2 dot 10^11,
    ) \
    &approx 0.02 "s".
$

== Exercise 5 -- MoE parameter and FLOP accounting

=== Exercise statement

Let's pretend the above model is actually an MoE. An MoE model is effectively a dense model with $E$ copies of the FFW block. Each token passes through $k$ of the FFW blocks and these $k$ are averaged to produce the output. Let's use $E=16$ and $k=2$ with the above settings.

1. How many total and activated parameters does it have? _Activated means used by any given token._
2. What batch size is needed to become FLOPs bound on TPU v5e?
3. How large are its KV caches per token?
4. How many FLOPs are involved in a forward pass with $T$ tokens?

=== Solution

Scan pages: 7-8

*5.1*

Let $E = 16$ and $k = 2$, then:

$
  "Total Params"
    = (3 E D F + 2 D (N + K) dot H) dot L + D V
    approx 212 "Billion".
$

$
  "Activated Params"
    = (3 k D F + 2 D (N + K) dot H) dot L + D V
    approx 31 "Billion".
$

*5.2*

From the approximation of the MLP part they use in this chapter, to be FLOPs bound we need $T_("math") > T_("comms")$:

$
  frac(2 dot B dot 3 k D F, C)
    &> frac(2 dot 3 dot E D F, "BW") \
  <=> frac(2 dot B dot 3 k D F, 2 dot 3 dot E D F)
    &> frac(C, "BW") \
  <=> B dot frac(k, E)
    &> frac(C, "BW") \
  <=> B
    &> frac(E, k) dot frac(C, "BW").
$

*5.3*

KV cache size per token does not change.

*5.4*

Using the formula from Chapter 4:

$
  "FLOPs"
    = 6 dot B T (3 k D F + 2 dot D (N + K) dot H) dot L.
$

== Exercise 6 -- Expert sharding and minimum slice

=== Exercise statement

With MoEs, we can do "expert sharding", where we split our experts across one axis of our mesh. In our standard notation, our first FFW weight has shape `[E, D, F]` and we shard it as $[E_Z, D_X, F_Y]$ where $X$ is only used during training as our FSDP dimension. Let's say we want to do inference on a TPU v5e:

1. What's the HBM weight loading time for the above model on a TPU v5e 8x16 slice with $Y=8$, $Z=16$? How much free HBM is available per TPU?
2. What is the smallest slice we could fit our model on?

=== Solution

Scan pages: 8-10

*6.1*

Assuming we shard the MLPs as:

$
  W_("in1")[E_Z, D, F_Y],
  quad W_("in2")[E_Z, D, F_Y],
  quad W_("out")[E_Z, F_Y, D].
$

The attention params as:

$
  W_q[D, N_Y, H],
  quad W_k[D, K_Y, H],
  quad W_v[D, K_Y, H],
  quad W_o[N_Y, H, D].
$

and the shared input-output embedding as:

$
  V[V_Y, D].
$

The total params per device is:

$
  "Params Per Device"
    = (3 frac(E, |Z|) D frac(F, |Y|)
       + 2 frac(D, |Y|) (N + K) H) dot L
       + frac(D V, |Y|)
    approx 2.3 "Billion".
$

Hence the weight loading time in bf16:

$
  frac(2 dot 2.3 dot 10^9, 8.2 dot 10^11)
    approx 0.005 "s"
    = 5 "ms".
$

As free space we have approximately:

$
  16 "GB" - 4.6 "GB" = 11.4 "GB".
$

*6.2*

Fit $Z = 16$ so that we can shard our expert copies, then we need to choose $Y$ so that:

$
  2 dot frac(1, |Y|)
    ((3 D F + 2 D (N + K) dot H) dot L + D V)
    &<= 16 \
  <=> frac(2 dot 18 dot 10^9, 16)
    &<= |Y| \
  <=> |Y|
    &>= 2.25.
$

Here if we need a slice with axis sizes being multiple of 2, the smallest slice we could fit our model on is $4 times 16$.

== Exercise 7 -- 2D model sharding

=== Exercise statement

Here we'll work through the math of what the #link("https://arxiv.org/pdf/2211.05102")[ESTI paper] calls 2D weight-stationary sharding. We describe this briefly in Appendix B, but try doing this problem first to see if you can work out the math. The basic idea of 2D weight stationary sharding is to shard our weights along both the $D$ and $F$ axes so that each chunk is roughly square. This reduces the comms load and allows us to scale slightly farther.

Here's the algorithm for 2D weight stationary:

1. $"In"[B, D_X] = "AllGather"_(Y Z)("In"[B, D_(X Y Z)])$
2. $"Tmp"[B, F_(Y Z)] \{U_X\} = "In"[B, D_X] dot_D W_("in")[D_X, F_(Y Z)]$
3. $"Tmp"[B, F_(Y Z)] = "AllReduce"_X("Tmp"[B, F_(Y Z)] \{U_X\})$
4. $"Out"[B, D_X] \{U_(Y Z)\} = "Tmp"[B, F_(Y Z)] dot_F W_("out")[F_(Y Z), D_X]$
5. $"Out"[B, D_(X Y Z)] = "ReduceScatter"_(Y Z)("Out"[B, D_X] \{U_(Y Z)\})$

Your goal is to work out $T_("math")$ and $T_("comms")$ for this algorithm and find when it will outperform traditional 3D model sharding?

=== Solution

Scan pages: 10-14

Let $N = |X| dot |Y| dot |Z|$ be the total number of devices in the slice. From the algorithm we can see that:

$
  T_("comms")
    &= frac(2 B D, |X|) dot frac(1, 2 dot W_("ICI"))
       + frac(2 dot 2 dot B F, |Y| dot |Z|) dot frac(1, W_("ICI"))
       + frac(2 B D, |X|) dot frac(1, 2 dot W_("ICI")) \
    &= frac(2 B D, |X| dot W_("ICI"))
       + frac(4 B F dot |X|, N dot W_("ICI")) \
    &= frac(1, W_("ICI"))
       (frac(2 B D, |X|) + frac(4 B F dot |X|, N)).
$

$
  T_("math")
    &= 2 dot B dot frac(D, |X|) dot frac(F, |Y| dot |Z|) dot frac(1, C)
       + 2 dot B dot frac(F, |Y| dot |Z|) dot frac(D, |X|) dot frac(1, C) \
    &= frac(4 B D F, |X| dot |Y| dot |Z|) dot frac(1, C) \
    &= frac(4 B D F, N dot C).
$

Note that given $N$ devices, $T_("comms")$ can change based in how we allocate devices for the different axes $X$, $Y$, $Z$. Therefore we need to find the setting where $T_("comms")$ is minimum.

$
  T_("comms")(|X|)
    = frac(1, W_("ICI"))
      (frac(2 B D, |X|) + frac(4 B F dot |X|, N)).
$

$
  frac(partial T_("comms"), partial |X|)
    = frac(1, W_("ICI"))
      (2 B D dot (-1) dot frac(1, |X|^2) + frac(4 B F, N)).
$

Solving $frac(partial T_("comms"), partial |X|) = 0$, we get a candidate for a stationary point:

$
  frac(4 B F, N)
    &= frac(2 B D, |X|^2) \
  <=> frac(N, 4 B F) dot 2 B D
    &= |X|^2 \
  <=> |X|^2
    &= frac(D, 2 F) dot N.
$

Since $|X|$ can only be positive in our case, we get:

$
  X^* = sqrt(frac(D, 2 F) dot N).
$

Let $F = 4D$ to get:

$
  X^* = sqrt(frac(N, 8)).
$

Since $frac(partial^2 T_("comms"), partial |X|^2) > 0$ at $|X| = X^*$, we know that $X^*$ is a global minimum and $T_("comms")$ becomes:

$
  T_("comms")
    &= frac(1, W_("ICI"))
       (frac(2 B D sqrt(8), sqrt(N))
        + frac(4 B F sqrt(N), N sqrt(8))) \
    &= frac(1, W_("ICI"))
       (frac(2 B D sqrt(8), sqrt(N))
        + frac(4 B F, sqrt(N) sqrt(8))) \
    &= frac(1, W_("ICI"))
       (frac(16 B D, sqrt(8) sqrt(N))
        + frac(4 B F, sqrt(N) sqrt(8))) \
    &= frac(1, sqrt(8) dot W_("ICI"))
       (frac(4 B dot (4D + F), sqrt(N))) \
    &= frac(1, sqrt(8) dot W_("ICI"))
       (frac(4 B dot 8D, sqrt(N))) \
    &= frac(1, sqrt(8) dot W_("ICI")) dot frac(32 B D, sqrt(N)).
$

$
  frac(T_("math"), T_("comms")) > 1
    &<=> frac(4 B D F, N C)
       dot frac(sqrt(8) W_("ICI") sqrt(N), 32 B D) > 1 \
    &<=> frac(4 sqrt(8) W_("ICI") sqrt(N) F, 32 N C) > 1 \
    &<=> frac(4 sqrt(8), 32) dot frac(1, sqrt(N))
       dot frac(W_("ICI") F, C) > 1 \
    &<=> frac(16 dot 8, 32^2) dot frac(1, N)
       dot frac(W_("ICI")^2 F^2, C^2) > 1 \
    &<=> frac(16 dot 8, 32^2) dot frac(F^2, alpha^2) > N \
    &<=> frac(1, 8) (frac(F, alpha))^2 > N.
$

If we compare to traditional 3D model sharding where the condition to be compute bound is:

$
  M_y dot frac(F, alpha) > N.
$

We see that with 2D weight stationary:

$
  frac(
    frac(1, 8) (frac(F, alpha))^2,
    M_(x y z) (frac(F, alpha)),
  )
  = frac(1, 8 M_(x y z)) dot frac(F, alpha).
$

During inference however if we are communication bound in the MLP because of a small batch size, we see that:

$
  frac(T_("comms-2D"), T_("comms-3D"))
    &= frac(32, sqrt(8)) dot frac(B D, W_("ICI") sqrt(N))
       dot frac(W_("ICI") M_(x y z), 4 B D) \
    &= frac(32, sqrt(8) dot 4) dot frac(M_(x y z), sqrt(N)) \
    &= 2 dot sqrt(2) dot frac(M_(x y z), sqrt(N)).
$
