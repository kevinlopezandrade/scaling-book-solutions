#import "../template/layout.typ": with-layout
#show: with-layout

= Scaling Book Exercises -- Chapter 6

Chapter: 6 -- Training LLaMA 3 on TPUs\
Source: handwritten Onyx Boox A4 PDF\
Note: Transcribed from handwritten solutions; book markdown used only for exercise statements and notation.

Remark: The following questions don't appear numbered in the scaling book. I will number them in order of appearance as 0.X where $X in NN$ and $X >= 1$.

== Exercise 0.1 -- LLaMA 3 FLOPs per token

=== Exercise statement

How many FLOPs does LLaMA-3 perform per token per training step? This helps us determine how expensive the whole training process will be.

=== Solution

Scan pages: 1.

Let $M = 70.4 e 9$. We know that

$
  "Transformer FLOPs" approx 6 dot B T dot M.
$

Therefore per token we know its GMFLOPs:

$
  6 M "FLOPs" = 422.4 e 9.
$

== Exercise 0.2 -- Total FLOPs for 15T tokens

=== Exercise statement

LLaMA 3 was trained for about 15 trillion tokens. How many FLOPs is that total?

=== Solution

Scan pages: 1.

Let $M = 70.4 e 9$ and $B T = 15 e 12$, hence:

$
  "Transformer FLOPs" = 6 dot B T dot M
    = (6 dot 70.4 dot 15) dot 10^21
    = 6336 dot 10^21
    approx 6.3 e 24
    = 6.3 "Yotta FLOPs".
$

== Exercise 0.3 -- Training time on a full TPU v5p pod

=== Exercise statement

Let's say we wanted to train on a full TPU v5p pod with 16x20x28 = 8960 chips. How long would this take to train at 40% MFU in bfloat16, assuming we are compute-bound?

=== Solution

Scan pages: 2.

Assume TPU v5p pod of $16 times 20 times 28 = 8960$ chips. Let v5p bf16 FLOPs/s $= 4.59 e 14$.

We know that FLOPs per chip is

$
  frac(1, 8960) dot 6 dot B T dot M.
$

Let $M = 70.4 e 9$ and $B T = 15 e 12$. Therefore

$
  "Time" = frac(6 dot B T dot M, 8960 dot 0.4 dot 4.59 e 14)
    = 44.3 "days".
$

== Exercise 0.4 -- Minimum TPUs for the 4M-token batch

=== Exercise statement

LLaMA 3-70B was pretrained with a batch size of about 4M tokens. How many TPUs do we need at minimum to train with this batch size? You can assume bfloat16 parameters and float32 optimizer state, and that you checkpoint gradients 4 times per layer.

=== Solution

Scan pages: 2--3.

Assume we checkpoint the three big matmuls in the MLP part and

$
  A = "softmax"(Q dot K^T) dot V,
$

which has shape $A[B, T, K, G, H]$, so it occupies $2 dot B T K G H = 2 B T D$.

Hence, combining our formula from exercise 5.2, we get that:

$
  "Activation Checkpointing" = L dot (4 B T F + 4 B T D).
$

Therefore total memory required is:

$
  10 dot M + L dot (B T) dot (4F + 4D),
$

where $M$ is number of model parameters. Let $B T = 4 e 6$. Therefore for LLaMA 3-70B we get:

$
  10 dot (70.4 e 9) + 80 dot (4 e 6) dot (4 dot 28672 + 4 dot 8192)
  approx 47.8 e 12
  = 47.8 "TB".
$

So assuming v5p TPU which has HBM capacity $= 96 "GB"$, we need at least:

$
  frac(47.8 e 12, 96 e 9) approx 498
$

TPUs.

== Exercise 0.5 -- Memory per chip on 8960 TPU v5p chips

=== Exercise statement

Under the same assumptions as the question above, if we use 8960 TPU v5p chips, how much memory will we use per-chip?

=== Solution

Scan pages: 4.

$
  "Memory per chip" = frac(47.8 e 12, 8960) "bytes"
    approx 5.3 e 9 "bytes"
    = 5.3 "GB".
$

== Exercise 0.6 -- FSDP alone without sequence/context parallelism

=== Exercise statement

Under the assumptions above, can we train our model with FSDP alone? To start, let's say we can't do any sequence/context parallelism. This should be the first idea you have, since it's simple and will introduce no extra communication if it works.

=== Solution

Scan pages: 4--6.

Modeling only the MLP part and using sharding strategy from [sequence parallelism, Shenggui Li et al.], both FSDP and sequence parallelism perform:

$
  "In"[B_X, L_Y, D] dot_D W_("in")[D_X, F] dot_F W_("out")[F, D_X].
$

So without sequence parallelism we get:

$
  "In"[B_X, L, D] dot_D W_("in")[D_X, F] dot_F W_("out")[F, D_X].
$

Hence:

$
  T_("math") = frac(2 dot 2 dot B dot L dot D dot F, X dot C) \
  T_("comms") = frac(2 dot 2 dot D dot F, W_("ICI"))
$

Therefore for being compute bound in this setting we need:

$
  frac(4 B L D F, 4 X D F) > 2550
    <=> frac(B L, X) > 2550 \
  => frac(B, X) > frac(2550, L).
$

From the exercise we know that $L = 4096$, therefore:

$
  frac(B, X) > 0.62.
$

Since $frac(B, X) = 0.11$, we cannot train with FSDP alone.

== Exercise 0.7 -- FSDP with sequence/context parallelism

=== Exercise statement

Let's relax the requirement of not doing any sequence sharding. If we allow ourselves to do FSDP over both the batch and sequence axes, can we train LLaMA 3-70B with only FSDP on 8960 chips?

=== Solution

Scan pages: 6--7.

Again modeling only the MLP part we can see that:

$
  "In"[B, L, D] dot_D W_("in")[D, F] dot_F W_("out")[F, D]
$

is equivalent to:

$
  "In"[B dot L, D] dot_D W_("in")[D, F] dot_F W_("out")[F, D].
$

Since no attention involved we just have to reshape after finishing. Therefore with FSDP we will do:

$
  "In"[(B dot L)_X, D] dot_D W_("in")[D_X, F] dot W_("out")[F, D].
$

Hence the condition to be compute bound is:

$
  frac(B dot L, X) > 2550.
$

Since

$
  frac(1024 dot 4096, 8960) approx 468,
$

we cannot use FSDP if we want to be compute bound.

== Exercise 0.8 -- Mixed tensor parallelism and FSDP

=== Exercise statement

Now let's look at mixed tensor parallelism and FSDP. Does there exist some combination that lets us remain compute-bound? What amount of FSDP and tensor parallelism should we do if so?

=== Solution

Scan pages: 7.

We require

$
  frac(B dot L, N) > frac(alpha^2, M_x M_y dot F)
$

and $X$ to be

$
  X_("opt") = sqrt(frac(B dot L, F) dot frac(M_x, M_y) dot N),
$

to be compute bound. Since $frac(B L, N) approx 468$ and

$
  frac(alpha^2, M_x M_y dot F) approx 113,
$

if we choose $X = 2240$, since is close to our $X_("opt")$, we will be compute bound.

== Exercise 1 -- Scaling LLaMA 70B to more chips

=== Exercise statement

Say we want to train LLaMA 3-70B on 4 pods with the same batch size. What parallelism scheme would we use? Would we be compute or communication bound? Roughly how long would it take to train? Make sure to use the correct roofline bound.

=== Solution

Scan pages: 8--11.

I need the per pod time to be compute bound, and we can use either FSDP or FSDP + TP. We can use data parallel over the DCN network and FSDP + TP. So let $B = N dot B_("local")$ where $N in NN$ and reshape $"In"[B, D]$ to $"In"[N, B_("local"), D]$.

So the sharding strategy will be:

$
  "In"[N_("DCN"), B_("local", X), D_Y]
    dot_D W_("in")[D_X, F_Y]
    dot_F W_("out")[F_Y, D_X].
$

Note that $T = max(T_("pod"), T_("comms-dcn"))$, where $T_("pod") = max(T_("math"), T_("comms"))$. To be compute bound we need:

$
  T_("comms-dcn") < T_("pod") and T_("math") > T_("comms") \
  => T_("math") > T_("comms-dcn").
$

$
  T_("comms-dcn") = frac(8 D F, |X| dot |Y|).
$

Assume $T_("math") > T_("comms")$ and note that in the backward pass:

$
  T_("math")
    = frac(N, "DCN") dot frac(B_("local"), |X|) dot D dot frac(F, |Y|) dot 2 dot 4 \
    = frac(8 B D F, "DCN" dot |X| dot |Y|).
$

Therefore if $T_("math") > T_("comms")$, the first condition to be compute bound is satisfied if:

$
  frac(8 B D F, "DCN" dot |X| dot |Y|)
    dot frac(|X| dot |Y|, 8 D F)
    > frac(C, W_("DCN"))
    = 73440 \
  => frac(B, "DCN") > 73440,
$

where DCN is the number of pods connected by DCN.

Now to satisfy condition 2 we need, as in the exercise before:

$
  frac(B_("local"), |X| dot |Y|)
    > frac(alpha^2, M_x M_y dot F)
    approx 113,
$

and choose $X$ integer close to $X_("opt")$ as defined before. Since $B = N dot B_("local")$ and $N > "DCN"$, we get:

$
  B_("local") > |X| dot |Y| dot 113 \
  => N dot B_("local") > N dot |X| dot |Y| dot 113
      >> "DCN" dot |X| dot |Y| dot 113 \
  => B > "DCN" dot |X| dot |Y| dot 113.
$

So if we choose $B = 4 e 6$ and $N = 4$, we satisfy the conditions.

The time required for training is roughly, assuming $40%$ MFU in bfloat16,

$
  frac(6 dot 1 e 6 dot 70 e 9, 8960 dot 0.4 dot 4.59 e 14)
    approx 255 "ms"
$

per 4M batch step. Therefore

$
  frac(15 e 12, 4 e 6) dot 255 dot 10^(-3)
    approx 11 "days".
$

== Exercise 2.a -- LLaMA 405B hyperparameters and FLOPs

=== Exercise statement

Using the LLaMA 3-405B config, write a table with all the key hyperparameters as above. How many total parameters does this model have? How many FLOPs per training step? How many FLOPs do we perform if we train for 15T tokens?

=== Solution

Scan pages: 11--12.

$
  L = 126, quad D = 16384, quad F = 53248, \
  N = 128, quad K = 8, quad H = 128, quad V = 128256.
$

Using formulas from previous section since architecture does not change, just its hyperparams, we get:

$
  "Parameters" = 405 e 9 \
  "FLOPs per step" = 6 dot 405 e 9 approx 2.4 "TFLOPs" \
  "FLOPs total" = 6 dot 15 e 12 dot 405 e 9
    = 3.645 e 25
    approx 36.45 "Yotta FLOPs".
$

== Exercise 2.b -- LLaMA 405B on 8 TPU v5p pods

=== Exercise statement

Assume we want to train on 8 TPU v5p pods. What parallelism scheme would we use? How long would training take? Would we be compute or comms bound?

=== Solution

Scan pages: 12--13.

Using our derivation from question 1, we can use that same sharding strategy and be compute bound if:

$
  frac(B, "DCN") > 73440
    and B > "DCN" dot |X| dot |Y| dot 113
    and N = "DCN"
    and B_("local") = frac(B, N).
$

Choosing $B = 16 e 6$ so that we are safely in the compute bound, we get $B_("local") = 2 e 6$, and choosing $X$ integer close to $X_("opt")$. In this case $X = 1120$ and $Y = 8$.

A rough estimate for training time would be:

$
  frac(6 dot 2 e 6 dot 405 e 9, 8960 dot 0.4 dot 4.59 e 14)
    dot frac(15 e 12, 16 e 6)
    approx 32 "days".
$
