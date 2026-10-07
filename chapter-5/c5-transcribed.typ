#import "../template/layout.typ": with-layout
#show: with-layout

= Scaling Book Exercises -- Chapter 5

#block[
  #link("https://github.com/kevinlopezandrade/scaling-book-solutions/blob/main/chapter-5/c5-handwritten.pdf")[Handwritten solutions].
  Watch me solve:
  #link("https://www.youtube.com/watch?v=NaH30lmvm-o")[Part~0]
] <chapter-resources>

== Shared setup from the book

Let's use LLaMA-2 13B as a basic model for this section. Here are the model details:

#table(
  columns: 2,
  align: (left, right),
  table.header([*hyperparam*], [*value*]),
  [$L$], [$40$],
  [$D$], [$5,120$],
  [$F$], [$13824$],
  [$N$], [$40$],
  [$K$], [$40$],
  [$H$], [$128$],
  [$V$], [$32,000$],
)

LLaMA-2 has separate embedding and output matrices and a gated MLP block.

== Exercise 1 -- LLaMA-2 13B parameter count

=== Exercise statement

How many parameters does LLaMA-2 13B have (I know that's silly but do the
math)? _Note that, as in Transformer Math, LLaMA-3 has 3 big FFW matrices, two
up-projection and one down-projection. We ignored the two "gating" einsum
matrices in this section, but they behave the same as $W_("in")$ in this
section._

=== Solution

Scan pages: 1

From LLAMA2 we know that $N = K$. Therefore:

$
  "MLP Params" &= 3 D F "per layer" \
  "Attention Params" &= 4 D N H "per layer" \
  "Embeddings" &= 2 D V "overall".
$

In total we have $(3 D F + 4 D N H) dot L + 2 D V$ parameters. Let $L = 40$, $N H = D$, $D = 5120$, $F = 13824$, and $V = 32,000$. Hence:

$
  (3 dot 5120 dot 13824 + 4 (5120)^2) dot 40 + 2 dot 5120 dot 32 e 3
  = 13,015,449,600 approx 13 dot 10^9 = 13 " Billion".
$

== Exercise 2 -- Memory for BS=16M training

=== Exercise statement

Let's assume we're training with BS=16M tokens and using Adam. Ignoring parallelism for a moment, how much total memory is used by the model's parameters, optimizer state, and activations? _Assume we store the parameters in bf16 and the optimizer state in fp32 and checkpoint activations three times per layer (after the three big matmuls)._ 

=== Solution

Scan pages: 1-2

Model Params $= 2 dot 13 e 9 = 26 e 9 " Bytes" = 26 "GB".$

Optimizer State:

$
  4 dot 13 e 9 + 4 dot 13 e 9
    = 8 dot 13 e 9
    = 104 "GB".
$

Activation Checkpointing:

$
  L dot (2 B T F + 2 B T F + 2 B T D)
    &= 40 dot (4 B T F + 2 B T D) \
    &= 40 dot (B T) dot (4 F + 2 D) \
    &= 40 dot (16 e 6) dot (4 dot 13824 + 2 dot 5120) \
    &approx 42 "TB".
$

== Exercise 3 -- 32k training on TPU v5p 16x16x16

=== Exercise statement

Assume we want to train with 32k sequence length and a total batch size of 3M tokens on a TPUv5p 16x16x16 slice. Assume we want to use bfloat16 weights and a float32 optimizer, as above.

+ Can we use pure data parallelism? Why or why not?
+ Can we use pure FSDP? Why or why not? With pure FSDP, how much memory will be used per device (assume we do gradient checkpointing only after the 3 big FFW matrices).
+ Can we use mixed FSDP + tensor parallelism? Why or why not? If so, what should $X$ and $Y$ be? How much memory will be stored per device? Using only roofline FLOPs estimates and ignoring attention, how long will each training step take at 40% MFU?

=== Solution

Scan pages: 2-11

Let:

$
  "HBM v5p" &= 96 "GB" \
  "bf16 FLOPs/s v5p" &approx 4.59 e 14 \
  W_("ICI") " v5p" &approx 1.8 e 11.
$

==== 3.1

We cannot use data parallelism since a lower bound in memory per chip we need is $130 "GB"$. (In fact we need more if we consider the activation checkpointing.)

==== 3.2

Assume FSDP shards the whole architecture as follows. For the attention params:
$
W_(q)[D_X, N, H] quad
W_(k)[D_X, N, H] quad
W_(v)[D_X, N, H] quad
W_(o)[N, H, D_X]
$
and for the MLP params as usual:
$
W_("in1")[D_X, F] quad
W_("in2")[D_X, F] quad
W_("out")[F, D_X] quad
$
and for input output embeddings as:
$
W_("embed")[V, D_X] quad
W_("outembed")[D_X, V] quad
$

Therefore the total params per device is:
$
  (3 frac(D, |X|) F + 4 frac(D, |X|) H N) dot L + 2 frac(D, |X|) V
  &= frac(1, |X|) dot ((3 D F + 4 D H N) dot L + 2 D V) \
  &approx frac("Model Params", |X|).
$

For LLAMA2 13B therefore:

$
  "Device Params"
    = frac(13 dot 10^9, 16^3)
    approx 3.2 dot 10^6
    = 3.2 " Million".
$

Assuming optimizer states also sharded, we get that each device needs at least:

$
  10 dot (3.2 dot 10^6) = 32 dot 10^6 = 32 "MB".
$

If we also take into account the activation checkpointings which are of shapes:
$
[B_X, T, F] quad
[B_X, T, F] quad
[B_X, T, D]
$

we get:

$
  "Checkpoints per device"
    &= L dot (2 frac(B, |X|) dot T D + 4 frac(B, |X|) T F) " Bytes" \
    &= frac(1, |X|) dot (L dot (B T) dot (4 F + 2 D)) " Bytes".
$

Let $B T = 3 dot 10^6$, then:

$
  frac(1, 16^3) dot (40 dot (3 dot 10^6) dot (4 dot 13824 + 2 dot 5120))
  approx 1.9 "GB".
$

So memory used per device is $32 "MB" + 1.96 "GB"$.

We could use FSDP since with FSDP sharding everything fits in memory, but we would be communication bound since we are compute bound iff:

$
  frac(B, |X|) > frac(2550, M_X)
    <=> frac(M_X dot B, 2550) > |X|.
$

Let $B = 3 dot 10^6$ and $M_X = 3$, therefore:

$
  frac(9 dot 10^6, 2550) approx 3530 > |X|,
$

and in our slice $|X| = 4046$.

==== 3.3

Assume we shard the attention matrices as:

$
W_(q)[D_X, N_Y, H] quad
W_(k)[D_X, N_Y, H] quad
W_(v)[D_X, N_Y, H] quad
W_(o)[N_Y, H, D_X]
$

and for the MLP params as:

$
W_("in1")[D_X, F_Y] quad
W_("in2")[D_X, F_Y] quad
W_("out")[F_Y, D_X]
$

and for input and output embeddings as:

$
V_("embed")[V_Y, D_X] quad
V_("outembed")[D_X, V_Y]
$

Therefore total params per device is:

$
  (3 frac(D, |X| |Y|) F + 4 frac(D, |X|) dot frac(H N, |Y|)) dot L
    + 2 frac(D, |X| |Y|) V 
  &= frac(1, |X| |Y|) dot (L dot (3 D F + 4 D H N) + 2 D V) \
  &= frac("Model Params", |X| |Y|).
$

For LLAMA2 13B we get:

$
  "Params per device"
    = frac(13 dot 10^9, (16)^3)
    approx 3.2 " Million".
$

If we take into account activation state then we get:

$
  10 dot (3.2 dot 10^6) = 32 "MB".
$

Activation checkpoints will be sharded with shapes:

$
[B_X, T, F_Y] quad
[B_X, T, F_Y] quad
[B_X, T, D_Y]
$

so we get:

$
  "Checkpoints per device"
    &= L dot (2 frac(B, |X|) dot T frac(D, |Y|)
      + 4 frac(B, |X|) T frac(F, |Y|)) "Bytes" \
    &= frac(1, |X| |Y|) dot (L dot (B T) dot (4 F + 2 D)) "Bytes".
$

Let $B T = 3 dot 10^6$, then:

$
  frac(1, 16^3) dot (40 dot (3 dot 10^6) dot (4 dot 13824 + 2 dot 5120))
  approx 1.9 "GB".
$

Therefore for memory used per device we get $approx 1.9 "GB"$.

If we use FSDP and TP to minimize communications we need to set
$
  X_("opt") = sqrt(frac(B, F) dot frac(M_X, M_Y) dot N)
$

Let $N = 16^3$, $M_X = 2$, $M_Y = 1$, $B = 3 dot 10^6$, and $F = 13824$. Therefore:

$
  X_("opt")
    = sqrt(frac(3 dot 10^6, 13824) dot 2 dot 16^3)
    approx 1333.3 approx 1333.
$

Since $N = X_("opt") dot Y => frac(16^3, 1333) = Y = 3.07.$
To get integer values therefore we choose $X_("opt") = 1024$ and $Y = 4$, which is closer to our optimum.
Having chosen this, we will be compute bound iff:

$
  frac(B, N) > frac(alpha^2, M_X M_Y F)
$

where $alpha = frac(C, W_("ICI")) = 2550$ for v5p.

Let $F = 13824$, $N = 16^3$, $M_X = 2$, $M_Y = 1$, and $B = 3 dot 10^6$:

$
  frac(B, N) &= frac(3 dot 10^6, 16^3) approx 733 \
  frac(alpha^2, M_X M_Y dot F) &= frac((2550)^2, 2 dot 13824) = 235.
$

therefore we are compute bound.

Assuming we are compute bound, using only roofline FLOPs estimates and ignoring attention we get that:

$
  "Time per Step"
    = frac(3 e 6 dot 13 e 9 dot 6, 16^3 dot (0.4 dot 4.59 e 14))
    = 0.312 "s"
    = 312 "ms".
$

== Transcription uncertainties

- The final subpart is labeled `3.2` again in the scan; I preserved the repeated handwritten label.
- On scan page 6, the slice size appears to be written as `$|X| = 4046$`; I marked it as uncertain in the transcription.
