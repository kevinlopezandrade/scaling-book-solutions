
#set page(paper: "a4", margin: (x: 1.7cm, y: 1.7cm))
#set text(
  font: ("New Computer Modern", "CMU Serif", "Libertinus Serif", "Times New Roman"),
  lang: "en",
  size: 10.5pt,
)
#set par(justify: false, leading: 0.55em)

= Scaling Book Exercises -- Chapter 4

Chapter: 4 -- All the Transformer Math You Need to Know\
Source: handwritten Onyx Boox A4 PDF\
Note: Transcribed from handwritten solutions; book markdown used only for exercise statements and notation.

== Exercise 1 -- Parameter count, attention fraction, and KV cache

=== Exercise statement

How many parameters does a model with $D = 4096$, $F = 4 dot D$, $V = 32,000$, and $L = 64$ have? What fraction of these are attention parameters? How large are our KV caches per token? You can assume $N dot H = D$ and multi-head attention with int8 KVs.

=== Solution

Scan pages: 1--2

The total parameters of a model are:
$
  (3 D F + 4 D N H + D) L + D V.
$

If we assume $F = 4D$, $N H = D$, we get:
$
  (16 D^2 + D) L + D V.
$

Let $D = 4096$, $V = 32000$, $L = 64$. Hence:
$
  (16 dot (4096)^2 + 4096) dot 64 + 4096 dot 32000
    approx 17.3 dot 10^9
    = 17.3 "Billion parameters".
$

Attention params are $4 D^2 L$ hence:
$
  4 dot (4096)^2 dot 64 approx 4.3 "Billion",
$
so the fraction is:
$
  frac(4.3, 17.3) approx 0.24.
$

The size of the KV cache is $2 dot S dot L dot N dot H$ and since $N H = D$, we get $2 dot L dot D$ per token i.e.
$
  2 dot 64 dot 4096 approx 524 "KB".
$

== Exercise 2 -- FLOPs under sharding

=== Exercise statement

How many total FLOPs are required to perform $A[B_X, D_Y] dot_D W[D_Y, F]$ on `{'X': 4, 'Y': 8, 'Z': 4}`. How many FLOPs are performed by each TPU?

=== Solution

Scan pages: 2

Let $"Mesh" = {X:4, Y:8, Z:4}$.

Assume the intended operation is $A[B_X, D_Y] dot_D W[D_Y, F]$. Each device $(i, j, k)$ performs
$
  2 dot frac(B, |X|) dot frac(D, |Y|) dot F
$
FLOPs. Hence total FLOPs is:
$
  "Total FLOPs"
    = |X| dot |Y| dot |Z| dot 2 dot frac(B, |X|) dot frac(D, |Y|) dot F
    = 8 B D F.
$

== Exercise 3 -- FLOPs for a tensor contraction

=== Exercise statement

How many FLOPs are involved in performing $A[I,J,K,L] dot B[I,J,M,N,O] -> C[K,L,M,N,O]$?

=== Solution

Scan pages: 3

Assume the intended operation is:
$
  A[I,J,K,L] dot B[I,J,M,N,O] -> C[K,L,M,N,O]
$
by means of:
$
  C[K,L,M,N,O]
    = sum_(j=1)^J sum_(i=1)^I A[i,j,K,L] dot B[i,j,M,N,O].
$
we do $K dot L dot M dot N dot O dot 2 dot I dot J$ FLOPs.

== Exercise 4 -- Self-attention arithmetic intensity and effective cost

=== Exercise statement

What is the arithmetic intensity of self-attention, ignoring the Q/K/V/O projections? Give the answer as a function of the Q and KV lengths $T$ and $S$. At what context length is attention FLOPs-bound? Given the HBM bandwidth of our TPUs, plot the effective relative cost of attention to the FFW block as the context length grows.

=== Solution

Scan pages: 4--8

Assume our naive attention algorithm writes every intermediate result back to HBM and also that $K = N$. Hence we do:

1. $L[B,T,S,N] <- Q[B,T,N,H] dot K[B,S,N,H]$.
2. $S[B,T,S,N] <- "SoftMax"_(S)(L[B,T,S,N])$.
3. $O[B,T,N,H] <- S[B,T,S,N] dot V[B,S,N,H]$.

Assume that $N H = D$.

Step 1:
$
  "Bytes Step 1" = 2 B T N H + 2 B S N H + 2 B T S N
    = 2 B T D + 2 B S D + 2 B T S N, \
  "FLOPs Step 1" = 2 B N T S H = 2 B T S D.
$

Step 2:
$
  "Bytes Step 2" = 2 B T S N + 2 B T S N = 4 B T S N, \
  "FLOPs Step 2" = O(B T S N).
$

Step 3:
$
  "Bytes Step 3" = 2 B T S N + 2 B S N H + 2 B T N H
    = 2 B T S N + 2 B S D + 2 B T D, \
  "FLOPs Step 3" = 2 B N T H S = 2 B T S D.
$

Assume Flash Attention, so $L[B,T,S,N]$ does not need to be in HBM.

$
  "Bytes Total" &= 4 (B S D + B T D) \
  "FLOPs Total" &= 2 B T S D + 2 B T S D = 4 B T S D \
  "AI"(T,S)
    &= frac(B T S D, B S D + B T D)
    = frac(T S D, S D + T D)
    = frac(T S, T + S).
$

Let $S = T$:
$
  "AI" = frac(T^2, 2 T) = frac(T, 2).
$

If $T >= 2 dot "AI"("accelerator")$, we are FLOPs bound. If we assume v5e, then $"AI"("v5e") approx 240$, hence $T >= 480$.

If by effective relative cost they mean:
$
  "Effective cost"(T) = frac(t_("attention")(T), t_("FFW")(T)),
$
where $T$ is context length, not time.

Assume $B > 240$ so that in the FFW block we become compute bound, hence:
$
  t_("FFW") = frac(6 B T D F, "FLOPs/s") \
  t_("attention") = max(frac(8 B T D, "BW"), frac(4 B T^2 D, "FLOPs/s"))
$
Therefore:
$
  "Effective cost"(T)
    = frac(max(frac(8 B T D, "BW"), frac(4 B T^2 D, "FLOPs/s")), frac(6 B T D F, "FLOPs/s"))
    = max(frac(8 B T D, "BW"), frac(4 B T^2 D, "FLOPs/s")) dot frac("FLOPs/s", 6 B T D F).
$

If $T >= 480$, then:
$
  "Effective cost"(T)
    = frac(4 B T^2 D, 6 B T D F)
    = frac(4 T, 6 F)
    = frac(4 T, 24 D)
    = frac(T, 6 D).
$

If $T < 480$, then:
$
  "Effective cost"(T)
    = frac(8 B T D, "BW") dot frac("FLOPs/s", 6 B T D F)
    = frac(8 B T D, 6 B T D F) dot frac(1, 240)
    = frac(8, 24 D) dot frac(1, 240)
    = frac(8, 24 dot 240 dot D)
    = frac(8, 5760 dot D)
    approx frac(13, 10^(-4)) dot frac(1, D).
$

Assume $D approx 8k$:

#image("effective-cost-plot.png", width: 80%)


== Exercise 5 -- Attention FLOPs vs. QKVO projection FLOPs

=== Exercise statement

At what sequence length are self-attention FLOPs equal to the QKVO projection FLOPs?

=== Solution

Scan pages: 9

Assuming FLOPs without training, $D = N H$, $S = T$, and $N = K$. Then we have to solve for:
$
  4 B T^2 D = 8 B T D N H
  => 4 B T^2 D = 8 B T D^2
  => T = frac(8, 4) D => T = 2D.
$

So at $T = 2D$ sequence length, self-attention FLOPs equal the QKVO projection FLOPs.

== Exercise 6 -- Rematerialization FLOPs

=== Exercise statement

Say we only save the output of each of the 7 main matmuls in a Transformer layer during our forward pass, namely Q, K, V, O + the three FFW matrices. How many extra FLOPs do we need to "rematerialize" during the backwards pass?

=== Solution

Scan pages: 9--12

Define a general layer as:
$
  O = f(I, W).
$

In a computational graph, during the backward pass, it will receive $frac(d L, d O)$, and it will compute $frac(d L, d I)$, $frac(d L, d W)$ and send $frac(d L, d I)$ down the graph. Note that in general:
$
  frac(d L, d W) = frac(d L, d O) dot frac(d O, d W)(W, I)
$
so it depends on $I$ as well. With this in mind, we can see that if we draw the computational graph we will need to rematerialize:

- $I = "gelu"(X  W_("in1")) ⊙ (X  W_("in2"))$ for $frac(d L, d W_("out"))$.
- $A = "softmax"(Q K^T)  V$ for $frac(d L, d W_o)$.
- $"softmax"(Q K^T)$ for $frac(d L, d V)$.

where we have ignored the layernorm operations. Therefore we need:
$
  "FLOPs" = 4 B T S N H + O(B T F)
$
to rematerialize during the backward pass.
#image("computational-graph.png", width: 30%)

== Exercise 7 -- DeepSeek V3 utilization

=== Exercise statement

DeepSeek v3 says it was trained for 2.79M H800 hours on 14.8T tokens. Given that it has 37B activated parameters, roughly what hardware utilization did they achieve? Hint: note that they used FP8 FLOPs without structured sparsity.

=== Solution

Scan pages: 12

FLOPs for V3 = $6 dot 37 e 9 dot 14.8 e 12$.

$"H800 SXM FP8 tensor core FLOPs/s" = 1474 "TFLOPs/s"$.

Available FLOPs in $2.79 "M"$ H800 hours, if using FP8 = $ 2.79 e 6 dot 3600 dot 1474 e 12$.

Hardware Utilization:
$
  frac(6 dot 37 dot 14.8 dot 10^21, 2.79 dot 3600 dot 1474 dot 10^18)
    = frac(6 dot 37 dot 14.8, 2.79 dot 3600 dot 1474) dot 10^3
    approx 0.00016 dot 10^3
    = 0.16.
$

== Exercise 8 -- MoE compute-bound batch size

=== Exercise statement

Mixture of Experts (MoE) models have $E$ copies of a standard dense MLP block, and each token activates $k$ of these experts. What batch size in tokens is required to be compute-bound for an MoE with weights in int8 on TPU v5e? For DeepSeek, which has 256 routed experts and $k=8$, what is this number?

=== Solution

Scan pages: 13--15

Assume training FLOPs. $E$ copies of standard MLP blocks, and each token activates $k$ of these experts. Consider the following model of MoE:
$
  h_t = sum_(i=1)^E g_(i,t) "FFN"_(i)(x_t),
$
where $"FFN"_(i)(x_t) = W_i x_t$ and $W_i in bb(R)^(d times d)$.

To compute the MoE layer, we will group the tokens belonging to an expert and then matmul. So in total we will do $E$ matmuls. Assume that every token randomly chooses $k$ numbers from ${1, ..., E}$.

Define:
$
  S_e = sum_(i=1)^T X_(i,e),
$
where $X_(i,e) = 1$ if token $i$ has sampled $e in {1, ..., E}$.

The expected value of $S_e$ gives us the expected value of the batch dimension of the matmul of expert $e$:
$
  E[S_e]
    = sum_(i=1)^T P(X_(i,e))
    = sum_(i=1)^T frac(k, E)
    = frac(T dot k, E).
$

So then we do:

```text
For e in {1,...,E}
  1: [T*k/E, D] @ int8[D, D]
```
Therefore:
$
  "FLOPs" = E dot (2 dot frac(T dot k, E) dot D^2)
    = 2 T k D^2 \
  "Bytes" = E dot (frac(2 T k, E) dot D + D^2)
    = 2 T k D + E D^2.
$

$
  "AI"
    = frac(2 dot T dot k dot D^2, 2 dot T dot k dot D + E dot D^2)
    approx frac(2 dot T dot k dot D^2, E dot D^2)
    = frac(2 T k, E).
$

Therefore for being compute bound:
$
  frac(2 T k, E) > 240 \
  => T > frac(E, k) dot frac(240, 2) := frac(E, k) dot 120.
$

So for DeepSeek V3 this number is $3840$.
(Note: While transcribing this execirse from the handwritten notes, I realized
that I made an error and assumed that the operations were done in int8, but
they are done in FP16, so I've corrected the of by 2 error in this transccribed
notes, in the handwritten notes is not corrected).
