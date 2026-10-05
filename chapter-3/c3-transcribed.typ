#import "../template/layout.typ": with-layout
#show: with-layout

= Scaling Book Exercises -- Chapter 3

Transcribed from handwritten solutions.
#link("https://github.com/kevinlopezandrade/scaling-book-solutions/blob/main/chapter-3/c3-handwritten.pdf")[Source]

== Exercise 1 -- replicated sharding

=== Exercise statement

An array is sharded $A[I_X, J, K, ...]$ (i.e., only sharded across $X$), with a mesh `Mesh({'X': 4, 'Y': 8, 'Z': 2})`. What is the ratio of the total number of bytes taken up by $A$ across all chips to the size of one copy of the array?

=== Solution

Scan pages: 1-2.

Let $A[I_X, J, K, ...]$ and $"Mesh" = {X:4, Y:8, Z:2}$.

At each device tuple $(i, j, k)$, size of array shard is:
$
  "Size of shard" = frac(abs(I), X) dot P dot "BytesForPrecision" "where" P = product_(d in "dimensions" - {I}) abs(d)
$


Assuming "one copy of array" means unsharded array, ratio is then:

$
  frac(X dot Y dot Z dot abs(I) dot X^(-1) dot P dot "BytesForPrecision",
       abs(I) dot P dot "BytesForPrecision") = 16
$

If copy means one shard, then ratio is:

$
  frac(X dot Y dot Z dot abs(I) dot X^(-1) dot P dot "BytesForPrecision",
       abs(I) dot X^(-1) dot P dot "BytesForPrecision") = 64
$

== Exercise 2 -- AllGather latency

=== Exercise statement

How long should $"AllGather"_(X)([B_X, D_Y])$ take on a TPU v4p 4x4x4 slice with mesh `Mesh({'X': 4, 'Y': 4, 'Z': 4})` if $B=1024$ and $D=4096$ in bfloat16? How about $"AllGather"_(X Y)([B_X, D_Y])$? How about $"AllReduce"_(Z)([B_X, D_Y] {U_Z})$?

=== Solution

Scan pages: 2-5.

$"v4p":$

- HBM capacity = $32 "GB"$
- HBM BW = $1.2 e 12 "Bytes/s"$
- FLOPs/s (bf16) = $2.75 e 14$
- ICI Bidi = $9.0 e 10$

Slice of $4 times 4 times 4$ $"v4p"$. $"Mesh" = {X: 4, Y: 4, Z: 4}$. Let $B = 1024$ and $D = 4096$ in bfloat16.
Assuming $A[B_X, D_Y]$, each device tuple in the mesh maps to:

$
(i,j,k) |-> A[
  i dot abs(B) / X : (i+1) dot abs(B) / X,
  j dot abs(D) / Y : (j+1) dot abs(D) / Y
]
$

Therefore size per shard is:
$
  frac(abs(B), X) dot frac(abs(D), Y) dot 2 = frac(B^2, 2) "bytes"
$

since $D = 4 dot B$.

Since v4p slice $4 times 4 times 4$ has wrap-around links, we get that:
$
  T("AllGather"_(X)([B_X, D_Y]))
    = frac(X, 2) dot frac(B^2, 2 dot W_("uni"))
    = frac(B^2, 4.5 e 10)
    approx 2 dot 10^(-5) "s"
    = 0.02 "ms."
$

For allgather on 2 axis we don't have an specific algorithm but we can derive a lower bound:
$
  T("AllGather"_(X Y)([B_X, D_Y]))
    >= frac(2 dot B dot D, 2 dot W_("ICI"))
    = frac(2 B dot 4 B, 2 W_("ICI"))
    = frac(4 B^2, 9 e 10)
    approx 0.04 "ms."
$

$T("AllReduce"_(Z)([B_X, D_Y] {U_Z})) = frac(Z, 2) dot (frac(B^2, 2 dot Z dot W_("uni"))) + frac(Z, 2) dot (frac(B^2, 2 dot Z dot W_("uni"))) = frac(B^2, 4 dot W_("uni")) + frac(B^2, 4 dot W_("uni"))
  = frac(2 B^2, 4 W_("uni"))
  = frac(B^2, 2 W_("uni"))
  = frac(B^2, W_("ICI"))
  = frac(1024^2, 9 e 10)
  = 1 dot 10^(-4)
  = 0.01 "ms."
$

== Exercise 3 -- latency-bound AllGather

=== Exercise statement

Let's say we're performing an $"AllGather"_X([B_X])$ but $B$ is very small, say 128. How long should this take on a TPU v4p 4x4x4 slice with mesh `Mesh({'X': 4, 'Y': 4, 'Z': 4})` in bfloat16? Hint: you're probably latency bound.

=== Solution

Scan pages: 6-7.

TPU v4p.

Let $"Mesh" = {X: 4, Y: 4, Z: 4}$
Let $B = 128 "and" A = "bfloat16"[B_X]$

Assuming communication model $T_("step") = alpha + frac(D, "BW")$
where $alpha = 1 mu "s"$, and wraparound links,
$T("AllGather"_(X)([B_X]))
  = frac(X, 2) dot (alpha + frac(B dot 2, X dot W_("uni")))
  = 2 dot alpha + frac(2 dot B, W_("ICI"))
  = 2 mu"s" + frac(256, 9 e 10)
  approx 2 mu"s" + 28.5 dot 10^(-10)
  = 2 mu"s" + 28.5 dot 10^(-3) dot 10^(-6)
  = 2 mu"s" + 0.0285 mu"s"
$

So yeah I'm bound by latency.

== Exercise 4 -- matmul strategies

=== Exercise statement

To perform $X[B, D] dot_D Y[D_X, F] -> Z[B, F]$, in this section we tell you to
perform $"AllGather"_(X)(Y[D_X, F])$ and multiply the fully replicated matrices
(Case 2, Strategy 1). Instead, you could multiply the local shards like $X[B,
D_X] dot_D Y[D_X, F] -> Z[B, F] {U_X}$ (Case 3, Strategy 2), and then
$"AllReduce"_(X)(Z[B, F] {U_X})$. How many FLOPs and comms does each of these
perform? Which is better and why?

=== Solution

Scan pages: 7-16.

$ X[B,D] dot_D Y[D_X,F] -> Z[B,F] $

_Strategy 1:_

1. $"AllGather"_(X)(Y[D_X,F]) -> Y[D,F]$
2. $X[B,D] dot_D Y[D,F] -> Z[B,F]$

Note that $2 dot X =$ number of directed links. Each link participates in $X / 2$ steps of the bidirectional ring algo and each payload is
$frac(D dot F dot 2, X)$, assuming bfloat16:
$
  "AllGather"[D_X,F] "Comms"
    = 2 dot (frac(X,2) dot frac(D dot F dot 2, X))
    = 2 D F
$

Comms for MatMul $= 2 B D + 2 D F + 2 B F$
$=> "Total comms strategy 1" = 2D F + 2 B D + 2 D F + 2 B F = 4 D F + 2 B D + 2 B F.$

$"FLOPs for strategy 1" = 2 dot B dot D dot F$

_Strategy 2:_

1. $X[B,D_X] dot_D Y[D_X,F] -> Z[B,F] {U_X}$
2. $"AllReduce"_(X)(Z[B,F] {U_X})$

$"Comms MatMul" = 2 dot B dot frac(D, X) + 2 dot frac(D, X) dot F + 2 dot B dot F.$

$"Comms" "AllReduce"_(X)(Z[B,F] {U_X}) = 2 dot (2 (frac(X,2) dot frac(B dot F dot 2, X))) = 4 B F.$

$"Total comms strategy 2" = 4 B F + 2 dot B dot frac(D, X) + 2 dot frac(D, X) dot F + 2 B F = 6 B F + 2 dot B dot frac(D, X) + 2 dot frac(D, X) dot F.$

$"FLOPs strategy 2" = 2 dot B dot frac(D, X) dot F.$

From the book they say you can overlap collective communications with the matrix multiplication itself. So then we can work only with the communications of the collectives to see which strategy is better.

So then assume:
$
  T_("strategy1") &= max(frac(2 D F, W_("ICI")), frac(2 B D F, "FLOPs/s")) \ 
  T_("strategy2") &= max(frac(4 B F, W_("ICI")), frac(2 B D F, X dot "FLOPs/s")) \
  A I_1 &= frac(2 B D F, 2 D F) = B quad "FLOPs/Byte" \ 
  A I_2 &= frac(2 B D F dot X^(-1), 4 B F) = frac(D, 2X) quad "FLOPs/Byte"
$

Let's fix $X in {4, 8, 16}$ valid in v4p slice, and they will still have a wraparound link.
$A I_1(B),quad A I_2(D; X) "for fixed values of" X.$

Achievable FLOPs:

$
  "Achievable FLOPs"(B) &= min(B dot 9 e 10, 2.75 e 14) \
  "Achievable FLOPs"(D; X) &= min(frac(D, 2 X) dot 9 e 10, 2.75 e 14)
$

#image("plot.png")

Algo 1 branches by $B > 3055$.Algo 2 branches by $D > 24 K$.
So in total 4 combinations:

$B > 3055$ and $D > 24 K$ $=>$ Both algos are compute bound.
$
  &T_("math")("Strat 1") < T_("math")("Strat 2") \
  &<=> 2 B D F < frac(2 B D F, X) quad "for" X = 4 \
  &<=> 2 < frac(1, 2) quad "Contradiction."
$
Therefore $T_("math")("Strat 2") >= T_("math")("Strat 1")$ and in fact it holds for every $X > 1$. Just the condition for when Algo 2 is compute bound changes, but assuming both compute bound strategy 2 is better.

$B > 3055$ and $D < 24 K$ $=>$ Algo 1 compute bound and Algo 2 communication bound.
$
  &T_("math")("Strat 1") < T_("comms")("Strat 2") \
  &<=> frac(2 B D F, "FLOPs/s") < frac(4 B F, W_("ICI")) \
  &<=> frac(2D, 4) < frac("FLOPs", W_("ICI")) = 3055 \
  &<=> frac(D, 2) < 3055 => D < 6110
$
So strategy 1 in this case wins for $D < 6110$ otherwise strategy 2 is better.

$B < 3055$ and $D > 24 K$ $=>$ Algo 1 communication bound and Algo 2 compute bound.
$
  &T_("comms")("Strat 1") < T_("math")("Strat 2") \
  &<=> frac(2 D F, W_("ICI")) < frac(2 B D F, X dot "FLOPs/s")) quad "for" X = 4 \
  &<=> frac(2, W_("ICI")) < frac(2 B, 4 dot "FLOPs/s") \
  &<=> B > 4 dot frac("FLOPs/s", W_("ICI")) = 4 dot 3055 approx 12 K
$
which is a contradiction, so then strategy 2 in this setting is always better, and in fact it will hold for all $X > 2$. It just changes the condition when algo 2 is compute bound.

$B <= 3055$ and $D <= 24 K$ $=>$ Both algos are communication bound.
$
  &T_("comms")("Strat 1") < T_("comms")("Strat 2") \
  &<=> 2 D F < 4 B F \
  &<=> 2D < 4B => D < 2 dot B
$
So if $D < 2 dot B$, strategy 1 wins, otherwise strategy 2 wins. In the book they often assume $D >> B$, so then $D > 2 dot B$ and strategy 2 is better.

== Exercise 5 -- minimum latency

=== Exercise statement

Let's say I want to do a matmul $A[I, J] dot_J B[J, K] -> C[I, K]$ on a TPU v4p 4x4x4 with the lowest possible latency. Assume the inputs can be sharded arbitrarily but the result should be fully replicated. How should my inputs be sharded? What is the total FLOPs and comms time?

=== Solution

Scan pages: 17-25.

Assume we have a $4 times 4 times 4$ slice of v4p devices with wraparound links. Assume the intended operation is:

$
  A[I, J] dot_J B[J, K] -> C[I, K]
$

with bfloat16 precision.

Note that there is only one operation where we do FLOPs, the matrix multiplication (ignoring AllReduce). This allows to divide the space of possible sharding strategies in three groups:

1. Shardings resulting in collective communications before doing matmul.
2. Shardings resulting in collective communications only after doing matmul.
3. Shardings resulting in collective communications before and after matmul.

In all groups, note that if we are going to do a collective operation, we should choose shardings for which the lower bound in $T_("comms")$ is the smallest, since an optimal algo can get closer to this lower bound. Note that for either $O in {A, B, C}$ we have:
$
  T_("comms") >= frac("size"(O), N_("axes") dot W_("ICI")).
$
So the smallest lower bound is achieved using shardings that use all the axis of our $4 times 4 times 4$ v4p slice. This argument wipes out our space of possible shardings by a lot.

_For group 1:_

1.1 $A[I, J_(X Y Z)] dot B[J, K] -> C[I, K]$

1.2 $A[I, J] dot B[J_(X Y Z), K] -> C[I, K]$

_For group 2:_

2.1 $A[I_(X Y Z), J] dot B[J, K] -> C[I, K]$

2.2 $A[I, J] dot B[J, K_(X Y Z)] -> C[I, K]$

2.3 $A[I, J_(X Y Z)] dot B[J_(X Y Z), K] -> C[I, K]$

_For group 3:_

3.1 $A[I_(X Y Z), J] dot B[J_(X Y Z), K] -> C[I, K]$

3.2 $A[I, J_(X Y Z)] dot B[J, K_(X Y Z)] -> C[I, K]$

3.3 $A[I_(X Y Z), J] dot J[J, K_(X Y Z)] -> C[I, K]$

Note that in group 3, depending in which axis you allgather first, you end back to a sharding that belongs to group 1 or 2.

Assume $J >> I$ and $K approx 4 dot J$.

_Then for group 1:_

1.1 $T_("comms") = 2 dot I dot J, T_("math") = 2 dot I dot J dot K$.
If $frac(2 I J K, 2 I J) = K > 3055 => "compute bound"$

1.2 $T_("comms") = 2 dot J dot K, T_("math") = 2 dot I dot J dot K$.
If $I > 3055 => "compute bound"$.

_Then for group 2:_

2.1 $T_("comms") = 2 dot I dot K, T_("math") = frac(2 I J K, X dot Y dot Z)$.
If $frac(J, X dot Y dot Z) > 3055 => "compute bound"$ i.e $J > 64 dot 3055 = 196 K$ not realistic as $B[J,K]$ will not fit in HBM. 

2.2 $T_("comms") = 2 dot I dot K, T_("math") = frac(2 I J K, X dot Y dot Z)$. If $frac(J, X dot Y dot Z) > 3055 => "compute bound"$ i.e. $J > 196K$. So again not realistic.

2.3 $T_("comms") = 4 dot I dot K, T_("math") = frac(2 I J K, X dot Y dot Z)$. If $frac(J, 2 X dot Y dot Z) > 3055 => "compute bound"$ i.e $J > 391K$. Not realistic.

Hence:
$
  T("Algo 1.1") &= frac(2 I J K, "FLOPs/s") approx frac(8 J^2, "FLOPs/s") \
  T("Algo 1.2") &= frac(2 I J K, "FLOPs/s") approx frac(8 J^2, "FLOPs/s") \
  &"or"\
  &= frac(2 J K, W_("ICI")) = frac(8 J^2, W_("ICI")) \
  T("Algo 2.2") &= frac(2 I K, W_("ICI")) approx frac(8 J, W_("ICI"))
$

So then Algo 2.2 and 2.1 are better than Algo 1.2 and Algo 1.1 when all are communication bound. And if Algo 1.2 and Algo 1.1 are compute bound, Algo 2.1 and Algo 2.2 are better when:

$
  frac(8 J, W_("ICI")) < frac(8 J^2, "FLOPs/s") => J > 3055
$

In that case, between Algo 2.1 and Algo 2.2, Algo 2.2 is better since $K >> I$ and:

$
  "comms Algo 2.1" = frac(2 I J, X Y Z) + 2 J K + 2 I K
$

is dominated by $2 J K$.

$
  "comms Algo 2.2" = 2 I K + frac(2 J K, X Y Z) + 2 I K
$

is dominated by $2 J K / (X Y Z)$. So if both memory bound, Algo 2.2 is better.
And we enter compute bound with smaller values of $I$ with Algo 2.2, since if
$I = A I("Algo 2.2") > 230$ we are compute bound and for Algo 2.1 we have:
$
  A I = frac(2 I J K, X Y Z) dot frac(1, 2 J K) = frac(2, 64) dot I > 230 => I > 7360
$

== Exercise 6 -- sharded matmul cases on TPU v5e 4x4

=== Exercise statement

Let's say we want to perform $A[I_X, J_Y] dot_J B[J_Y, K] -> C[I_X, K]$ on TPU v5e 4x4. What communication do we perform? How much time is spent on communication vs. computation?

What about $A[I_X, J] dot_J B[J_X, K_Y] -> C[I_X, K_Y]$? This is the most standard setting for training where we combine data, tensor, and ZeRO sharding.

What about $A[I_X, J] dot_J B[J, K_Y] -> C[I_X, K_Y]$? This is standard for inference, where we do pure tensor parallelism plus data.

=== Solution

Scan pages: 25-28.

Assume we have a $4 times 4$ slice of TPU v5e. Since $4 times 4$ is not a full pod in this setting we don't have wraparound links.

Let:

- v5e ICI BW unidirectional = $4.5 e 10$
- v5e FLOPs/s (bf16) = $1.97 e 14$

Assuming BFLOAT16.

For $A[I_X, J_Y] dot_J B[J_Y, K] -> C[I_X, K]$:

1. $O[I_X, K] {U_Y} = A[I_X, J_Y] dot J_("local")[J_Y, K]$
2. $C[I_X, K] = "AllReduce"_(Y)(O[I_X, K] {U_Y})$

We cannot use a bidirectional ring algorithm for the ReduceScatter + AllGather, since no wraparound links in a $4 times 4$ slice.
Hence:
$
  "Communication Load" = 2 dot (abs(Y) - 1) dot frac(I dot K dot 2, abs(Y) abs(X))
  approx frac(2 dot I dot K dot 2, abs(X))
  => T_("comms") = frac(4 I K, abs(X) dot "BW") \
  T_("math") = frac(2 I J K, abs(X) abs(Y) dot "FLOPs/s").
$
Therefore:
$
  frac(T_("comms"), T_("math"))
    = frac(4 I K, abs(X) dot "BW") dot frac(abs(X) abs(Y) dot "FLOPs/s", 2 I J K)
  = frac(4 abs(Y), J) dot frac("FLOPs/s", "BW")
  = frac(16, J) dot 4380 approx frac(70080, J).
$

For $A[I_X, J] dot_J B[J_X, K_Y] -> C[I_X, K_Y]$:

1. $B[J, K_Y] = "AllGather"_(X)(B[J_X, K_Y])$
2. $C[I_X, K_Y] = A[I_X, J] dot B[J, K_Y]$

$
  "Communication load" = (abs(X) - 1) dot (frac(2 J dot K, abs(X) abs(Y)))
  approx 2 dot J dot K dot abs(Y)^(-1)
  => T_("comms") = frac(2 J K, abs(Y) dot "BW") \
  T_("math") = frac(2 I J K, abs(X) abs(Y) dot "FLOPs/s").
$
Therefore:
$
  frac(T_("comms"), T_("math"))
  = frac(2 J K, abs(Y)) dot frac(abs(X) abs(Y) dot "FLOPs/s", 2 I J K dot "BW")
  = frac(abs(X), I) dot 4380 = frac(17520, I).
$

For $A[I_X, J] dot_J B[J, K_Y] -> C[I_X, K_Y]$:

No communication needs to be done.
$
  T_("math") = frac(2 I J K, abs(X) abs(Y) dot "FLOPs/s")
$
All the time is in computation.

== Exercise 7 -- Transformer block sharding with memory limit

=== Exercise statement

A typical Transformer block has two matrices $W_("in")[D, F]$ and $W_("out")[F, D]$ where $F >> D$. Say we have a batch size B. Then the full block is $"In"[B, D] dot W_("in")[D, F] dot W_("out")[F, D]$. Let's pick $D=8192$, $F=32768$, and $B=128$ and assume everything is in bfloat16. Assume we're running on a TPU v5e 2x2 slice but let's pretend each TPU only has 300MB of free memory. How should In, $W_("in")$, $W_("out")$, and Out be sharded to stay below the memory limit while minimizing overall time? How much time is spent on comms and FLOPs? Hint: the final output doesn't need to be fully replicated, but it should be sharded the same as the input so the layer can be repeated.

=== Solution

Scan pages: 29-32.

Assume bfloat16. Let $D = 8192$, $F = 32768$, $B = 128$. The intended operation is:

$
  "In"[B, D] dot W_("in")[D, F] dot W_("out")[F, D]
$

$"Size"(W_("in")) = 8192^2 dot 4 dot 2 approx 536 "MB"$.

$"Size"(W_("out")) = "Size"(W_("in"))$.

$"Size"("In") approx 2 "MB"$.

$"Size"("Aux"[B,F]) approx 8 "MB"$.

Since we only have left 300MB, then we have to shard $W_("in")$ and $W_("out")$
so that both sharded $W_("in")$ and $W_("out")$ can be found in each device. Sharding just
one dimension along one axis does not work since then both $W_("in")$ and $W_("out")$ do not
fit in a single device:

$
  frac("Size"(W_("in")), 2) + frac("Size"(W_("out")), 2) > 300 "MB"
$

Therefore we have to either shard each dimension with a single axis i.e:

$
  W_("in")[D_X, F_Y] forall in {X, Y}
$

where $X,Y$ are the axis of the device mesh, or shard one dimension along both axis of the mesh i.e:

$
  W_("in")[D, F_(X Y)] " or " W_("in")[D_(X Y), F]
$

In both cases we shrink both $"Size"(W_("in"))$ and $"Size"(W_("out"))$ by 4 and:

$
  frac("Size"(W_("in")), 4) + frac("Size"(W_("out")), 4) <= 300 "MB"
$

The way to shard to minimize overall time is:

$
  "In"[B, D] @ W_("in")[D, F_(X Y)] @ W_("out")[F_(X Y), D]
$

where we construct a 1-D ring in $2 times 2$ slice of v5e TPUs.

For our chosen sharding we only do one collective communication at the end, specifically an AllReduce = ReduceScatter + AllGather. Every other sharding strategy will require:

$
  T_("comms")("Alternative sharding") >= frac(4 B D, W_("ICI"))
$

You could also do:

$
  "In"[B, D_(X Y)] dot W_("in")[D, F_(X Y)] dot W_("out")[F_(X Y), D]
$

with an AllGather at the beginning and a ReduceScatter at the end and get same:

$
  T_("comms") = frac(2 B D, W_("ICI")) + frac(2 B D, W_("ICI")) = frac(4 B D, W_("ICI"))
$

So for our chosen sharding strategy:

1. $"Aux"[B, F_(X Y)] = "In"[B,D] dot W_("in")[D, F_(X Y)]$
2. $O[B, D] {U_(X Y)} = "Aux"[B, F_(X Y)] dot W_("out")[F_(X Y), D]$
3. $O[B, D] = "AllReduce"_(X Y)(O[B, D] {U_(X Y)})$

Therefore:

$
  T_("math") &= frac(4 B D F, abs(X) abs(Y) dot 1.97 e 14)
  = frac(128 dot 8192 dot 32768, 1.97 e 14)
  approx 0.0002 "s" = 0.2 "ms" \
  T_("comms") &= frac(4 B D, 9.0 e 10)
  approx 0.00004 "s"
  = 4 dot 10^(-5)
  = 4 dot 10^(-2) dot 10^(-3) "s"
  = 0.04 "ms"
$

== Exercise 9 -- another strategy for sharded matmuls?

=== Exercise statement

Above, we claimed that when only one input to a matmul is sharded along its contracting dimension, we should AllGather the sharded matrix and perform the resulting contraction locally. Another strategy you might think of is to perform the sharded matmul and then AllReduce the result, as if both inputs were sharded along the contracting dimension, i.e. $A[I, J_X] *_J B[J, K] -> C[I, K]$ by way of:

1. $C[I, K] {U_X} = A[I, J_X] dot B[J_X, K]$
2. $C[I, K] = "AllReduce"(C[I, K] {U_X})$

Answer the following:

1. Explicitly write out this algorithm for matrices $A[N, M]$ and $B[M, K]$, using indices to show exactly what computation is done on what device. Assume $A$ is sharded as $A[I, J_X]$ across ND devices, and you want your output to be replicated across all devices.
2. Now suppose you are ok with the final result not being replicated on each device, but instead sharded, across either the N or K dimension. How would the algorithm above change?
3. Looking purely at the communication cost of the strategy above, in part 2 not 1, how does this communication cost compare to the communication cost of the algorithm in which we first AllGather A and then do the matmul?

=== Solution

Scan pages: 33-34.

_9.1_

Let $A[N,M]$ be sharded as $A[I, J_X]$ where $abs(X) = D$. Let $B[M,K]$ be sharded as $B[J_X, K]$. Assume $M / D in NN$.

Define:
$
  A_i := A[:, i dot frac(M, D) : (i+1) dot frac(M, D)]
$
for $0 <= i <= D - 1$, where we use Python slicing notation and semantics.

Define:
$
  B_i := B[i dot frac(M, D) : (i+1) dot frac(M, D), :]
$

for $0 <= i <= D - 1$.

The computation at each device $i$ is then $A_i dot B_i in RR^(N times K)$.
After that we do $C[I, K] = sum_(i=0)^(D-1) A_i dot B_i$ with an AllReduce.

_9.2_

1. $C[I, K] {U_X} = A[I, J_X] dot B[J_X, K]$
2. $"ReduceScatter"_(X, I "or" K)(C[I, K] {U_X})$

_9.3_

Assume bfloat16.

$"Communication Load Part 2" = 2 dot N dot K$.

$"Communication load AllGather first" = 2 dot M dot K$.

$"Ratio" = frac(2 dot N dot K, 2 dot M dot K) = frac(N, M)$.

== Exercise 10 -- Fun with AllToAll

=== Exercise statement

In the table above, it was noted that the time to perform an AllToAll is a factor of 4 lower than the time to perform an AllGather or ReduceScatter, in the regime where we are throughput-bound. In this problem we will see where that factor of 4 comes from, and also see how this factor would change if we only had single-direction ICI links, rather than bidirectional ICI links.

1. Let's start with the single-direction case first. Imagine we have D devices in a ring topology and want to do either an AllGather or a ReduceScatter on an N x N matrix $A[I_X, J]$, say $D$ divides $N$ for simplicity. Describe the comms involved in these two collectives, and calculate the total number of scalars, floats or ints, which are transferred across a single ICI link during the entirety of this algorithm.
2. Now let's think about an AllToAll, still in the single-directional ICI case. How is the algorithm different in this case than the all-gather case? Calculate the number of scalars that are transferred across a single ICI link in this algorithm.
3. You should have found that the ratio between your answers to part (a) and part (b) is a nice number. Explain where this factor comes from in simple terms.
4. Now let's add bidirectional communication. How does this affect the total time needed in the all-gather case?
5. How does adding bidirectional communication affect the total time needed in the AllToAll case?
6. Now simply explain the ratio between AllGather time and AllToAll time in a bidirectional ring.

=== Solution

Scan pages: 35-40.

_10.1_

Let $A[N,N]$ be sharded as $A[I_X, J]$ where $abs(X)=D$ and $N / D in NN$.

With only one direction, the ring algo requires $D - 1$ steps to finish. Define
a link $i$ as a directed edge $(i, i+1 mod D)$ for $0 <= i <= D - 1$. Note that
at one step of the algo link $i$ transfers $N^2 / D$ scalars. Since there are
$D - 1$ steps, the scalars being transferred by a single link during the
entirety of the algorithm is:

$
  (D - 1) dot frac(N^2, D)
$

for both AllGather and ReduceScatter.

_10.2_

At step $k$ of the AllToAll algorithm, where $1 <= k <= D - 1$, the directed link $(i, i+1 mod D)$ transfers:
$
  frac(N^2, D) - k dot frac(N^2, D^2)
$

Therefore the scalars over the entirety of the algorithm in a single link is:

$
  sum_(k=1)^(D-1) (frac(N^2, D) - k dot frac(N^2, D^2))
  = (D-1) frac(N^2, D) - frac((D-1) D dot N^2, 2 dot D^2)
  = frac(D-1, D) dot (N^2 - frac(N^2, 2))
$

_10.3_
$
  "Scalars"("AllGather") &= (D-1) dot frac(N^2, D) \
  "Scalars"("AllToAll") &= frac(D-1, D) dot (N^2 - frac(N^2, 2)) \
  frac("Scalars"("AllToAll"), "Scalars"("AllGather")) &= frac(1, 2)
$

It comes from the fact that the payload per link is smaller at each step.

_10.4_

With bidirectional communication the algo for AllGather takes $D / 2$ to finish, so then:

$
  frac(D, 2) dot (frac(N^2, D)) = frac(N^2, 2) quad "scalars"
$
being transferred by a single link $(i, i+1 mod D)$ or a link $(i+1 mod D, i)$.

$
  frac("Scalars"("AllGather Bidi"), "Scalars"("AllGather"))
  = frac(N^2 / 2, (D-1) dot N^2 / D)
$

$
  = frac(1, 2) dot frac(D, D-1) approx frac(1, 2) quad "as" D -> oo
$

It takes half the time.

_10.5_

With bidirectional communication the algorithm for AllToAll takes $D / 2$ to finish, so an upper bound is:
$
  sum_(k=1)^(D/2) (frac(N^2, D dot 2) - k frac(N^2, D^2))
  approx frac(D, 2) dot frac(N^2, D dot 2) - frac(N^2, D^2) dot frac(D^2, 8)
  = frac(N^2, 4) - frac(N^2, 8)
  = frac(1, 4) dot (N^2 - frac(N^2, 2))
$

scalars being transferred by a single link $(i, i+1 mod D)$ or $(i+1 mod D, i)$. So it takes $1 / 4$ of the time of the single direction AllToAll.

_10.6_

$
  frac("Scalars"("Bidi AllToAll"), "Scalars"("Bidi AllGather"))
  = frac(N^2, 8) dot frac(2, N^2) = frac(2, 8) = frac(1, 4)
$
