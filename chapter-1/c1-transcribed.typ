#set page(
  paper: "a4",
  margin: (x: 18mm, y: 17mm),
)
#set text(
  font: ("New Computer Modern", "CMU Serif", "Libertinus Serif", "Times New Roman"),
  lang: "en",
  size: 10.3pt,
)
#set heading(numbering: none)
#set par(justify: false, leading: 0.62em)

= Scaling Book Exercises -- Chapter 1

Chapter: 1 -- All About Rooflines\
Source: handwritten Onyx Boox A4 PDF\
Note: Transcribed from handwritten solutions; book markdown used only for exercise statements and notation.

== Exercise 1 -- int8 matmul

=== Exercise statement

Say we want to do the matmul $X[B, D] dot_D Y[D, F] -> Z[B, F]$ in int8 precision (1 byte per parameter) instead of bfloat16 (2 bytes per parameter) since TPUs/GPUs can do matmuls faster in lower precision.

1. How many bytes need to be loaded from memory? How many need to be written back to memory?
2. How many total OPs are performed?
3. What is the arithmetic intensity?
4. What is a roofline estimate for $T_("math")$ and $T_("comms")$? What are reasonable upper and lower bounds for the runtime of the whole operation?

Assume our HBM bandwidth is $8.2 dot 10^11 "bytes/s"$ and our int8 peak OPs/s is $3.94 dot 10^14$ (about 2x bfloat16).

=== Solution

Scan pages: 1-2

$X[B, D] dot_D Y[D, F] -> Z[B, F]$.

Bytes per int8:
$
  1 dot (B D + D F + B F) = B D + D F + B F.
$

There are $B F$ dot products. A dot product has $D$ multiplications and
$
  a_1 b_1 + a_2 b_2 + dots + a_D b_D,
$
so $D - 1$ sums. Total OPs is then
$
  "OPs" = B F (D + D - 1) = B F (2D - 1).
$

$
  "AI" = frac(B F (2D - 1), B D + D F + B F).
$

$
  T_("math")^(*)
    &= frac(B F (2D - 1), 3.94 dot 10^14 "s"^(-1)) \
    &approx frac(B F (D - 1), 2 dot 10^14) "seconds".
$

$G = 10^9$ and $T = 10^12$.

$
  T_("comms") = frac(B D + D F + B F, 8.2 dot 10^11 "bytes/s").
$

$
  max(T_("comms"), T_("math"))
    <= T_("operation")
    <= T_("comms") + T_("math").
$

== Exercise 2 -- int8 weights and bfloat16 activations

=== Exercise statement

In practice we often do different weight vs. activation quantization, so we might store our weights in very low precision but keep activations (and compute) in a higher precision. Say we want to quantize our weights in int8 but keep activations (and compute) in bfloat16. At what batch size do we become compute bound? Assume $1.97 dot 10^14$ bfloat16 FLOPs/s.

Hint: this means specifically `bf16[B, D] * int8[D, F] -> bf16[B, F]` where $B$ is the "batch size".

=== Solution

Scan pages: 3-4

Accelerator FLOPs/s for bf16:
$
  1.97 dot 10^14 approx 2 dot 10^14.
$

```text
bf16[B, D]_D @ int8[D, F] -> bf16[B, F]
```

Assuming no conv. needed to cast int8 to bf16.

$
  "AI"
    &= frac(B F (2D - 1), 2 B D + D F + 2 B F) \
    &>= frac(1.97 dot 10^14 "FLOPs/s", 8.2 dot 10^11 "bytes/s") \
    &approx frac(2, 8) dot frac(10^14, 10^11) \
    &= frac(1, 4) dot 10^3 \
    &= 2.5 dot 10^2 \
    &= 250 "FLOPs/Byte".
$

Approximating:
$
  "AI" approx frac(2 B F D, D F).
$

Assume:
$
  D >> B, quad F >> B, \
  D F >> 2 B D, quad D F >> 2 B F, \
  "then" quad D F >> 2 B D + 2 B F.
$

Hence:
$
  frac(2 B F D, D F) > 250
  quad => quad B > 125.
$

"For second I do more [illegible] at algorithm since less bytes to be transfer, so then per-algo FLOPs can be smaller."

== Exercise 3 -- roofline plot for two matrix sizes

=== Exercise statement

Taking the setup from Question 2, make a roofline plot of peak FLOPs/s vs. $B$ for $F = D = 4096$ and $F = D = 1024$. Use the exact number of bytes loaded, not an approximation.

=== Solution

Scan pages: 5-8

- Accelerator bfloat16: $1.97 dot 10^14$.
- Accelerator HBM: $8.2 dot 10^11$.

Achievable FLOPs:
$
  "AI" times "BW"
    = frac(2 B F D, 2 B D + D F + 2 B F) dot "BW".
$

Use $D = F = 4096$ and $F = D = 1024$.

$
  "Achievable FLOPs/s"
    = max(
      frac(2 B D F, 2 B D + D F + 2 B F) dot "BW",
      1.97 dot 10^14
    ).
$

Assuming $D = F$:
$
  frac(2 B D^2, 2 B D + D^2 + 2 B D)
    &= frac(2 B D^2, 4 B D + D^2) \
    &= frac(2 B D^2, D (4 B + D)) \
    &= frac(2 B D, 4 B + D).
$

Let $D = 2^k$:
$
  frac(2 B 2^k, 4 B + 2^k)
    &= frac(2 B 2^k, 2^k (frac(1 B, 2^(k - 2)) + 1)) \
    &= frac(2 B, frac(1 B, 2^(k - 2)) + 1) dot "BW".
$

$k = 12$ for $D = 4096$; $k = 10$ for $D = 1024$.

$
  log(2 dot B)
    - log(frac(1 B, 2^(k - 2)) + 1)
    + log("BW") \
  approx log(2 dot B)
    - frac(B, 2^(k - 2))
    + log("BW").
$

The plot below reconstructs the rising curves and common plateau shown in the handwritten sketch.

#align(center)[
  #image("c1-roofline-q3.png", width: 92%)
]

For $D = 4096$:
$
  frac(2 B dot 4096, 4 B + 4096) > 250 "FLOPs/token"
  quad => quad B approx 142.
$

For $D = 1024$:
$
  frac(2 B dot 1024, 4 B + 1024) > 250
  quad => quad B approx 244.
$

== Exercise 4 -- different matrix for each batch element

=== Exercise statement

What if we wanted to perform $"int8"[B, D] dot_D "int8"[B, D, F] -> "int8"[B, F]$ where we imagine having a different matrix for each batch element. What is the arithmetic intensity of this operation?

=== Solution

Scan page: 9

```text
B times int8[D] @ int8[D, F] -> int8[F]
```

$
  "FLOPs" = B (2 F D).
$

Therefore:
$
  "AI"
    = frac(B dot 2 F D, B (D + D F + F))
    = frac(2 F D, D + D F + F).
$

== Exercise 5 -- memory roofline for H100 SXM

=== Exercise statement

Using the spec sheet provided by NVIDIA for the H100 SXM, calculate the batch size at which a bfloat16 matrix multiplication will become compute-bound. Note that the Tensor Core FLOPs numbers are twice the true value since they are only achievable with structured sparsity.

=== Solution

Scan pages: 9-10

Accelerator FLOPs:
$
  frac(1979 dot 10^12, 2)
    approx 989.5 dot 10^12
    = 9.89 dot 10^14.
$

$
  frac(2 B F D, 2 B D + 2 D F + 2 B F)
    >= frac(9.84 dot 10^14, 3.35 dot 10^12).
$

Assume $D >> B$ and $F >> B$.

Hence, cancelling $2 D F$:
$
  frac(2 B F D, 2 D F) >> 295,
  quad B >> 295.
$

== Transcription uncertainties

- On scan page 4, the word after "more" in the final quoted note is illegible and is marked `[illegible]`.
- In Exercise 3, scan page 6 writes $max(...)$, while the scan-page-8 sketch shows curves capped at the common peak. The formula is transcribed as written; the generated plot follows the drawn capped shape.
- In Exercise 5, scan page 9 gives $9.89 dot 10^14$, while the numerator on scan page 10 appears to be $9.84 dot 10^14$. Both values are preserved where they occur.
