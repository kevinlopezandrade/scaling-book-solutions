#set page(paper: "a4", margin: (x: 18mm, y: 17mm))
#set text(size: 10.4pt, lang: "en")
#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.")

#let statement(body) = block(width: 100%, fill: rgb("#f6f6f6"), stroke: rgb("#d7d7d7"), radius: 4pt, inset: 8pt)[
  *Exercise statement (from book markdown).* #body
]

#let solution(body) = block(width: 100%, fill: rgb("#fbfcff"), stroke: rgb("#c9d7ef"), radius: 4pt, inset: 8pt)[
  *Transcribed solution (from handwriting only).* #body
]

#let note(body) = block(width: 100%, fill: rgb("#fff8e6"), stroke: rgb("#ead7a0"), radius: 4pt, inset: 8pt)[#body]

= Chapter 1 Solutions: All About Rooflines

#note[
This document typesets the exercise statements for Chapter 1 and the handwritten solution content from `C1.pdf`. The solution sections preserve the handwritten derivations, including incomplete or uncertain portions. Book solution content was not used to complete, correct, or improve the transcriptions.
]

== Exercise 1: int8 matmul

#statement[
Say we want to do the matmul $X[B, D] dot_D Y[D, F] arrow.r Z[B, F]$ in int8 precision (1 byte per parameter) instead of bfloat16 (2 bytes per parameter) since TPUs/GPUs can do matmuls faster in lower precision.

1. How many bytes need to be loaded from memory? How many need to be written back to memory?
2. How many total OPs are performed?
3. What is the arithmetic intensity?
4. What is a roofline estimate for $T_"math"$ and $T_"comms"$? What are reasonable upper and lower bounds for the runtime of the whole operation?

Assume our HBM bandwidth is `8.2e11` bytes/s and our int8 peak OPs/s is `3.94e14` (about 2x bfloat16).
]

#solution[
The handwritten setup is
$ X[B, D] dot_D Y[D, F] arrow.r Z[B, F]. $

Communication bytes are written as
$ B dot D + D dot F + B dot F. $

The dot products are counted as follows: there are $B dot F$ dot products, and each dot product has $D$ multiplications and $D - 1$ sums. Therefore,
$ "OPs" = B dot F dot (D + D - 1) = B dot F dot (2 dot D - 1). $

The arithmetic intensity is written as
$ "AI" = frac(B dot F dot (2 dot D - 1), B dot D + D dot F + B dot F). $

The math time is
$ T_"math" = frac(B dot F dot (2 dot D - 1), 3.94 times 10^14). $

The handwritten approximation note for this line is
$ T_"math" approx frac(B dot F dot (D - 1), 2 times 10^14) quad "seconds". $

The communication time is
$ T_"comms" = frac(B dot D + D dot F + B dot F, 8.2 times 10^11). $

The operation time is bounded by
$ max(T_"comms", T_"math") <= T_"operation" <= T_"comms" + T_"math". $
]

== Exercise 2: int8 + bf16 matmul

#statement[
In practice we often do different weight vs. activation quantization, so we might store our weights in very low precision but keep activations (and compute) in a higher precision. Say we want to quantize our weights in int8 but keep activations (and compute) in bfloat16. At what batch size do we become compute bound? Assume `1.97e14` bfloat16 FLOPs/s.

Hint: this means specifically `bf16[B, D] * int8[D, F] -> bf16[B, F]` where $B$ is the batch size.
]

#solution[
The accelerator FLOPs for bfloat16 are transcribed as
$ 1.97 times 10^14 approx 2 times 10^14. $

The operation is written as
$ "bf16"[B, D] dot_D "int8"[D, F] arrow.r "bf16"[B, F]. $

A handwritten note says: assuming no copy is needed to cast int8 to bf16.

The arithmetic intensity is written as
$ "AI" = frac(B dot F dot (2 dot D - 1), 2 dot B dot D + D dot F + 2 dot B dot F). $

The hardware ratio is written as
$ frac(1.97 times 10^14 " FLOPs/s", 8.2 times 10^11 " bytes/s") approx frac(2, 8) dot frac(10^14, 10^11) = frac(1, 4) dot 10^3 = 2.5 dot 10^2. $

Approximating the intensity, the handwritten assumptions are
$D >> B, quad F >> B, quad D F >> 2 dot B D, quad D F >> 2 dot B F,$
so
$D F >> 2 dot B D + 2 dot B F.$

Then
$ frac(2 dot B dot F dot D, D F) > 250, $
so
$ B > 125. $

The final handwritten prose note is transcribed as:
#quote(block: true)[
"Per second I do more starts of the algorithm since less bytes to be transfer, so then per algorithm FLOPs can be smaller"
]
]

== Exercise 3: roofline plot for two sizes

#statement[
Taking the setup from Question 2, make a roofline plot of peak FLOPs/s vs. $B$ for $F = D = 4096$ and $F = D = 1024$. Use the exact number of bytes loaded, not an approximation.
]

#solution[
The accelerator numbers are written as:

- Accelerator bfloat16: $1.97 times 10^14$.
- Accelerator HBM: $8.2 times 10^11$.

The handwritten formula for achievable FLOPs is
$ "Achievable FLOPs" = "AI" times "BW". $

Using the exact bytes from the handwritten notes:
$ frac(2 dot B F D, 2 dot B D + D F + 2 dot B F) dot "BW". $

The two cases are
$D = F = 4096 quad | quad F = D = 1024.$

The handwritten line for achievable FLOPs/s is written as
$ "Achievable FLOPs/s" = min( frac(2 dot B D F, 2 dot B D + D F + 2 dot B F) dot "BW", 1.97 times 10^14 ). $

Assuming $D = F$:
$ frac(2 dot B dot D^2, 2 dot B dot D + D^2 + 2 dot B dot D)
 = frac(2 dot B dot D^2, 4 dot B dot D + D^2)
 = frac(2 dot B dot D^2, D dot (4 dot B + D))
 = frac(2 dot B dot D, 4 dot B + D). $

With $D = 2^k$,
$ frac(2 dot B dot 2^k, 4 dot B + 2^k)
 = frac(2 dot B dot 2^k, 2^k dot (frac(B, 2^(k - 2)) + 1))
 = frac(2 dot B, frac(B, 2^(k - 2)) + 1) dot "BW". $

The handwritten parameter choices are:

- $K = 12$ for $D = 4096$.
- $K = 10$ for $D = 1024$.

The logarithmic form is written as
$ log(2 dot B) - log(frac(B, 2^(k - 2)) + 1) + log("BW"). $

The next approximation is written as
$ approx log(2 dot B) - frac(B, 2^(k - 2)) + log("BW"). $

The handwritten sketch is a roofline plot with vertical axis $log("FLOPs")$ and horizontal axis $log(B)$. It shows two rising lines that flatten at a horizontal peak; the line labeled `4096` reaches the flat region before the line labeled `1024`. The horizontal axis has marked positions near `7` and `8`.

The threshold calculations under the sketch are transcribed as
$ frac(2 dot B dot 4096, 4B + 4096) > 250 quad "FLOPs/Byte" $
which gives
$ B approx 142, $

and
$ frac(2 dot B dot 1024, 4B + 1024) > 250, $
which gives
$ B approx 244. $
]

== Exercise 4: batch-specific matrices

#statement[
What if we wanted to perform $"int8"[B, D] dot_D "int8"[B, D, F] arrow.r "int8"[B, F]$ where we imagine having a different matrix for each batch element. What is the arithmetic intensity of this operation?
]

#solution[
The handwritten setup is transcribed as:
$ B " times " "int8"[D] dot "int8"[D, F] arrow.r "int8"[F] $

The FLOPs count is
$ "FLOPs" = B dot (2 dot F D). $

Therefore,
$ "AI" = frac(B dot 2 dot F D, B dot (D + D F + F))
 = frac(2 dot F D, D + D F + F). $
]

== Exercise 5: Memory rooflines for GPUs

#statement[
Using the spec sheet provided by NVIDIA for the H100 SXM, calculate the batch size at which a bfloat16 matrix multiplication will become compute-bound. Note that the Tensor Core FLOPs numbers are twice the true value since they're only achievable with structured sparsity.
]

#solution[
The accelerator FLOPs are transcribed as
$ frac(1979, 2) dot 10^12 approx 989.5 dot 10^12 = 9.89 times 10^14. $

The condition is written as
$ frac(2 dot B F D, 2 dot B D + 2 dot D F + 2 dot B F) > frac(9.84 times 10^14, 3.75 times 10^12). $

Using the handwritten assumptions
$D >> B, quad F >> B,$
this becomes
$ frac(2 dot B F D, 2 dot D F) > 295. $

Thus,
$ B > 295. $
]
