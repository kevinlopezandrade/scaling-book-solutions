You are helping me convert my handwritten solutions into a clean, editable Typst document.

Inputs I will provide:
1. A scanned/exported A4 PDF from my Onyx Boox containing my handwritten solutions. The file is named `cx-handwritten.pdf`, where `x` is the chapter or part number, for example `c2-handwritten.pdf` for Chapter 2 or `c11-handwritten.pdf` for Chapter/Part 11.
2. The full book markdown file: `scaling-book-combined.md`.

Your task:
Convert ONLY my handwritten solutions from the scanned A4 PDF into a polished Typst document, compile it with `typst.compile(...)`, and give me back both:
- the `.typ` source file
- the compiled `.pdf`

Critical source-separation rule:
The scanned A4 PDF is the only source of my solution content.

The markdown book file may be used ONLY for:
- identifying the relevant chapter or part from the `cx-handwritten.pdf` filename
- recovering exercise numbers, titles, and statements
- understanding notation used in the book
- checking section names or chapter structure

The markdown book file must NOT be used to:
- copy, paraphrase, complete, repair, or improve my solutions
- fill in missing algebra or missing reasoning
- replace my handwritten derivation with the book's derivation
- silently correct my answer using the book solution
- add steps that are present in the book but absent from my handwriting

The book contains official/reference solutions. Do not mix those into my writeup. If my handwritten solution is incomplete, wrong, messy, or ambiguous, preserve that fact rather than using the book to fix it.

Target transcription style:
Use a direct, plain Typst style that is compact and easy to edit. The output should look like a cleaned transcription of handwritten work, not a polished textbook rewrite. Avoid custom wrapper blocks, decorative boxes, and repeated narration about what the handwriting says.

Required Typst structure:
- Use Typst, not LaTeX.
- Create a self-contained `.typ` file.
- Format for A4 paper.
- Prefer plain Typst constructs. Do not define custom `statement`, `solution`, or `note` block helpers unless there is a strong reason.
- Do not put exercise statements or solutions inside decorative filled/stroked boxes.
- Use a simple text setup near the top, similar to:

```typst
#set page(paper: "a4")
#set text(
  font: ("New Computer Modern", "CMU Serif", "Libertinus Serif", "Times New Roman"),
  lang: "en",
)

= Scaling Book Exercises -- {CHAPTER_OR_PART_LABEL} {NUMBER}

{CHAPTER_OR_PART_LABEL}: {NUMBER} -- {TITLE}\
Source: handwritten Onyx Boox A4 PDF\
Note: Transcribed from handwritten solutions; book markdown used only for exercise statements and notation.
```

For each exercise, use this shape:

```typst
== Exercise {N} -- {exercise title}

=== Exercise statement

{statement copied from the book markdown}

=== Solution

Scan pages: {page range in the input scan}

{direct transcription of my handwritten work}
```

When transcribing:
- Faithfully convert my handwritten math, text, diagrams, and code-like snippets into Typst.
- Use Typst math syntax, not LaTeX.
- Preserve my reasoning structure, assumptions, intermediate steps, dead ends, approximations, and final answers.
- Preserve my actual wording when it is legible, including terse notes like `Assume:`, `Since:`, `=>`, `like`, `approx`, and short explanatory fragments.
- Do not rewrite the solution into polished textbook prose. It should read like a cleaned transcription of my work, not a new explanation.
- Do not repeatedly write phrases such as "The handwritten setup is transcribed as" or "The handwritten note says". JUST typeset the content directly.
- Keep the ordering of problems as shown in my scanned PDF unless the scan clearly labels them otherwise.
- Include exercise statements extracted from the markdown book, clearly separated from my solution under `=== Exercise statement`.
- Put my work under `=== Solution`, beginning with `Scan pages: ...`.
- If something in my handwriting is illegible, write `[illegible]` or add a Typst comment near that location. Do not guess from the book solution.
- If a symbol is uncertain, mark it inline, for example `x [?]`, or add a short final `== Transcription uncertainties` section.
- Do not invent missing derivation steps.
- Do not simplify, optimize, or rewrite my proof in a way that changes its content.
- You can improve layout, notation consistency, spacing, headings, and obvious transcription formatting, but not the mathematical substance. I often write many blank spaces and place each implication arrow or derivation step on a new handwritten line; in Typst, group related steps into readable display equations without changing the substance.

Formatting guidance:
- Use readable display math for multi-step derivations:

```typst
$
  T_("comms")
    = frac(12.5 dot 10^9 "bytes", 1.2 e 12 "bytes/s")
    approx 10 dot 10^(-3) "s"
    = 10 "ms".
$
```

- Use normal paragraphs and bullets for assumptions or hardware facts.
- Use fenced `text` code blocks for ASCII sketches, diagrams, tensor layouts, arrows, and simple tables from the handwriting.
- Preserve diagrams as ASCII sketches when that is the clearest faithful transcription. Do not replace a handwritten schematic with only a prose description.
- Use Typst strings inside math for units and labels, for example `$12.5 "GB"$`, `$T_("comms")$`, and `$"bf16" [B, D]$`.
- Do not put leading or trailing spaces inside quoted math strings. Prefer `$10 "ms"$`, `$10^9 "bytes"$`, and `$1.2 e 12 "bytes/s"$`, not `$10 " ms"$` or `$10^9 " bytes"$`.
- Keep approximate handwritten arithmetic as written when it is part of the solution, even if it is rough.
- If my answer appears numerically or conceptually different from the book, preserve my answer and mark uncertainty only if the handwriting itself is unclear.
- Use page breaks only where useful. Do not force every exercise onto a new page unless the document genuinely reads better that way.
- Avoid external Typst packages unless absolutely necessary.

Typst compilation requirements:
- Use the installed Typst Python binding with `typst.compile(...)`.
- Compile the `.typ` file to PDF.
- If compilation fails, debug the Typst syntax and recompile until successful.
- After compilation, inspect/render the resulting PDF enough to verify:
  - it opens successfully
  - pages are not blank
  - major equations are visible
  - text is not obviously clipped
  - the document corresponds to the handwritten PDF and matching book chapter/part

Final output:
Give me links to:
1. the compiled PDF
2. the `.typ` source file

Also include a brief note with:
- how many pages were in the input scan
- how many exercises were transcribed
- whether any handwriting was illegible or uncertain
- whether the PDF compiled successfully

Hard prohibition:
Never use the book's solution text to complete or alter my solution. If the scanned solution is unclear, say so. If my solution appears different from the book solution, preserve my version.
