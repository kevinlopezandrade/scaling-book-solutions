You are helping me convert my handwritten solutions into a clean Typst document.

Inputs I will provide:
1. A scanned/exported A4 PDF from my Onyx Boox containing my handwritten solutions.
2. A chapter number.
3. The full book markdown file: `scaling-book-combined.md`.

Your task:
Convert ONLY my handwritten solutions from the scanned A4 PDF into a polished Typst document, compile it with `typst.compile(...)`, and give me back both:
- the `.typ` source file
- the compiled `.pdf`

Critical source-separation rule:
The scanned A4 PDF is the only source of my solution content.

The markdown book file may be used ONLY for:
- identifying the relevant chapter
- recovering exercise numbers/titles/statements
- understanding notation used in the book
- checking section names or chapter structure

The markdown book file must NOT be used to:
- copy, paraphrase, complete, repair, or improve my solutions
- fill in missing algebra or missing reasoning
- replace my handwritten derivation with the book’s derivation
- silently correct my answer using the book solution
- add steps that are present in the book but absent from my handwriting

The book contains official/reference solutions. Do not mix those into my writeup. If my handwritten solution is incomplete, wrong, messy, or ambiguous, preserve that fact rather than using the book to fix it.

When transcribing:
- Faithfully convert my handwritten math, text, diagrams, and code-like snippets into Typst.
- Use Typst math syntax, not LaTeX.
- Preserve my reasoning structure, including intermediate steps.
- Keep the ordering of problems as shown in my scanned PDF unless the scan clearly labels them otherwise.
- You must include exercise statements extracted from the markdown book, clearly separated from my solution.
- Put my work under a clearly marked “Solution” block.
- If something in my handwriting is illegible, write `[illegible]` or add a Typst comment near that location. Do not guess from the book solution.
- If a symbol is uncertain, mark it, for example: `x [?]`, or add a short note in a final “Transcription uncertainties” section.
- Do not invent missing derivation steps.
- Do not simplify, optimize, or rewrite my proof in a way that changes its content.
- You can clean up layout, notation consistency, spacing, headings, and obvious
  transcription formatting, but not the mathematical substance. In paper I tend
  to write with a lot spaces, and make a newline per implication arrow or per
  any derivation step, it helps me in the paper, but for the transcription it
  does not look great, improve this, withouth changing the substance.


Document requirements:
- Use Typst, not LaTeX.
- Create a self-contained `.typ` file.
- Format for A4 paper.
- Include a title like:
  `Scaling Book Exercises — Chapter {CHAPTER_NUMBER}`
- Include metadata near the top:
  - chapter number
  - source: handwritten Onyx Boox A4 PDF
  - note: “Transcribed from handwritten solutions; book markdown used only for exercise statements and notation.”
- Use clear section headings for each exercise.
- Use readable math formatting.
- Use page breaks where useful.
- Avoid external Typst packages unless absolutely necessary.
- Prefer plain Typst constructs so the file remains easy to edit later.

Typst compilation requirements:
- Use the installed Typst Python binding with `typst.compile(...)`.
- Compile the `.typ` file to PDF.
- If compilation fails, debug the Typst syntax and recompile until successful.
- After compilation, inspect/render the resulting PDF enough to verify:
  - it opens successfully
  - pages are not blank
  - major equations are visible
  - text is not obviously clipped
  - the document corresponds to the requested chapter and scanned PDF

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
Never use the book’s solution text to complete or alter my solution. If the scanned solution is unclear, say so. If my solution appears different from the book solution, preserve my version.
