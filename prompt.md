Convert my handwritten A4 PDF solutions for Chapter {CHAPTER_NUMBER} into a clean Typst document and compile it to PDF.

Inputs:
- Handwritten solution scan: {HANDWRITTEN_PDF}
- Full book markdown: `scaling-book-combined.md`
- Chapter: {CHAPTER_NUMBER}

Use the handwritten PDF as the ONLY source for my solution content.

The book markdown contains solutions. You may use it only to identify the
chapter, exercise numbers, exercise statements, notation, and section
structure. You must not copy, paraphrase, complete, correct, or improve my
solutions using the book’s solutions. If my handwriting is incomplete,
ambiguous, or wrong, preserve that rather than fixing it from the book.

You must include exercise statements extracted from the markdown book, clearly
separated from my solution.

Output requirements:
- Create an editable `.typ` file.
- Compile it using `typst.compile(...)`.
- Return both the compiled PDF and the `.typ` source.
- Use A4 layout.
- Use Typst math syntax, not LaTeX.
- Mark illegible handwriting as `[illegible]`.
- Mark uncertain transcriptions with `[?]` or a short note.
- Do not invent missing steps.

Before finalizing:
- compile successfully
- inspect the rendered PDF
- fix Typst syntax/layout errors
- confirm the result is not blank or clipped

Final response should include:
- link to PDF
- link to `.typ`
- number of exercises transcribed
- list of uncertainties/illegible parts, if any
- confirmation that no book solution content was used
