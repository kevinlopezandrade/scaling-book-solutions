# Scaling, Worked Out

Detailed, web-native worked solutions to the exercises in Google DeepMind's
[_How to Scale Your Model_](https://jax-ml.github.io/scaling-book/).

The project separates source work from publication:

- `chapter-N/cN-handwritten.pdf` contains the original handwritten work.
- `chapter-N/cN-transcribed.typ` is the canonical editable transcription for
  both PDF and HTML output.
- `template/layout.typ` is the single chapter template. It intentionally retains
  Typst's native paged defaults until the print design is defined.
- `web/site.typ` builds the Typst content into a minimal static site with
  Typst 0.15's native bundle and MathML export.

The book is used only for exercise statements, chapter structure, and notation.
Solution reasoning comes from the handwritten scans and is kept separate from
the book's reference answers.

## Run Locally

Requirements:

- Node.js 20.18.1 or newer
- Typst 0.15.1 exactly
- The main website checkout at `../web/` for previews and shared-asset checks

Keep the two repositories beside each other:

```text
parent/
  web/                     # Main website and assets/theme/
  scaling-book-solutions/
```

```sh
cd web
npm install
npm run dev
```

Typst's bundle watcher serves the site at <http://127.0.0.1:4173> and reloads
the browser when the Typst sources or CSS change. It reads the shared article
theme directly from `../web/assets/theme/` and serves it locally, including the
fonts and TOC script. Preview output goes into the ignored
`web/.artifacts/preview/` directory, separate from the production build.

Typst still marks HTML and bundle export as experimental. The build pins 0.15.1
and fails before compilation when another Typst version is active.

Compile an individual chapter PDF from the repository root:

```sh
typst compile --root . chapter-1/c1-transcribed.typ chapter-1/c1-transcribed.pdf
```

The repository root is required because chapter files import the chapter template.
Every canonical transcription starts with the same hook:

```typst
#import "../template/layout.typ": with-layout
#show: with-layout
```

New print styling should be added to `template/layout.typ`, not to individual
chapter preambles. Content-specific structure such as image sizing and table
columns remains in the chapter source.

## Build And Verify

```sh
cd web
npm run verify
```

The generated static site is written to `web/dist/`. A single Typst compilation
emits the chapter index, eight chapter pages, a small Typst-specific stylesheet,
and an SVG favicon. Chapter content is wrapped in a stable semantic article
layout, and each table of contents is a document-scoped native Typst outline.
A small shared script highlights the current section; the content itself is
static HTML and native MathML.

Production pages load the shared article CSS, fonts, and TOC script from
`/assets/theme/` at the domain root. The main website owns those files. Deploy
them before deploying this build at `/posts/scaling-book-solutions/`. Do not
publish the preview directory. `npm run build` works without the main website
checkout; `npm run dev` and `npm run check` need it to read the shared assets.

`verify` builds the site, checks the generated HTML, and compiles every chapter
to a temporary PDF. The individual commands are `npm run build`, `npm run check`,
and `npm run check:pdf`. Run the build before checking the HTML.

The HTML check uses Cheerio to read HTML elements. It checks that:

- The homepage lists every chapter directory in order, and every chapter page exists.
- MathML contains no explicit error elements.
- Contents links cover the chapter sections, and element IDs are unique.
- Local links, fragment targets, stylesheets, scripts, and image files exist,
  including references to the main website's shared article theme.

These checks cover navigation, local files, and explicit MathML errors. Content
completeness, embedded images, and visual layout are reviewed manually. The tests
do not inspect Typst source syntax, compare text or equations, or check external
websites. They use no regular expressions.

## Review A Chapter Before Publishing

After the automatic checks pass, run `npm run dev` and open the chapter at
<http://127.0.0.1:4173>. Review it at both a desktop width and a narrow mobile width:

- Compare the chapter with its canonical transcription for missing or incorrect content.
- Read through the equations and check fractions, subscripts, alignment, and scrolling.
- Check figures, tables, code blocks, and list numbering for readability or clipping.
- Follow the homepage and contents links to check navigation.

After publishing, repeat this review on the live chapter. Display checks are
manual; there is no automated browser test or Chrome dependency.

The web source is split by responsibility:

- `web/chapters.typ` is the explicit chapter registry.
- `web/layout.typ` owns the semantic HTML documents.
- `web/components.typ` contains small HTML compatibility helpers.
- `web/site.typ` declares bundle documents and local assets, plus shared assets
  for previews only.
- `web/src/typst.css` contains image-alignment and MathML adjustments.
- `../web/assets/theme/` owns the shared article CSS, fonts, and TOC script.
