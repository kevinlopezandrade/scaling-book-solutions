#let el(tag, attrs: (:), body) = html.elem(tag, attrs: attrs, body)
#let void(tag, attrs: (:)) = html.elem(tag, attrs: attrs)

#let html-align(it) = block(
  el("div", attrs: (
    class: "typst-align",
    "data-alignment": repr(it.alignment),
  ), it.body),
)

#let transcription(source) = [
  // The web page already supplies a chapter title; retain this one only in PDFs.
  #show heading.where(level: 1): none
  // Give the source and recording links their own web-only styling hook.
  #show <chapter-resources>: it => el("div", attrs: (
    class: "chapter-resources",
  ), it.body)
  // Preserve content while mapping layout primitives unsupported by HTML.
  #show align: html-align
  #include source
]
