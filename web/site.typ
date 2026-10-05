#import "chapters.typ": chapters
#import "layout.typ": chapter-document-title, chapter-page, home-page, site-title

#document("index.html", title: site-title)[
  #home-page(chapters)
]

#for chapter in chapters {
  let document-title = chapter-document-title(chapter)
  counter(heading).update(0)
  document(
    chapter.output,
    title: document-title,
    chapter-page(chapter, document-title),
  )
}

#asset("assets/typst.css", read("src/typst.css", encoding: none))
#asset("assets/favicon.svg", read("src/favicon.svg", encoding: none))

// Production uses the main website's theme. Preview serves those same files
// locally and watches them for changes, alongside the chapter sources.
#if sys.inputs.at("preview", default: "false") == "true" {
  let theme = "../../web/assets/theme/"
  for file in json(theme + "assets.json") {
    asset(
      "assets/theme/" + file,
      read(theme + file, encoding: none),
    )
  }
}
