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

#asset("assets/site.css", read("src/site.css", encoding: none))
#asset("assets/toc.js", read("src/toc.js", encoding: none))
#asset("assets/favicon.svg", read("src/favicon.svg", encoding: none))

#for font in (
  "iowan-regular",
  "iowan-italic",
  "iowan-bold",
  "gt-america-regular",
  "chakra-petch-medium",
) {
  asset(
    "assets/fonts/" + font + ".woff2",
    read("src/fonts/" + font + ".woff2", encoding: none),
  )
}
