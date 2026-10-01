#import "components.typ": el, transcription, void

#let site-title = [Scaling Book Solutions]
#let site-description = "Detailed Typst transcriptions of worked scaling book exercises."

#let chapter-document-title(chapter) = [
  Chapter #(chapter.number): #(chapter.title)
]

#let page-head(title, stylesheet, favicon, description: none) = [
  #void("meta", attrs: (charset: "utf-8"))
  #void("meta", attrs: (
    name: "viewport",
    content: "width=device-width, initial-scale=1",
  ))
  #if description != none {
    void("meta", attrs: (name: "description", content: description))
  }
  #el("title")[#title]
  #void("link", attrs: (
    rel: "icon",
    href: favicon,
    type: "image/svg+xml",
  ))
  #void("link", attrs: (rel: "stylesheet", href: stylesheet))
]

#let table-of-contents(document-title) = el(
  "aside",
  attrs: (class: "toc"),
)[
  #el("a", attrs: (href: "../index.html"))[All chapters]
  #outline(
    title: [Contents],
    target: selector(heading.where(level: 2)).within(
      document.where(title: document-title),
    ),
  )
]

#let home-page(chapters) = el("html", attrs: (lang: "en"))[
  #el("head")[
    #page-head(
      site-title,
      "assets/site.css",
      "assets/favicon.svg",
      description: site-description,
    )
  ]
  #el("body")[
    #el("main", attrs: (class: "home"))[
      #title()
      #el("p")[
        Typst transcriptions of detailed solutions to the exercises in
        #emph[How to Scale Your Model].
      ]
      #el("nav", attrs: (
        class: "chapter-list",
        "aria-label": "Chapters",
      ))[
        #el("ol")[
          #for chapter in chapters {
            el("li")[
              #el("a", attrs: (href: chapter.output))[
                Chapter #(chapter.number): #(chapter.title)
              ]
            ]
          }
        ]
      ]
    ]
  ]
]

#let chapter-page(chapter, document-title) = el(
  "html",
  attrs: (lang: "en"),
)[
  #el("head")[
    #page-head(
      document-title,
      "../assets/site.css",
      "../assets/favicon.svg",
    )
  ]
  #el("body")[
    #el("div", attrs: (class: "site-shell"))[
      #table-of-contents(document-title)
      #el("main")[
        #el("article", attrs: (class: "chapter"))[
          #title()
          #transcription(chapter.source)
        ]
      ]
    ]
  ]
]
