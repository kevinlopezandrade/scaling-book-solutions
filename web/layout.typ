#import "components.typ": el, transcription, void

#let site-title = [Scaling Book Solutions]
#let site-description = "Detailed Typst transcriptions of worked scaling book exercises."
#let theme = "/assets/theme"

#let chapter-document-title(chapter) = [
  Chapter #(chapter.number): #(chapter.title)
]

#let page-head(title, assets, description: none) = [
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
    href: assets + "/favicon.svg",
    type: "image/svg+xml",
  ))
  #void("link", attrs: (rel: "stylesheet", href: theme + "/article.css"))
  #void("link", attrs: (rel: "stylesheet", href: assets + "/typst.css"))
]

#let site-header(home) = el("header", attrs: (class: "site-header"))[
  #el("nav", attrs: ("aria-label": "Site"))[
    #el("a", attrs: (
      href: "/",
      "aria-label": "Main website",
      title: "Main website",
    ))[/]
    #el("a", attrs: (href: home))[Chapters]
    #el("a", attrs: (
      href: "https://github.com/kevinlopezandrade/scaling-book-solutions",
    ))[Code]
  ]
]

#let table-of-contents(document-title) = outline(
  title: none,
  target: selector(heading.where(level: 2)).within(
    document.where(title: document-title),
  ),
)

#let home-page(chapters) = el("html", attrs: (lang: "en"))[
  #el("head")[
    #page-head(
      site-title,
      "assets",
      description: site-description,
    )
  ]
  #el("body")[
    #site-header("index.html")
    #el("main", attrs: (class: "post"))[
      #el("header", attrs: (class: "post-heading"))[#title()]
      #el("div", attrs: (class: "post-content"))[
        #el("p")[
          Typst transcriptions of my solutions to the exercises in
          #el("a", attrs: (href: "https://jax-ml.github.io/scaling-book/"))[
            #emph[How to Scale Your Model]
          ]
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
]

#let chapter-page(chapter, document-title) = el("html", attrs: (lang: "en"))[
  #el("head")[
    #page-head(
      document-title,
      "../assets",
    )
  ]
  #el("body")[
    #site-header("../index.html")
    #el("main", attrs: (class: "post", id: "top"))[
      #el("header", attrs: (class: "post-heading"))[
        #title()
        #el("p", attrs: (class: "post-meta"))[Worked solutions]
      ]
      #el("div", attrs: (class: "post-body"))[
        #el("aside", attrs: (class: "toc", "aria-label": "On this page"))[
          #el("span", attrs: (class: "toc-marker", "aria-hidden": "true"))[]
          #table-of-contents(document-title)
        ]
        #el("article", attrs: (class: "post-content transcription"))[
          #transcription(chapter.source)
        ]
      ]
      #el("footer", attrs: (class: "post-footer"))[
        #el("a", attrs: (href: "../index.html"))[All chapters]
        #el("a", attrs: (href: "#top"))[Back to top ↑]
      ]
    ]
    #el("script", attrs: (src: theme + "/toc.js", type: "module"))[]
  ]
]
