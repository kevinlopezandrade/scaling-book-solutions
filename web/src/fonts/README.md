# Web math font

`NewCMMath-Book.woff2` is the **Book** face of New Computer Modern Math by
Antonis Tsolomitis, copyright 2019–2026. It is used at normal CSS weight 400:
the heavier strokes are part of the face, not browser-synthesized bold.

Source: <https://ctan.org/pkg/newcomputermodern>, package 8.1.1.
Upstream file: <https://mirrors.ctan.org/fonts/newcomputermodern/otf/NewCMMath-Book.otf>
(retrieved 2026-10-05). The font's internal version string is `4.0`.

The Book math face is under the GUST Font License. Its embedded copyright
notice and the package's `License.txt` identify that license; the package's
GPL exceptions apply to other faces, not this one. See the accompanying
`GUST-FONT-LICENSE.txt` and `LPPL-1.3c.txt`.

The WOFF2 file is a compressed copy of the upstream OTF, without subsetting,
glyph changes, or changes to the Unicode mapping, metrics, or MATH table.
Upstream OTF SHA-256:

```text
2ea09ebc9167b1e1a66f31390dc917f2d4004ecfca72d51b28010e4ad6becd95
```

Bundled WOFF2 SHA-256:

```text
41261f362726a687c17987304576b0f178f7660a2d12933a88fc52bca6574220
```

To reproduce the conversion from the matching upstream OTF:

```sh
uv run --no-project --with 'fonttools[woff]==4.66.1' fonttools ttLib.woff2 compress \
  NewCMMath-Book.otf -o NewCMMath-Book.woff2
```

Normal builds use the checked-in WOFF2 directly and need neither FontTools
nor TeX Live. `web/site.typ` bundles the font and licenses under `assets/fonts/`
in production and previews. `web/src/typst.css` applies the face only to
`.transcription math`; prose and PDF output are unaffected.
