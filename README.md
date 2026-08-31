# Project page — *OMa: Dense Object Matching for Dense Reconstruction*

Source for the project page (codebase: **GloPose**), served via GitHub Pages at
**https://tjelinek.github.io/oma/**.

Paper: *OMa: Dense Object Matching for Dense Reconstruction* — Tomáš Jelínek, Dmytro Mishkin, Jiří Matas
(Visual Recognition Group, CTU in Prague). Title/authors/abstract/BibTeX mirror the Overleaf paper.

> The repo was renamed from `glopose-page` to `oma` (GitHub auto-redirects the old
> `tjelinek.github.io/glopose-page/` URL to the new one).

Built from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
(Nerfies-derived). Static HTML/CSS/JS — no build step.

## Edit & deploy

Edit `index.html`, commit, and push to `main`; GitHub Pages redeploys automatically.

```bash
git add -A && git commit -m "Update page" && git push
```

> Note: this repo's `origin` uses **SSH** (`git@github.com:tjelinek/oma.git`) — the HTTPS
> credential helper wasn't working on this machine.

## TODO before submission / release

Search `index.html` for `TODO(...)` markers (title/authors/abstract/method/BibTeX are filled from the paper):

- `TODO(venue)` — set the venue once decided / accepted (paper is currently under review).
- `TODO(arxiv)` — arXiv id, and wire the **Paper** / **arXiv** buttons (currently `href="#"`).
- `TODO(bibtex)` — switch from the `@misc` preprint entry to the published `@inproceedings` on acceptance.
- `TODO(meta)` — `citation_pdf_url` + venue once the paper is public.
- Result gallery — the media carousel is commented out until qualitative result assets exist.
  Source figures live in the paper repo: `glopose-paper/figs/gallery/`.

The teaser (`static/images/teaser.png`) is rendered from the paper's `figs/teaser_static.pdf`.
The favicon was intentionally removed (blank browser-tab icon via `<link rel="icon" href="data:,">`).

## Layout

- `index.html` — the page.
- `static/{css,js,images,videos,pdfs}` — template assets (`images/teaser.png` is the hero figure).
- `.nojekyll` — tells GitHub Pages to serve files as-is (no Jekyll processing).
