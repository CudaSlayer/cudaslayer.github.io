# Minimal React Blog — Deep Learning, Math, Physics

This repository hosts a minimal, dependency-light blog built with React via CDN, Markdown for posts, and KaTeX for math. Designed for GitHub Pages, no build step is required.

## Features

- Simple React SPA with hash routing (`#/` and `#/post/:slug`)
- Dark: Catppuccin Mocha • Light: AnuPpuccin-inspired
- Markdown posts fetched from `/posts/*.md`
- KaTeX auto-render for `$..$` and `$$..$$`

## Structure

- `index.html` — entry with CDN scripts and CSS
- `assets/styles.css` — theme variables and layout
- `assets/app.js` — SPA logic, routing, markdown + math rendering
- `posts/index.json` — post manifest (slug, title, date, file, ...)
- `posts/*.md` — markdown source for each post
- `404.html` — redirect fallback for GitHub Pages

## Writing a new post

1. Add a Markdown file under `posts/your-post.md`.
2. Update `posts/index.json` with an object:

   ```json
   {
     "slug": "your-post",
     "title": "Your Post Title",
     "date": "2025-09-15",
     "tags": ["deep-learning"],
     "summary": "Optional summary.",
     "file": "posts/your-post.md"
   }
   ```

3. Open `#/post/your-post` to view it.

## Math

Use `$inline$` or `$$display$$` LaTeX delimiters. KaTeX auto-render is included via CDN in `index.html`.

## Theming

Theme toggle persists preference (`localStorage`). Variables live in `assets/styles.css` under `html[data-theme="dark"]` (Mocha) and `html[data-theme="light"]` (AnuPpuccin). The Rationale font is loaded from Google Fonts.

## Local preview

Serve the folder locally (any static server). Example with Python:

```bash
python3 -m http.server -b 127.0.0.1 8080
```

Then visit `http://127.0.0.1:8080`.
