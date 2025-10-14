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

## Integrations

Both comments and engagement counters are optional; the UI hides itself when configuration is missing.

### GitHub-powered comments (giscus)

1. Follow <https://giscus.app> to generate repository, category, and ID values.
2. Populate the `window.__BLOG_GISCUS` object in `index.html` with the values that giscus provides (`repo`, `repoId`, `category`, `categoryId`, optional custom themes).
3. The SPA automatically mounts giscus at the end of every post and keeps the embedded theme in sync with the site toggle.

### GitHub-backed likes & view counts

1. Create (or choose) a GitHub repository to store metrics. Open an issue and add a comment whose body is `{ "slugs": {} }`. Note the numeric comment ID.
2. Deploy the Cloudflare Worker in `workers/github-metrics.js` with `wrangler`. The worker expects these environment variables:
   - `GITHUB_TOKEN` — fine-grained PAT with `issues:read` and `issues:write` on the repo above.
   - `GITHUB_OWNER` / `GITHUB_REPO` — repository that holds the metrics comment.
   - `GITHUB_COMMENT_ID` — ID of the comment created in step 1.
3. Set the worker's public URL as `window.__BLOG_METRICS.baseUrl` in `index.html`.
4. Each page view sends `POST /view`; clicking the like chip sends `POST /like`. Both endpoints update the JSON stored inside the GitHub comment, so values are permanent and versioned in GitHub. A lightweight `GET /metrics?slug=...` keeps the UI in sync.
5. The client also stores a `localStorage` flag (`post:liked:slug`) to prevent accidental repeat likes from the same browser.
