/* Minimal blog SPA with hash routing, theme toggle, and Markdown+KaTeX support. */
const { useEffect, useMemo, useState } = React;

// --- Utilities ---
const THEME_KEY = "blog:theme";
const prefersDark = () => window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

function getInitialTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === 'light' || saved === 'dark') return saved;
  return prefersDark() ? 'dark' : 'light';
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(THEME_KEY, theme);
  try {
    const light = document.getElementById('hljs-light');
    const dark = document.getElementById('hljs-dark');
    if (light && dark) { light.disabled = (theme !== 'light'); dark.disabled = (theme !== 'dark'); }
  } catch (_) {}
}

function formatDate(iso) {
  try { return new Date(iso).toLocaleDateString(undefined, { year:'numeric', month:'short', day:'2-digit' }); }
  catch { return iso; }
}

// --- Post manifest ---
// Keep a simple static manifest to avoid directory listing
const POSTS_INDEX_URL = 'posts/index.json';

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.json();
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.text();
}

// Configure marked to use highlight.js if available
if (window.marked) {
  const opts = {
    mangle: false,
    headerIds: true,
    langPrefix: 'hljs language-',
  };
  if (window.hljs) {
    opts.highlight = function(code, lang) {
      try {
        if (lang && window.hljs.getLanguage(lang)) {
          return window.hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
        }
        return window.hljs.highlightAuto(code).value;
      } catch (_) {
        return code;
      }
    };
  }
  window.marked.setOptions(opts);
}

// --- Components ---
function Header({ theme, onToggle }) {
  return (
    React.createElement('header', { className: 'site backdrop-blur' },
      React.createElement('div', { className: 'site-inner max-w-5xl mx-auto flex items-center justify-between gap-4 px-4 py-3 sm:px-6' },
        React.createElement('a', { className: 'brand flex items-center gap-3 no-underline', href: '#/' },
          React.createElement('span', { className: 'dot w-2.5 h-2.5 rounded-full', 'aria-hidden': true }),
          React.createElement('h1', { className: 'truncate' }, 'CudaSlayer — Notes')
        ),
        React.createElement('div', { className: 'toolbar' },
          React.createElement('button', { className: 'kbd rounded-full border px-3 py-1 text-sm', onClick: onToggle, title: 'Toggle theme' }, theme === 'dark' ? '☾ Dark' : '☀ Light')
        )
      )
    )
  );
}

function Footer() {
  return React.createElement('footer', { className: 'site' },
    '© ', new Date().getFullYear(), ' • ',
    React.createElement('a', { href: 'https://github.com/cudaslayer', target: '_blank', rel: 'noreferrer' }, 'cudaslayer'),
    ' • Built with React + Markdown + KaTeX'
  );
}

function PostList({ posts }) {
  return React.createElement('div', { className: 'post-list grid gap-3' },
    posts.map(p => (
      React.createElement('article', { key: p.slug, className: 'post-item card p-4 rounded-lg' },
        React.createElement('h2', null, React.createElement('a', { href: `#/post/${p.slug}` }, p.title)),
        React.createElement('div', { className: 'meta' }, `${formatDate(p.date)} · ${p.tags?.join(', ') || ''}`),
        p.summary ? React.createElement('p', null, p.summary) : null
      )
    ))
  );
}

function useHashRoute() {
  const [hash, setHash] = useState(window.location.hash || '#/');
  useEffect(() => {
    const onHash = () => setHash(window.location.hash || '#/');
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  return hash;
}

function renderMathIn(el) {
  if (!window.renderMathInElement) return; // KaTeX auto-render may not be loaded
  try {
    window.renderMathInElement(el, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\(', right: '\\)', display: false },
        { left: '\\[', right: '\\]', display: true }
      ],
      throwOnError: false,
      strict: false,
      processEnvironments: true,
      ignoredTags: ['script','noscript','style','textarea','pre','code']
    });
  } catch (e) {
    console.warn('Math render error:', e);
  }
}

function PostView({ meta }) {
  const [html, setHtml] = useState('<p>Loading…</p>');
  const [toc, setToc] = useState([]);
  const [activeId, setActiveId] = useState('');
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const md = await fetchText(meta.file);
        const content = marked.parse(md);
        if (!active) return;
        setHtml(content);
        // render math + ensure code highlighting after DOM updates
        setTimeout(() => {
          const el = document.querySelector('#post-content');
          if (el) {
            renderMathIn(el);
            try {
              if (window.hljs) {
                // Broad pass first
                if (window.hljs.highlightAll) window.hljs.highlightAll();
                el.querySelectorAll('pre code').forEach((block) => {
                  // If marked didn't already highlight, do it now
                  if (!block.classList.contains('hljs')) {
                    window.hljs.highlightElement(block);
                    block.classList.add('hljs');
                  }
                });
              }
            } catch (_) {}
            // Build table of contents from headings
            try {
              const headings = Array.from(el.querySelectorAll('h1, h2, h3'));
              const slug = (s) => s.toLowerCase().trim()
                .replace(/[^a-z0-9\s-]/g, '')
                .replace(/\s+/g, '-')
                .replace(/-+/g, '-');
              const items = [];
              headings.forEach(h => {
                const level = h.tagName.toLowerCase();
                const text = h.textContent || '';
                if (!h.id) h.id = slug(text);
                if (level === 'h2' || level === 'h3') items.push({ id: h.id, text, level });
              });
              setToc(items);
              if (items.length && !activeId) setActiveId(items[0].id);
            } catch (_) {}
          }
        }, 0);
      } catch (e) {
        setHtml(`<p style="color:var(--danger)">Failed to load post: ${e.message}</p>`);
      }
    })();
    return () => { active = false; };
  }, [meta.file]);

  return (
    React.createElement('div', { className: 'post-layout grid gap-6 md:grid-cols-12' },
      React.createElement('aside', { className: 'toc hidden md:block md:col-span-3 lg:col-span-3' },
        React.createElement('div', { className: 'toc-inner' },
          React.createElement('div', { className: 'toc-title' }, 'Contents'),
          toc.length === 0 ? React.createElement('div', { className: 'toc-empty' }, '—') :
          React.createElement('nav', null,
            React.createElement('ul', { className: 'toc-list' },
              toc.map(item => (
                React.createElement('li', {
                  key: item.id,
                  className: 'toc-li ' + (item.level === 'h3' ? 'toc-depth-2' : 'toc-depth-1')
                },
                  React.createElement('a', {
                    className: 'toc-link ' + (item.level === 'h3' ? 'toc-depth-2' : 'toc-depth-1') + (activeId === item.id ? ' active' : ''),
                    href: `#/post/${meta.slug}`,
                    onClick: (e) => {
                      e.preventDefault();
                      const section = document.getElementById(item.id);
                      if (section) {
                        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        setActiveId(item.id);
                      }
                    }
                  }, item.text.replace(/^\s*\d+[\).\:-]?\s*/, ''))
                )
              ))
            )
          )
        )
      ),
      React.createElement('article', { className: 'post md:col-span-9' },
        React.createElement('h1', null, meta.title),
        React.createElement('div', { className: 'meta' }, `${formatDate(meta.date)} · ${meta.tags?.join(', ') || ''}`),
        React.createElement('div', { id: 'post-content', className: 'content', dangerouslySetInnerHTML: { __html: html } })
      )
    )
  );
}

function NotFound() {
  return React.createElement('div', { className: 'card' },
    React.createElement('h2', null, 'Not Found'),
    React.createElement('p', null, 'The page you are looking for does not exist.'),
    React.createElement('p', null, React.createElement('a', { href: '#/' }, 'Go home'))
  );
}

function App() {
  const [theme, setThemeState] = useState(getInitialTheme());
  const [posts, setPosts] = useState([]);
  const [ready, setReady] = useState(false);
  const route = useHashRoute();

  useEffect(() => { setTheme(theme); }, [theme]);

  useEffect(() => {
    (async () => {
      try {
        const index = await fetchJSON(POSTS_INDEX_URL);
        // sort by date desc
        index.sort((a, b) => new Date(b.date) - new Date(a.date));
        setPosts(index);
      } catch (e) {
        console.warn('Failed to load posts index', e);
      } finally {
        setReady(true);
      }
    })();
  }, []);

  const view = useMemo(() => {
    const parts = (route || '#/').slice(1).split('/').filter(Boolean);
    if (parts.length === 0) return { name: 'home' };
    if (parts[0] === 'post' && parts[1]) return { name: 'post', slug: parts[1] };
    return { name: '404' };
  }, [route]);

  const onToggleTheme = () => setThemeState(t => (t === 'dark' ? 'light' : 'dark'));

  return (
    React.createElement(React.Fragment, null,
      React.createElement(Header, { theme, onToggle: onToggleTheme }),
      React.createElement('main', { className: 'px-4 sm:px-6' },
        React.createElement('div', { className: 'container max-w-5xl mx-auto' },
          !ready && React.createElement('div', { className: 'card' }, 'Loading…'),
          ready && view.name === 'home' && React.createElement(PostList, { posts }),
          ready && view.name === 'post' && (() => {
            const meta = posts.find(p => p.slug === view.slug);
            return meta ? React.createElement(PostView, { meta }) : React.createElement(NotFound);
          })(),
          ready && view.name === '404' && React.createElement(NotFound)
        )
      ),
      React.createElement(Footer)
    )
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(React.createElement(App));
