/* Minimal blog SPA with hash routing, theme toggle, and Markdown+KaTeX support. */
const { useEffect, useMemo, useRef, useState } = React;

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
  try {
    syncGiscusTheme(theme);
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

const GISCUS_CONFIG = window.__BLOG_GISCUS || null;
const METRICS_CONFIG = window.__BLOG_METRICS || null;
const METRICS_BASE_URL = (METRICS_CONFIG && typeof METRICS_CONFIG.baseUrl === 'string') ? METRICS_CONFIG.baseUrl.trim() : '';
const METRICS_ENDPOINT = METRICS_BASE_URL ? METRICS_BASE_URL.replace(/\/+$/, '') : '';
const METRICS_ENABLED = METRICS_ENDPOINT.length > 0;

function createWidgetElement(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
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

const POST_WIDGET_RENDERERS = {
  'cuda-triton': renderCudaTritonWidget,
  'cuda-hello': renderCudaHelloWidget,
  'cuda-vector': renderCudaVectorWidget
};

function hydratePostWidgets(el) {
  if (!el) return;
  const widgets = el.querySelectorAll('[data-widget]');
  if (!widgets.length) return;
  widgets.forEach(node => {
    const type = node.dataset.widget;
    if (!POST_WIDGET_RENDERERS[type]) return;
    if (node.dataset.mounted === 'true') return;
    POST_WIDGET_RENDERERS[type](node);
    node.dataset.mounted = 'true';
  });
}

function renderCudaTritonWidget(root) {
  if (!root) return;

  const x = [1, 2, 3, 4, 5, 6];
  const y = [10, 20, 30, 40, 50, 60];
  const blockSize = 4;
  const total = x.length;
  const blocks = Math.ceil(total / blockSize);
  const z = x.map((val, idx) => val + y[idx]);

  root.classList.add('triton', 'triton-widget');
  root.innerHTML = '';

  const makeEl = (tag, className, text) => {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  };

  const makeArrayRow = (label, values, modifier) => {
    const row = makeEl('div', 'triton-array');
    row.appendChild(makeEl('span', 'triton-array-label', `${label}:`));
    const cells = makeEl('div', 'triton-array-cells');
    values.forEach((value, idx) => {
      const cell = makeEl('div', `triton-cell ${modifier}`);
      cell.textContent = value;
      cell.setAttribute('data-index', idx);
      cells.appendChild(cell);
    });
    row.appendChild(cells);
    return row;
  };

  const title = makeEl('h3', 'triton-title', 'Vector addition · size 6, block size 4');
  root.appendChild(title);
  root.appendChild(makeEl('p', 'triton-lead', 'A quick visual of how CUDA assigns scalar threads while Triton works with vector-sized programs.'));

  const inputs = makeEl('section', 'triton-card');
  inputs.appendChild(makeEl('h4', 'triton-subtitle', 'Input arrays'));
  inputs.appendChild(makeArrayRow('x', x, 'triton-cell--x'));
  inputs.appendChild(makeArrayRow('y', y, 'triton-cell--y'));
  root.appendChild(inputs);

  const cudaCard = makeEl('section', 'triton-card');
  cudaCard.appendChild(makeEl('h4', 'triton-subtitle', 'CUDA · scalar threads'));
  cudaCard.appendChild(makeEl('p', 'triton-note', 'Two blocks × four threads. Each thread guards a single scalar and falls back to an if-guard when it runs out of work.'));
  const cudaGrid = makeEl('div', 'triton-block-grid');
  for (let block = 0; block < blocks; block += 1) {
    const blockBox = makeEl('div', 'triton-block');
    blockBox.appendChild(makeEl('h5', 'triton-block-title', `Block ${block}`));
    const threadGrid = makeEl('div', 'triton-thread-grid');
    for (let thread = 0; thread < blockSize; thread += 1) {
      const globalIdx = block * blockSize + thread;
      const outOfBounds = globalIdx >= total;
      const threadBox = makeEl('div', `triton-thread${outOfBounds ? ' triton-thread--inactive' : ''}`);
      threadBox.appendChild(makeEl('div', 'triton-thread-title', `Thread ${thread}`));
      threadBox.appendChild(makeEl('div', 'triton-thread-detail', `global = ${globalIdx}`));
      const op = outOfBounds ? 'masked' : `z[${globalIdx}] = x[${globalIdx}] + y[${globalIdx}]`;
      threadBox.appendChild(makeEl('div', 'triton-thread-op', op));
      threadGrid.appendChild(threadBox);
    }
    blockBox.appendChild(threadGrid);
    cudaGrid.appendChild(blockBox);
  }
  cudaCard.appendChild(cudaGrid);
  cudaCard.appendChild(makeEl('p', 'triton-footnote', 'Out-of-bounds threads simply skip work thanks to scalar guards.'));
  root.appendChild(cudaCard);

  const tritonCard = makeEl('section', 'triton-card');
  tritonCard.appendChild(makeEl('h4', 'triton-subtitle', 'Triton · vector programs'));
  tritonCard.appendChild(makeEl('p', 'triton-note', 'Same grid, but each program operates on a vector of offsets and applies a vectorised mask.'));
  const programGrid = makeEl('div', 'triton-program-grid');
  for (let block = 0; block < blocks; block += 1) {
    const programBox = makeEl('div', 'triton-program');
    const start = block * blockSize;
    const indices = Array.from({ length: blockSize }, (_, idx) => start + idx);
    programBox.appendChild(makeEl('h5', 'triton-program-title', `Program ${block}`));

    const sliceRow = (label, values, modifier) => {
      const row = makeEl('div', 'triton-vector-row');
      row.appendChild(makeEl('span', 'triton-vector-label', `${label}[${start}:${start + blockSize}]`));
      const cells = makeEl('div', 'triton-array-cells');
      indices.forEach(idx => {
        const out = idx >= total;
        const cell = makeEl('div', `triton-cell ${modifier}${out ? ' triton-cell--masked' : ''}`);
        cell.textContent = out ? '—' : values[idx];
        cells.appendChild(cell);
      });
      row.appendChild(cells);
      return row;
    };

    programBox.appendChild(sliceRow('x', x, 'triton-cell--x'));
    programBox.appendChild(sliceRow('y', y, 'triton-cell--y'));

    const opRow = makeEl('div', 'triton-vector-row');
    opRow.appendChild(makeEl('span', 'triton-vector-label', 'mask'));
    const maskCells = makeEl('div', 'triton-array-cells');
    indices.forEach(idx => {
      const cell = makeEl('div', `triton-cell triton-cell--mask${idx >= total ? ' triton-cell--masked' : ''}`);
      cell.textContent = idx < total ? '✓' : '✗';
      maskCells.appendChild(cell);
    });
    opRow.appendChild(maskCells);
    programBox.appendChild(opRow);

    programBox.appendChild(makeEl('div', 'triton-program-note', 'Vector loads, adds, and stores honour the same mask.'));
    programGrid.appendChild(programBox);
  }
  tritonCard.appendChild(programGrid);
  tritonCard.appendChild(makeEl('p', 'triton-footnote', 'No explicit conditionals—masking keeps the math vectorised.'));
  root.appendChild(tritonCard);

  const outputCard = makeEl('section', 'triton-card');
  outputCard.appendChild(makeEl('h4', 'triton-subtitle', 'Output vector'));
  outputCard.appendChild(makeArrayRow('z', z, 'triton-cell--z'));
  root.appendChild(outputCard);

  const summaryCard = makeEl('section', 'triton-card');
  summaryCard.appendChild(makeEl('h4', 'triton-subtitle', 'Mental model recap'));
  const summaryGrid = makeEl('div', 'triton-summary-grid');

  const cudaSummary = makeEl('div', 'triton-summary');
  cudaSummary.appendChild(makeEl('h5', 'triton-summary-title', 'CUDA threads'));
  const cudaList = makeEl('ul', 'triton-list');
  ['Thread-level control', 'Scalar guards for boundaries', 'Manual shared-memory management'].forEach(item => {
    const li = makeEl('li');
    li.textContent = item;
    cudaList.appendChild(li);
  });
  cudaSummary.appendChild(cudaList);

  const tritonSummary = makeEl('div', 'triton-summary');
  tritonSummary.appendChild(makeEl('h5', 'triton-summary-title', 'Triton programs'));
  const tritonList = makeEl('ul', 'triton-list');
  ['Vector-first mindset', 'Mask arguments on load/store', 'Compiler handles scratch memory'].forEach(item => {
    const li = makeEl('li');
    li.textContent = item;
    tritonList.appendChild(li);
  });
  tritonSummary.appendChild(tritonList);

  summaryGrid.appendChild(cudaSummary);
  summaryGrid.appendChild(tritonSummary);
  summaryCard.appendChild(summaryGrid);
  root.appendChild(summaryCard);
}

function renderCudaHelloWidget(root) {
  if (!root) return;
  root.classList.add('diagram', 'diagram--hello');
  root.innerHTML = '';

  root.appendChild(createWidgetElement('h4', 'diagram-title', 'Blocks and threads at a glance'));
  root.appendChild(createWidgetElement('p', 'diagram-caption', 'Two blocks of four threads; think of rows and seats in a classroom.'));

  const grid = createWidgetElement('div', 'diagram-grid');
  root.appendChild(grid);

  const blocks = [
    { idx: 0, label: 'Block 0', threads: 4 },
    { idx: 1, label: 'Block 1', threads: 4 }
  ];

  blocks.forEach(block => {
    const blockEl = createWidgetElement('div', 'diagram-block');
    blockEl.appendChild(createWidgetElement('div', 'diagram-block-title', `${block.label} · blockIdx.x = ${block.idx}`));
    const threadWrap = createWidgetElement('div', 'diagram-threads');
    for (let t = 0; t < block.threads; t += 1) {
      const threadEl = createWidgetElement('div', 'diagram-thread');
      threadEl.appendChild(createWidgetElement('div', 'diagram-thread-id', `threadIdx.x = ${t}`));
      threadEl.appendChild(createWidgetElement('div', 'diagram-thread-note', `Prints: block ${block.idx}, thread ${t}`));
      threadWrap.appendChild(threadEl);
    }
    blockEl.appendChild(threadWrap);
    grid.appendChild(blockEl);
  });

  root.appendChild(createWidgetElement('p', 'diagram-footnote', 'GPU scheduling may shuffle the greeting order between blocks.'));
}

function renderCudaVectorWidget(root) {
  if (!root) return;
  root.classList.add('diagram', 'diagram--vector');
  root.innerHTML = '';

  root.appendChild(createWidgetElement('h4', 'diagram-title', 'Vector addition data flow'));
  root.appendChild(createWidgetElement('p', 'diagram-caption', 'Host arrays move to the device, threads add element pairs, and the result returns.'));

  const layout = createWidgetElement('div', 'diagram-layout');
  root.appendChild(layout);

  const hostCol = createWidgetElement('div', 'diagram-column diagram-column--host');
  hostCol.appendChild(createWidgetElement('div', 'diagram-column-title', 'CPU (Host)'));
  hostCol.appendChild(createWidgetElement('div', 'diagram-card', 'h_a = [0, 1, 2, …]'));
  hostCol.appendChild(createWidgetElement('div', 'diagram-card', 'h_b = [0, 2, 4, …]'));
  hostCol.appendChild(createWidgetElement('div', 'diagram-card', 'h_c = results buffer'));

  const flowCol = createWidgetElement('div', 'diagram-column diagram-column--flow');
  flowCol.appendChild(createWidgetElement('div', 'diagram-arrow diagram-arrow--down', 'cudaMemcpy →'));
  flowCol.appendChild(createWidgetElement('div', 'diagram-arrow diagram-arrow--kernel', 'vectorAdd<<<blocks, threads>>>'));
  flowCol.appendChild(createWidgetElement('div', 'diagram-arrow diagram-arrow--up', 'cudaMemcpy ←'));

  const deviceCol = createWidgetElement('div', 'diagram-column diagram-column--device');
  deviceCol.appendChild(createWidgetElement('div', 'diagram-column-title', 'GPU (Device)'));
  deviceCol.appendChild(createWidgetElement('div', 'diagram-card', 'd_a (device copy)'));
  deviceCol.appendChild(createWidgetElement('div', 'diagram-card', 'd_b (device copy)'));

  const kernelCard = createWidgetElement('div', 'diagram-card diagram-card--kernel');
  kernelCard.appendChild(createWidgetElement('div', 'diagram-card-heading', 'vectorAdd kernel'));
  kernelCard.appendChild(createWidgetElement('div', 'diagram-card-body', 'c[idx] = a[idx] + b[idx]'));
  deviceCol.appendChild(kernelCard);

  const threadMatrix = createWidgetElement('div', 'diagram-thread-matrix');
  ['idx 0', 'idx 1', '⋯', 'idx (n-1)'].forEach(label => {
    threadMatrix.appendChild(createWidgetElement('div', 'diagram-thread-cell', label));
  });
  threadMatrix.appendChild(createWidgetElement('div', 'diagram-thread-formula', 'idx = blockIdx.x · blockDim.x + threadIdx.x'));
  deviceCol.appendChild(threadMatrix);

  layout.appendChild(hostCol);
  layout.appendChild(flowCol);
  layout.appendChild(deviceCol);

  root.appendChild(createWidgetElement('p', 'diagram-footnote', 'Each GPU thread handles one position; boundary checks stop extra threads from reading past n.'));
}

function usePostMetrics(slug) {
  const enabled = METRICS_ENABLED && !!slug;
  const [counts, setCounts] = useState({ views: null, likes: null });
  const [pending, setPending] = useState(false);
  const [liked, setLiked] = useState(false);

  useEffect(() => {
    if (!enabled) {
      setCounts({ views: null, likes: null });
      setLiked(false);
      return undefined;
    }
    let cancelled = false;
    setCounts({ views: null, likes: null });
    const localKey = `post:liked:${slug}`;
    try {
      setLiked(localStorage.getItem(localKey) === '1');
    } catch (_) {}

    const fetchMetrics = async () => {
      try {
        const res = await fetch(`${METRICS_ENDPOINT}/metrics?slug=${encodeURIComponent(slug)}`, { credentials: 'omit' });
        if (res.ok) {
          const data = await res.json();
          if (!cancelled) {
            setCounts({
              views: typeof data.views === 'number' ? data.views : null,
              likes: typeof data.likes === 'number' ? data.likes : null
            });
          }
        }
      } catch (err) {
        console.warn('Metrics fetch failed', err);
      }
      try {
        const res = await fetch(`${METRICS_ENDPOINT}/view`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ slug })
        });
        if (res.ok) {
          const payload = await res.json();
          if (!cancelled && typeof payload.views === 'number') {
            setCounts(prev => ({ ...prev, views: payload.views }));
          }
        }
      } catch (err) {
        console.warn('View increment failed', err);
      }
    };

    fetchMetrics();
    return () => { cancelled = true; };
  }, [enabled, slug]);

  const onLike = async () => {
    if (!enabled || liked || pending) return;
    const localKey = `post:liked:${slug}`;
    setPending(true);
    try {
      const res = await fetch(`${METRICS_ENDPOINT}/like`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slug })
      });
      if (res.ok) {
        const payload = await res.json();
        setCounts(prev => ({ ...prev, likes: typeof payload.likes === 'number' ? payload.likes : prev.likes }));
        setLiked(true);
        try { localStorage.setItem(localKey, '1'); } catch (_) {}
      }
    } catch (err) {
      console.warn('Like failed', err);
    } finally {
      setPending(false);
    }
  };

  return { enabled, counts, liked, pending, like: onLike };
}

function Comments({ slug, theme }) {
  const containerRef = useRef(null);
  const config = GISCUS_CONFIG;

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !config || !config.repo || !config.repoId || !config.categoryId) return undefined;
    container.innerHTML = '';

    const script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.setAttribute('data-repo', config.repo);
    script.setAttribute('data-repo-id', config.repoId);
    script.setAttribute('data-category', config.category || '');
    script.setAttribute('data-category-id', config.categoryId);
    script.setAttribute('data-mapping', config.mapping || 'specific');
    if ((config.termStrategy || 'slug') === 'slug') {
      script.setAttribute('data-term', slug);
    } else if (config.mapping !== 'pathname') {
      script.setAttribute('data-term', `${config.termPrefix || ''}${slug}`);
    }
    script.setAttribute('data-strict', '1');
    script.setAttribute('data-reactions-enabled', config.reactions ?? '1');
    script.setAttribute('data-emit-metadata', config.emitMetadata ?? '0');
    script.setAttribute('data-input-position', config.inputPosition || 'top');
    script.setAttribute('data-lang', config.lang || 'en');
    const themeName = theme === 'dark' ? (config.themeDark || 'dark_dimmed') : (config.themeLight || 'light');
    script.setAttribute('data-theme', themeName);
    container.appendChild(script);

    return () => { container.innerHTML = ''; };
  }, [config, slug, theme]);

  if (!config || !config.repo || !config.repoId || !config.categoryId) {
    return null;
  }

  return React.createElement('section', { className: 'comments-block' },
    React.createElement('h3', { className: 'comments-title' }, 'Comments'),
    React.createElement('div', { ref: containerRef })
  );
}

function syncGiscusTheme(theme) {
  if (!GISCUS_CONFIG) return;
  const frame = document.querySelector('iframe.giscus-frame');
  if (!frame || !frame.contentWindow) return;
  const themeName = theme === 'dark' ? (GISCUS_CONFIG.themeDark || 'dark_dimmed') : (GISCUS_CONFIG.themeLight || 'light');
  frame.contentWindow.postMessage({ giscus: { setConfig: { theme: themeName } } }, 'https://giscus.app');
}

function PostView({ meta, theme }) {
  const [html, setHtml] = useState('<p>Loading…</p>');
  const [toc, setToc] = useState([]);
  const [activeId, setActiveId] = useState('');
  const metrics = usePostMetrics(meta.slug);
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
            hydratePostWidgets(el);
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
        metrics.enabled && React.createElement('div', { className: 'post-engagement' },
          React.createElement('button', {
            className: 'like-pill',
            type: 'button',
            onClick: metrics.like,
            disabled: metrics.liked || metrics.pending
          },
            metrics.liked ? 'Liked' : 'Like',
            typeof metrics.counts.likes === 'number' ? ` · ${metrics.counts.likes}` : ''
          ),
          typeof metrics.counts.views === 'number' && React.createElement('span', { className: 'views-pill' }, `${metrics.counts.views} views`)
        ),
        React.createElement('div', { id: 'post-content', className: 'content', dangerouslySetInnerHTML: { __html: html } }),
        React.createElement(Comments, { slug: meta.slug, theme })
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
            return meta ? React.createElement(PostView, { meta, theme }) : React.createElement(NotFound);
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
