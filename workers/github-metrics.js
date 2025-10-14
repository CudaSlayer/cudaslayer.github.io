export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: corsHeaders()
      });
    }

    if (!env.GITHUB_TOKEN || !env.GITHUB_OWNER || !env.GITHUB_REPO || !env.GITHUB_COMMENT_ID) {
      return json({ error: 'Worker is missing GitHub configuration.' }, 500);
    }

    try {
      if (request.method === 'GET' && url.pathname === '/metrics') {
        const slug = url.searchParams.get('slug');
        if (!slug) return json({ error: 'Missing slug' }, 400);
        const store = await readStore(env);
        const entry = store.slugs[slug] || { views: 0, likes: 0 };
        return json(entry, 200);
      }

      if (request.method === 'POST' && url.pathname === '/view') {
        const { slug } = await parseBody(request);
        if (!slug) return json({ error: 'Missing slug' }, 400);
        const store = await readStore(env);
        const entry = store.slugs[slug] || { views: 0, likes: 0 };
        entry.views = Number(entry.views || 0) + 1;
        store.slugs[slug] = entry;
        await persistStore(env, store);
        return json({ views: entry.views }, 200);
      }

      if (request.method === 'POST' && url.pathname === '/like') {
        const { slug } = await parseBody(request);
        if (!slug) return json({ error: 'Missing slug' }, 400);
        const store = await readStore(env);
        const entry = store.slugs[slug] || { views: 0, likes: 0 };
        entry.likes = Number(entry.likes || 0) + 1;
        store.slugs[slug] = entry;
        await persistStore(env, store);
        return json({ likes: entry.likes }, 200);
      }

      return json({ error: 'Not found' }, 404);
    } catch (err) {
      console.error(err);
      return json({ error: err.message || 'Unexpected error' }, 500);
    }
  }
};

async function parseBody(request) {
  try {
    const data = await request.json();
    return typeof data === 'object' && data ? data : {};
  } catch (_) {
    return {};
  }
}

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  };
}

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders()
    }
  });
}

async function readStore(env) {
  const { GITHUB_OWNER, GITHUB_REPO, GITHUB_COMMENT_ID, GITHUB_TOKEN } = env;
  const res = await fetch(`https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/issues/comments/${GITHUB_COMMENT_ID}`, {
    headers: githubHeaders(GITHUB_TOKEN)
  });
  if (!res.ok) {
    throw new Error(`Failed to read GitHub comment: ${res.status}`);
  }
  const data = await res.json();
  if (!data || typeof data.body !== 'string') {
    return { slugs: {} };
  }
  try {
    const parsed = JSON.parse(data.body);
    if (parsed && typeof parsed === 'object' && parsed.slugs) {
      return { slugs: parsed.slugs };
    }
    return { slugs: parsed?.slugs || {} };
  } catch (_) {
    return { slugs: {} };
  }
}

async function persistStore(env, store) {
  const { GITHUB_OWNER, GITHUB_REPO, GITHUB_COMMENT_ID, GITHUB_TOKEN } = env;
  const body = JSON.stringify({ slugs: store.slugs }, null, 2);
  const res = await fetch(`https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/issues/comments/${GITHUB_COMMENT_ID}`, {
    method: 'PATCH',
    headers: {
      ...githubHeaders(GITHUB_TOKEN),
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ body })
  });
  if (!res.ok) {
    throw new Error(`Failed to persist metrics: ${res.status}`);
  }
}

function githubHeaders(token) {
  return {
    Authorization: `Bearer ${token}`,
    'User-Agent': 'cudaslayer-blog-metrics',
    Accept: 'application/vnd.github+json'
  };
}
