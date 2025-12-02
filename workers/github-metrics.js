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

    // Track homepage visits
    if (request.method === 'POST' && url.pathname === '/homepage-visit') {
      const { userAgent, referrer, timestamp } = await parseBody(request);
      const store = await readStore(env);
      const homepageKey = 'homepage';

      if (!store.slugs[homepageKey]) {
        store.slugs[homepageKey] = { views: 0, visits: [], lastVisit: null };
      }

      store.slugs[homepageKey].views = (store.slugs[homepageKey].views || 0) + 1;
      store.slugs[homepageKey].visits.push({
        timestamp: timestamp || new Date().toISOString(),
        userAgent: userAgent || 'unknown',
        referrer: referrer || 'direct'
      });
      store.slugs[homepageKey].lastVisit = timestamp || new Date().toISOString();

      // Keep only last 100 visits to avoid storage bloat
      if (store.slugs[homepageKey].visits.length > 100) {
        store.slugs[homepageKey].visits = store.slugs[homepageKey].visits.slice(-100);
      }

      await persistStore(env, store);
      return json({ views: store.slugs[homepageKey].views }, 200);
    }

    // Track custom events
    if (request.method === 'POST' && url.pathname === '/event') {
      const { eventType, page, element, timestamp, metadata } = await parseBody(request);
      const store = await readStore(env);
      const eventsKey = 'events';

      if (!store.slugs[eventsKey]) {
        store.slugs[eventsKey] = [];
      }

      store.slugs[eventsKey].push({
        eventType,
        page: page || 'unknown',
        element: element || 'unknown',
        timestamp: timestamp || new Date().toISOString(),
        metadata: metadata || {}
      });

      // Keep only last 500 events
      if (store.slugs[eventsKey].length > 500) {
        store.slugs[eventsKey] = store.slugs[eventsKey].slice(-500);
      }

      await persistStore(env, store);
      return json({ success: true }, 200);
    }

    // Get analytics summary
    if (request.method === 'GET' && url.pathname === '/analytics-summary') {
      const store = await readStore(env);
      const homepage = store.slugs['homepage'] || { views: 0, visits: [] };
      const posts = Object.keys(store.slugs).filter(key => key !== 'homepage' && key !== 'events');
      const totalPostViews = posts.reduce((sum, slug) => sum + (store.slugs[slug]?.views || 0), 0);
      const totalEvents = store.slugs['events']?.length || 0;

      return json({
        homepage: {
          totalViews: homepage.views,
          lastVisit: homepage.lastVisit,
          recentVisits: homepage.visits?.slice(-10) || []
        },
        blog: {
          totalPostViews,
          postCount: posts.length,
          topPosts: posts
            .map(slug => ({ slug, views: store.slugs[slug]?.views || 0 }))
            .sort((a, b) => b.views - a.views)
            .slice(0, 5)
        },
        events: {
          totalEvents,
          recentEvents: store.slugs['events']?.slice(-20) || []
        }
      }, 200);
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
