/* DirMap static service worker.
 *
 * The application is deployed as plain files, either at the site root
 * (`https://user.github.io/`) or at a repository sub-path
 * (`https://user.github.io/<repo>/`). Every application URL is therefore derived
 * from the registration scope at runtime instead of being hard-coded.
 *
 * The worker keeps the application shell (index.html + its hashed JS/CSS) plus
 * the data of at most three saved courses: the static release JSON, its map
 * overlay, the map state (`data/maps/<assetId>/state.json`) and the shared
 * health/reference documents. Graph bundles stay network-only. OpenStreetMap
 * tiles are cross-origin and are intentionally neither prefetched nor
 * intercepted.
 */
// The build replaces this placeholder with a content fingerprint so every new
// application refreshes its offline shell, even if the worker logic is unchanged.
const SHELL = 'dirmap-shell-__DIRMAP_BUILD_ID__';
const COURSES = 'dirmap-courses-v1';

/** Deployment path (`/` or `/<repo>/`), always ending in a slash. */
const BASE = (() => {
  const scope = self.registration && self.registration.scope;
  if (scope) return new URL(scope).pathname.replace(/\/?$/, '/');
  // Defensive fallback: the worker always lives in the deployment root.
  return self.location.pathname.replace(/[^/]*$/, '') || '/';
})();

/** Application URL from a path relative to the base. */
function appUrl(path) {
  return `${BASE}${String(path).replace(/^\/+/, '')}`;
}

const escapeRegExp = text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const BASE_RE = escapeRegExp(BASE);
const RELEASE = new RegExp(`^${BASE_RE}data/releases/([0-9a-f]{32})\\.json$`);
const OVERLAY = new RegExp(`^${BASE_RE}data/maps/([^/]+)/overlay\\.png$`);
const STATE_JSON = new RegExp(`^${BASE_RE}data/maps/([^/]+)/state\\.json$`);
const HASHED_ASSET = new RegExp(`^${BASE_RE}assets/`);
const SHELL_INDEX = appUrl('index.html');
const HEALTHISH = new Set([appUrl('data/health.json'), appUrl('data/reference.json'), appUrl('data/releases.json')]);
const PRECACHE_SHARED = [appUrl('data/health.json'), appUrl('data/reference.json')];

self.addEventListener('install', event => event.waitUntil((async () => {
  const response = await fetch(SHELL_INDEX, { cache: 'reload' });
  if (!response.ok) throw new Error('Cannot download application shell');
  const html = await response.clone().text();
  const assets = [...new Set([...html.matchAll(/(?:src|href)=["']([^"']+\.(?:js|css))["']/g)]
    .map(match => new URL(match[1], response.url).pathname)
    .filter(path => HASHED_ASSET.test(path)))];
  const cache = await caches.open(SHELL);
  await cache.put(SHELL_INDEX, response);
  await cache.addAll(assets);
  await self.skipWaiting();
})()));

self.addEventListener('activate', event => event.waitUntil((async () => {
  for (const name of await caches.keys()) {
    if ((name.startsWith('dirmap-shell-') || name.startsWith('dirmap-courses-')) && name !== SHELL && name !== COURSES) await caches.delete(name);
  }
  await self.clients.claim();
})()));

async function limitedFetch(input, options) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4000);
  try { return await fetch(input, { ...options, signal: controller.signal }); }
  finally { clearTimeout(timer); }
}

async function saveJSON(cache, request, response) {
  const headers = new Headers(response.headers);
  headers.set('X-DirMap-Saved-At', new Date().toISOString());
  await cache.put(request, new Response(await response.clone().arrayBuffer(), { status: response.status, headers }));
}

/** Decode a URL path segment so it can be compared with an `assetId` from JSON. */
function decodePathPart(value) {
  try { return decodeURIComponent(value); } catch { return value; }
}

/** Path (query stripped) of an application data file such as a release overlay. */
function pathOf(url) {
  return new URL(url, self.location.origin + BASE).pathname;
}

/** Keep at most three saved courses and remove overlays no kept course needs. */
async function trimCourses(cache) {
  const entries = [];
  for (const request of await cache.keys()) {
    const match = new URL(request.url).pathname.match(RELEASE);
    if (!match) continue;
    const response = await cache.match(request);
    if (!response) continue; // cache changed between keys() and match() (concurrent trim)
    let overlay = null, assetId = null;
    try {
      const release = await response.json();
      if (release && typeof release.overlay === 'string') overlay = release.overlay;
      if (release && typeof release.assetId === 'string') assetId = release.assetId;
      else if (overlay) {
        const fromOverlay = overlay.match(/data\/maps\/([^/]+)\//);
        if (fromOverlay) assetId = decodePathPart(fromOverlay[1]);
      }
      if (!overlay && assetId) overlay = `data/maps/${assetId}/overlay.png`;
    } catch { /* keep the course even if its body is unreadable */ }
    entries.push({ id: match[1], request, saved: response.headers.get('X-DirMap-Saved-At') || '', overlay, assetId });
  }
  entries.sort((a, b) => b.saved.localeCompare(a.saved));
  const kept = entries.slice(0, 3);
  for (const entry of entries.slice(3)) await cache.delete(entry.request);
  const keptOverlays = new Set(), keptAssets = new Set();
  for (const entry of kept) {
    if (entry.overlay) keptOverlays.add(pathOf(entry.overlay));
    if (entry.assetId) keptAssets.add(entry.assetId);
  }
  for (const request of await cache.keys()) {
    const path = new URL(request.url).pathname;
    if (OVERLAY.test(path) && !keptOverlays.has(path)) await cache.delete(request);
    const stateMatch = path.match(STATE_JSON);
    if (stateMatch && !keptAssets.has(decodePathPart(stateMatch[1]))) await cache.delete(request);
  }
}

/** Network-first JSON with a cached copy flagged as offline when the network fails. */
async function freshJSON(request) {
  const cache = await caches.open(COURSES);
  try {
    const response = await limitedFetch(request);
    // An explicit 404 must not resurrect a release that was removed from the site.
    if (response.ok) { await saveJSON(cache, request, response); await trimCourses(cache); }
    return response;
  } catch (error) {
    const cached = await cache.match(request);
    if (!cached) throw error;
    return Response.json({ ...await cached.json(), offline: true, savedAt: cached.headers.get('X-DirMap-Saved-At') },
      { headers: { 'X-DirMap-Offline': '1' } });
  }
}

async function cachedShellIndex() {
  const cached = await (await caches.open(SHELL)).match(SHELL_INDEX);
  if (cached) return cached;
  return new Response('离线，且本机尚未缓存应用外壳。', { status: 503, headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
}

self.addEventListener('fetch', event => {
  const request = event.request, url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;
  if (request.mode === 'navigate') {
    // Same-origin navigation: always online-fresh when possible, never fail offline.
    event.respondWith(limitedFetch(request).catch(cachedShellIndex));
  } else if (RELEASE.test(url.pathname) || HEALTHISH.has(url.pathname) || STATE_JSON.test(url.pathname)) {
    event.respondWith(freshJSON(request));
  } else if (OVERLAY.test(url.pathname)) {
    event.respondWith((async () => {
      const cache = await caches.open(COURSES);
      const key = url.pathname; // ignore cache-busting query strings (`?v=...`)
      const cached = await cache.match(key);
      if (cached) return cached;
      const response = await fetch(request);
      if (response.ok) { await cache.put(key, response.clone()); await trimCourses(cache); }
      return response;
    })());
  } else if (HASHED_ASSET.test(url.pathname)) {
    // Vite emits content-hashed files: once the shell is installed they never change.
    event.respondWith((async () => {
      const cache = await caches.open(SHELL);
      return await cache.match(request) || fetch(request);
    })());
  }
  // Everything else is left to the network and the browser HTTP cache. In
  // particular map scans and the (large) graph files
  // `data/maps/<id>/graph.json` + `data/maps/<id>/graph.bin.gz` are NOT
  // intercepted or precached here, so a saved course never bloats the cache.
  // Only the shell and the three saved-course packages are stored.
});

self.addEventListener('message', event => {
  if (event.data?.type !== 'CACHE_COURSE' || !/^[0-9a-f]{32}$/.test(String(event.data.id))) return;
  event.waitUntil((async () => {
    let reply;
    try {
      const id = event.data.id, cache = await caches.open(COURSES);
      const releasePath = appUrl(`data/releases/${id}.json`);
      const course = await limitedFetch(releasePath, { cache: 'no-store' });
      if (!course.ok) throw new Error('课程已更新或不可用');
      const release = await course.clone().json();
      if (!release || release.superseded) throw new Error('课程已更新或不可用');
      const overlayPath = release.overlay
        ? pathOf(release.overlay)
        : appUrl(`data/maps/${encodeURIComponent(release.assetId || '')}/overlay.png`);
      const overlay = await limitedFetch(overlayPath);
      if (!overlay.ok) throw new Error('原图尚未下载完整');
      await cache.put(overlayPath, overlay);
      const fromOverlay = overlayPath.match(/data\/maps\/([^/]+)\//);
      const assetId = (typeof release.assetId === 'string' && release.assetId)
        || (fromOverlay && decodePathPart(fromOverlay[1]));
      if (!assetId) throw new Error('课程缺少地图标识');
      const statePath = appUrl(`data/maps/${encodeURIComponent(assetId)}/state.json`);
      const mapState = await limitedFetch(statePath);
      if (!mapState.ok) throw new Error('地图状态尚未下载完整');
      await saveJSON(cache, statePath, mapState);
      await saveJSON(cache, releasePath, course);
      for (const path of PRECACHE_SHARED) {
        try { const response = await limitedFetch(path); if (response.ok) await saveJSON(cache, path, response); } catch { /* overlay and routes are already complete */ }
      }
      await trimCourses(cache);
      reply = { ok: true };
    } catch (error) {
      reply = { ok: false, message: error instanceof Error ? error.message : String(error) };
    }
    event.ports[0]?.postMessage(reply);
  })());
});
