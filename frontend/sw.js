/**
 * VetAI — Progressive Web App Service Worker (PWA v1.0)
 * Provides offline shell caching, stale-while-revalidate for veterinary reference libraries,
 * and seamless rural connectivity resilience.
 */

const CACHE_NAME = 'vetai-pwa-v2';

// Core shell and veterinary reference resources to pre-cache on install
const PRECACHE_URLS = [
  '/',
  '/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/icon.svg',
  '/api/symptoms',
  '/api/diseases',
  '/api/vaccines/standard'
];

// External CDN dependencies to cache
const CDN_URLS = [
  'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then(async (cache) => {
      // Pre-cache internal assets
      try {
        await cache.addAll(PRECACHE_URLS);
      } catch (err) {
        console.warn('[VetAI SW] Pre-cache partial failure:', err);
      }
      // Pre-cache CDN assets (best-effort, cors opaque)
      for (const url of CDN_URLS) {
        try {
          const req = new Request(url, { mode: 'no-cors' });
          const res = await fetch(req);
          if (res) await cache.put(req, res);
        } catch (e) {
          // ignore external fetch failures during install
        }
      }
    })
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // Handle non-GET requests (e.g. POST /api/predict, POST /api/chat)
  if (req.method !== 'GET') {
    event.respondWith(
      fetch(req).catch(() => {
        return new Response(
          JSON.stringify({
            offline: true,
            detail: "ऑफ़लाइन मोड: सर्वर से संपर्क नहीं हो सका। कृपया जांचें कि बैकएंड सर्वर सक्रिय है। (Offline: Could not connect to backend server.)",
            message: "ऑफ़लाइन मोड: सर्वर से संपर्क नहीं हो सका। कृपया जांचें कि बैकएंड सर्वर सक्रिय है। (Offline: Could not connect to backend server.)"
          }),
          {
            status: 503,
            statusText: "Service Unavailable (Offline)",
            headers: { 'Content-Type': 'application/json' }
          }
        );
      })
    );
    return;
  }

  // 1. Navigation requests (HTML pages) -> Network-first with cache fallback to root '/'
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req)
        .then((networkRes) => {
          if (networkRes && networkRes.status === 200) {
            const resClone = networkRes.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(req, resClone));
          }
          return networkRes;
        })
        .catch(() => {
          return caches.match('/') || caches.match(req);
        })
    );
    return;
  }

  // 2. Reference APIs (/api/symptoms, /api/diseases, /api/vaccines/standard)
  // Stale-While-Revalidate: Serve immediate cache, update in background
  if (url.pathname.startsWith('/api/symptoms') ||
    url.pathname.startsWith('/api/diseases') ||
    url.pathname.startsWith('/api/vaccines/standard')) {
    event.respondWith(
      caches.open(CACHE_NAME).then((cache) => {
        return cache.match(req).then((cachedRes) => {
          const fetchPromise = fetch(req)
            .then((networkRes) => {
              if (networkRes && networkRes.status === 200) {
                cache.put(req, networkRes.clone());
              }
              return networkRes;
            })
            .catch(() => cachedRes); // fallback to cached if fetch fails

          return cachedRes || fetchPromise;
        });
      })
    );
    return;
  }

  // Dynamic API requests (/api/animals, /api/dashboard, /api/me, health, etc.)
  // Must ALWAYS be network-only and never cached or intercepted by static cache-first!
  if (url.pathname.startsWith('/api/')) {
    return;
  }

  // 3. Static assets, fonts, icons, CSS, JS -> Cache-first with network fallback
  event.respondWith(
    caches.match(req).then((cachedRes) => {
      if (cachedRes) return cachedRes;
      return fetch(req).then((networkRes) => {
        if (networkRes && (networkRes.status === 200 || networkRes.type === 'opaque')) {
          const resClone = networkRes.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, resClone));
        }
        return networkRes;
      }).catch((err) => {
        // If resource is unavailable, return empty or fallback
        return null;
      });
    })
  );
});
