import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const schemas = JSON.parse(readFileSync(new URL('../dirmap/schemas.json', import.meta.url), 'utf8'));

/** Normalise the deployment base so dev, preview and build all agree.
 *
 * GitHub Pages serves project sites from `https://<user>.github.io/<repo>/`, so
 * CI builds with `DIRMAP_BASE=/<repo>/`; user sites / custom domains keep `/`.
 * A missing leading/trailing slash is fixed up so `import.meta.env.BASE_URL`
 * always ends in `/` (the app builds every data URL from it).
 */
function normalizeBase(raw: string): string {
  const value = raw.trim() || '/';
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(value)) return value.endsWith('/') ? value : `${value}/`;
  const path = value.replace(/^\/+/, '').replace(/\/+$/, '');
  return path ? `/${path}/` : '/';
}

const base = normalizeBase(process.env.DIRMAP_BASE || '/');

/** Short commit that produced this build; shown in the in-app diagnostics. */
const commit = (() => {
  try { return execSync('git rev-parse --short HEAD', { cwd: fileURLToPath(new URL('..', import.meta.url)) }).toString().trim(); }
  catch { return 'dev'; }
})();

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  base,
  define: {
    __DIRMAP_ROUTING_SCHEMA__: JSON.stringify(schemas.routing),
    __DIRMAP_COMMIT__: JSON.stringify(commit),
  },
  plugins: [{
    // The build inserts a content fingerprint so every new application refreshes
    // its offline shell, even when the service worker's logic itself has not
    // changed. `web/sw.js` is a source template containing the placeholder; the
    // emitted `dist/sw.js` gets the real id. Keep this plugin working with the
    // GitHub Pages base path: the worker derives its scope at runtime.
    name: 'versioned-offline-shell', apply: 'build', enforce: 'post',
    generateBundle(_options, bundle) {
      const source = readFileSync(new URL('./sw.js', import.meta.url), 'utf8');
      const hash = createHash('sha256').update(source);
      for (const name of Object.keys(bundle).sort()) {
        const asset = bundle[name];
        hash.update(name).update(asset.type === 'chunk' ? asset.code : asset.source);
      }
      this.emitFile({ type: 'asset', fileName: 'sw.js',
        source: source.replace('__DIRMAP_BUILD_ID__', hash.digest('hex').slice(0, 16)) });
    },
  }],
  // Development is purely static as well: all data lives under `web/public/data`.
  server: { host: '0.0.0.0', port: 5173 },
  preview: { host: '127.0.0.1', port: 4173 },
  build: { outDir: 'dist', emptyOutDir: true, target: 'es2022' },
});
