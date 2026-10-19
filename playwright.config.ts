import { defineConfig } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

/**
 * End-to-end configuration for the pure-static DirMap build.
 *
 * There is no API server any more: every project builds the Vite application
 * and serves the generated files with `tools/static_server.py`, exactly like
 * GitHub Pages would serve them.
 *
 * - `root`    builds `web/dist` with base `/` and serves it on port 4173.
 * - `subpath` builds `web/dist-sub` with `DIRMAP_BASE=/DIRMAP/` and serves it
 *             under `/DIRMAP/` on port 4174 (the GitHub project-page shape).
 *
 * All artifacts stay inside the workspace (`test-results/`, `playwright-report/`);
 * `/tmp` is an ephemeral per-command sandbox and must never receive state.
 */

/** Workspace root; `npm run test:browser` always runs from there. */
const ROOT = process.cwd();

/**
 * Locate a Chromium binary from the installed Playwright browser cache.
 *
 * The installed `@playwright/test` and the browser cache can drift apart (this
 * machine has `chromium-1228`, while a newer Playwright would download a newer
 * revision). Resolving the cached binary explicitly keeps the suite runnable
 * offline. `DIRMAP_BROWSER_PATH` always wins and remains the documented escape
 * hatch; when nothing is found we leave `undefined` so Playwright's own
 * resolution error explains the problem.
 */
function chromiumExecutable(): string | undefined {
  const override = process.env.DIRMAP_BROWSER_PATH;
  if (override) return override;
  const cache = process.env.PLAYWRIGHT_BROWSERS_PATH || path.join(os.homedir(), '.cache', 'ms-playwright');
  const layouts: string[][] = [
    ['chrome-linux64', 'chrome'],
    ['chrome-linux', 'chrome'],
    ['chrome-headless-shell-linux64', 'chrome-headless-shell'],
    ['chrome-mac', 'Chromium.app', 'Contents', 'MacOS', 'Chromium'],
    ['chrome-win', 'chrome.exe'],
  ];
  let entries: string[] = [];
  try { entries = fs.readdirSync(cache); } catch { entries = []; }
  const fullBrowsers = entries.filter(name => /^chromium-\d+$/.test(name)).sort().reverse();
  const headlessShells = entries.filter(name => /^chromium_headless_shell-\d+$/.test(name)).sort().reverse();
  for (const entry of [...fullBrowsers, ...headlessShells]) {
    for (const layout of layouts) {
      const candidate = path.join(cache, entry, ...layout);
      if (fs.existsSync(candidate)) return candidate;
    }
  }
  // Last resort: a system Chrome (this box ships Google Chrome 155, which runs
  // the suite headless). Playwright's own revision-managed resolution is skipped
  // because the pinned @playwright/test revision may not be downloaded here.
  for (const candidate of ['/usr/bin/google-chrome', '/opt/google/chrome/chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser']) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return undefined;
}

const ROOT_SERVER = 'http://127.0.0.1:4173/';
const SUBPATH_SERVER = 'http://127.0.0.1:4174/DIRMAP/';

export default defineConfig({
  testDir: './tests/browser',
  // Relative paths are resolved against this config file, i.e. the workspace root.
  outputDir: 'test-results',
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: [
    ['list'],
    ['html', { outputFolder: path.join(ROOT, 'playwright-report'), open: 'never' }],
  ],
  use: {
    headless: true,
    viewport: { width: 1440, height: 980 },
    serviceWorkers: 'allow',
    // Publishing a course triggers a release JSON download; accept it silently.
    acceptDownloads: true,
    // Traces are disabled on purpose: with 12 heavy WebAssembly/OCR tests the
    // trace artifact writer raced with context teardown (ENOENT on close) and
    // flaked once. Failure screenshots are enough for this suite's diagnostics.
    trace: 'off',
    screenshot: 'only-on-failure',
    video: 'off',
    launchOptions: {
      executablePath: chromiumExecutable(),
      args: ['--disable-dev-shm-usage'],
    },
  },
  projects: [
    {
      name: 'root',
      use: { baseURL: ROOT_SERVER },
      testIgnore: /basepath\.spec\.ts$/,
    },
    {
      name: 'subpath',
      use: { baseURL: SUBPATH_SERVER },
      testMatch: /basepath\.spec\.ts$/,
    },
  ],
  // Both static deployments are built once and then served by the dependency-free
  // verification server. Playwright starts them before the first test and kills
  // them afterwards, so no stray processes are left behind.
  webServer: [
    {
      // Root deployment: `tsc --noEmit` + `vite build` into web/dist, served at /.
      command: 'npm run build && python3 -B tools/static_server.py --dir web/dist --base / --port 4173 --quiet',
      url: ROOT_SERVER,
      reuseExistingServer: false,
      timeout: 180_000,
      stdout: 'pipe',
      stderr: 'pipe',
    },
    {
      // GitHub-project-page deployment: same application built with base /DIRMAP/.
      command: 'DIRMAP_BASE=/DIRMAP/ npx vite build --config web/vite.config.ts --outDir dist-sub && python3 -B tools/static_server.py --dir web/dist-sub --base /DIRMAP/ --port 4174 --quiet',
      url: SUBPATH_SERVER,
      reuseExistingServer: false,
      timeout: 180_000,
      stdout: 'pipe',
      stderr: 'pipe',
    },
  ],
});
