/** Shared helpers for the static-build Playwright suite.
 *
 * There is no backend any more: the tests read the committed baked data with
 * Node's `fs` (for golden expectations) and drive the real static application in
 * Chromium. Every helper here keeps the tests independent: each test gets a
 * fresh browser context, so localStorage and service workers never leak.
 */
import { expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

/** Workspace root (Playwright is started from there by `npm run test:browser`). */
export const ROOT = process.cwd();

/** Read and parse a workspace-relative JSON file (baked data / golden fixture). */
export function readWorkspaceJSON<T = unknown>(relativePath: string): T {
  const absolute = path.resolve(ROOT, relativePath);
  try {
    return JSON.parse(fs.readFileSync(absolute, 'utf8')) as T;
  } catch (error) {
    throw new Error(`Cannot read workspace fixture ${relativePath} (${absolute}): ${error instanceof Error ? error.message : String(error)}. Run \`npm run bake\` if web/public/data is missing.`);
  }
}

export interface ReleaseSummary {
  id: string;
  title: string;
  assetId: string;
  createdAt: number;
}

export interface StaticReleaseFixture {
  id: string;
  title: string;
  assetId: string;
  version: number;
  superseded: boolean;
  static: boolean;
  routingSchema: string;
  overlay: string;
  frame: unknown;
  extent: number[];
  controls: {
    id: string;
    label: string;
    role?: string;
    lnglat?: number[];
    pixel: number[];
    required?: boolean;
  }[];
  plan: {
    status: string;
    graphHash: string;
    routes: {
      teamId: string;
      people: number;
      controlIds: string[];
      geometry: number[][];
      distanceM: number;
      durationSec: number;
    }[];
  };
}

/**
 * The first baked release from `web/public/data/releases.json`, together with
 * the full release document. Fails loudly (instead of skipping) when the baked
 * data was not committed: a clean checkout must contain it.
 */
export function firstBakedRelease(): StaticReleaseFixture {
  const index = readWorkspaceJSON<{ releases?: ReleaseSummary[] }>('web/public/data/releases.json');
  const summary = index.releases?.[0];
  if (!summary) throw new Error('web/public/data/releases.json contains no release; re-run `npm run bake`');
  return readWorkspaceJSON<StaticReleaseFixture>(`web/public/data/releases/${summary.id}.json`);
}

export interface PageAudit {
  /** Every URL containing `/api/` requested by the page — must stay empty. */
  apiRequests: string[];
  /** Uncaught page errors (`pageerror`) — must stay empty. */
  pageErrors: string[];
}

/**
 * Start auditing a page for back-end requests and uncaught errors.
 * Also installs a hard safety net: any `/api/**` request is aborted, so a
 * regression cannot silently reach a (nonexistent) server.
 */
export function auditPage(page: Page): PageAudit {
  const audit: PageAudit = { apiRequests: [], pageErrors: [] };
  page.on('request', request => {
    if (/(^|\/)api\//.test(new URL(request.url()).pathname)) audit.apiRequests.push(request.url());
  });
  page.on('pageerror', error => audit.pageErrors.push(error instanceof Error ? error.message : String(error)));
  void page.route('**/api/**', route => route.abort('failed'));
  return audit;
}

/**
 * Cross-origin dependencies that are part of the *existing visual design* and
 * are intentionally loaded from public CDNs by a static GitHub Pages site:
 *
 * - OpenStreetMap raster tiles: the optional online base map under the scan;
 * - Google Fonts: the pre-existing `@import` in `web/src/style.css`.
 *
 * Application code, data, map imagery and the service worker must always stay
 * below the deployment base; everything else is a test-visible violation.
 */
export const ALLOWED_EXTERNAL: RegExp[] = [
  /^https:\/\/(?:[a-c]\.)?tile\.openstreetmap\.org\//,
  /^https:\/\/fonts\.googleapis\.com\//,
  /^https:\/\/fonts\.gstatic\.com\//,
];

export function isAllowedExternal(url: string): boolean {
  return ALLOWED_EXTERNAL.some(pattern => pattern.test(url));
}

/** Wait until the teacher workbench is fully bootstrapped (3 catalog maps + controls). */
export async function waitForTeacherApp(page: Page): Promise<void> {
  await expect(page.locator('[data-map]')).toHaveCount(3, { timeout: 30_000 });
  await expect(page.locator('.control-row').first()).toBeVisible({ timeout: 30_000 });
  await expect(page.locator('#map-title')).not.toBeEmpty({ timeout: 30_000 });
}

/**
 * Wait until no background `task()` is running.
 *
 * `main.ts` disables these buttons around every async task (map switch, overlay
 * stitch, plan, reset), so their enabled state is a reliable idle signal.
 */
export async function waitForIdle(page: Page): Promise<void> {
  await expect(page.locator('#plan-button')).toBeEnabled({ timeout: 30_000 });
  await expect(page.locator('#edit-controls')).toBeEnabled({ timeout: 30_000 });
}

/** Wait until the student course view is bootstrapped for a baked release. */
export async function waitForStudentApp(page: Page, routeCount: number): Promise<void> {
  await expect(page.locator('#student-panel')).toBeVisible({ timeout: 30_000 });
  await expect(page.locator('#header-mode')).toHaveText('学生导航', { timeout: 30_000 });
  await expect(page.locator('#student-team option')).toHaveCount(routeCount, { timeout: 30_000 });
  await expect(page.locator('#check-in')).toBeEnabled({ timeout: 30_000 });
}

/** Wait for the `?e2e=1` hook (`window.dirmap`) to expose a booted course. */
export async function waitForDirmapHook(page: Page): Promise<void> {
  await page.waitForFunction(() => {
    const app = (window as unknown as { dirmap?: Record<string, unknown> }).dirmap;
    return Boolean(app && app.getState && app.getRelease && app.getPlan && (app.getState as () => unknown)() && (app.getRelease as () => unknown)());
  }, undefined, { timeout: 90_000 });
}

/** Browser-side mapper: control id → human label for a baked map state. */
export function labelsOf(state: { controls: { id: string; label: string }[] }): Map<string, string> {
  return new Map(state.controls.map(control => [control.id, control.label]));
}
