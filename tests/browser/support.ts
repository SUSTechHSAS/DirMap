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

// ---------------------------------------------------------------------------
// In-browser pipeline (?e2e=1) helpers
// ---------------------------------------------------------------------------

/**
 * The deterministic `window.dirmap` hook exposed by the production build when
 * the URL carries `?e2e=1` (see web/src/main.ts). Only the members the browser
 * suite uses are typed here.
 */
export interface DirmapHook {
  getState: () => { controls: { id: string; label: string; role?: string; pixel: number[]; lnglat?: number[]; calibrated?: boolean }[]; width: number; height: number; version: number; graphHash?: string | null; registration?: unknown; extent?: number[] } | null;
  getRelease: () => Record<string, any> | null;
  getPlan: () => Record<string, any> | null;
  getLocalRecord: () => { summary: { assetId: string; kind: string }; state: Record<string, any>; approaches?: { controls?: Record<string, any> } } | null;
  getCatalog: () => { maps: { id: string; local?: boolean; processed?: boolean; controls?: number | null }[]; legends: unknown[] };
  getClientPlanner: (record: unknown, progress?: (percent: number, message: string) => void) => Promise<{ graphHash: string; navigate: (...args: any[]) => Promise<any> }>;
  processCurrent: () => Promise<void>;
  uploadBlob: (blob: Blob, name: string) => Promise<void>;
  campusMap: { map: { getView: () => { getAnimating: () => boolean }; getPixelFromCoordinate: (coordinate: number[]) => [number, number] } };
}

declare global {
  interface Window {
    dirmap?: DirmapHook;
  }
}

/** Navigate to the teacher workbench with the deterministic E2E hook enabled. */
export async function gotoTeacherWithHook(page: Page, path = '/?e2e=1'): Promise<void> {
  await page.goto(path);
  await waitForTeacherApp(page);
  expect(await page.evaluate(() => Boolean((window as Window).dirmap?.getState))).toBe(true);
}

/** Select a map card through a real click and wait until its async task finished. */
export async function selectMap(page: Page, id: string): Promise<void> {
  await expect(page.locator('#process-current')).toBeEnabled({ timeout: 60_000 });
  await page.locator(`[data-map="${id}"]`).click();
  await page.waitForFunction(id => document.querySelector<HTMLElement>('.map-card.active')?.dataset.map === id, id, { timeout: 60_000 });
  await waitForIdle(page);
}

/** Wait until the map view has finished its fit animation (screen-position math). */
export async function waitForMapIdle(page: Page): Promise<void> {
  await page.waitForFunction(() => !(window as Window).dirmap?.campusMap.map.getView().getAnimating(), undefined, { timeout: 30_000 });
}

/**
 * Wait for a heavy in-browser pipeline task.
 *
 * The application shows `#job-status` while a pipeline stage reports progress and
 * hides it in the task's `finally`. Cheap tasks may finish before we observe the
 * visible state, so the "became visible" probe is best-effort.
 */
export async function waitForJob(page: Page, timeout = 240_000): Promise<void> {
  await page
    .waitForFunction(() => {
      const element = document.querySelector('#job-status');
      return Boolean(element && !element.classList.contains('hidden'));
    }, undefined, { timeout: 20_000 })
    .catch(() => { /* task was faster than the probe */ });
  await page.waitForFunction(() => {
    const element = document.querySelector('#job-status');
    return Boolean(element && element.classList.contains('hidden'));
  }, undefined, { timeout, polling: 250 });
}

/** Run in-browser recognition + registration for the active map through the UI. */
export async function processCurrentMap(page: Page): Promise<number> {
  const startedAt = Date.now();
  await page.locator('#process-current').click();
  await waitForJob(page);
  return (Date.now() - startedAt) / 1000;
}

export interface PlannedRouteSummary {
  teamId: string;
  controlIds: string[];
  distanceM: number;
  durationSec: number;
}

export interface PlanSummary {
  status: string;
  graphHash: string;
  routes: PlannedRouteSummary[];
  approaches: { controlId: string; lnglat?: number[]; automatic?: boolean }[];
  /** Unresolved points found on each NEEDS_REVIEW attempt (before unchecking). */
  rejected: { id: string; message: string }[][];
}

/**
 * Plan a course, automatically excluding controls the planner reports as
 * unreachable/ambiguous (e.g. client-processed szu-2 has an unreachable target).
 *
 * This is exactly the teacher workflow: plan → read "这些位置需要核对后再规划"
 * → uncheck those rows → plan again. It returns the first successful plan.
 */
export async function planReachableSubset(page: Page, options: { people?: number; teams?: number } = {}): Promise<PlanSummary> {
  const people = options.people ?? 6;
  const teams = options.teams ?? 2;
  await page.locator('[data-tab="course"]').click();
  await page.locator('#people').fill(String(people));
  await page.locator('#teams').fill(String(teams));
  const rejected: { id: string; message: string }[][] = [];
  for (let attempt = 0; attempt < 6; attempt += 1) {
    await expect(page.locator('#plan-button')).toBeEnabled({ timeout: 60_000 });
    await page.locator('#plan-button').click();
    await page.waitForFunction(() => {
      const plan = (window as Window).dirmap?.getPlan();
      return Boolean(plan && ['OPTIMAL', 'FEASIBLE', 'NEEDS_REVIEW'].includes(String(plan.status)));
    }, undefined, { timeout: 180_000 });
    const plan = await page.evaluate<PlanSummary & { status: string; unresolved: { id: string; message: string }[] }>(() => {
      const current = (window as Window).dirmap!.getPlan()!;
      return {
        status: String(current.status),
        graphHash: String(current.graphHash),
        routes: (current.routes as any[]).map(route => ({ teamId: route.teamId, controlIds: route.controlIds, distanceM: route.distanceM, durationSec: route.durationSec })),
        approaches: ((current.approaches ?? []) as any[]).map(approach => ({ controlId: approach.controlId, lnglat: approach.lnglat, automatic: approach.automatic })),
        rejected: [],
        unresolved: ((current.unresolved ?? []) as any[]).map(item => ({ id: item.id, message: item.message })),
      };
    });
    if (plan.status !== 'NEEDS_REVIEW') return { status: plan.status, graphHash: plan.graphHash, routes: plan.routes, approaches: plan.approaches, rejected };
    rejected.push(plan.unresolved);
    const state = await page.evaluate(() => (window as Window).dirmap!.getState()!);
    const labelById = new Map(state.controls.map(control => [control.id, control.label]));
    for (const item of plan.unresolved) {
      const label = labelById.get(item.id);
      if (label === undefined) throw new Error(`Planner reports unresolved "${item.id}" (${item.message}) which is not a selectable control row; cannot plan a reachable subset.`);
      const rows = page.locator('.control-row');
      const count = await rows.count();
      let unchecked = false;
      for (let index = 0; index < count; index += 1) {
        const row = rows.nth(index);
        const input = row.locator('[data-required]');
        if (await input.count() === 0 || await input.isDisabled()) continue;
        if ((await row.locator('.control-number').innerText()).trim() === label) {
          await input.uncheck();
          unchecked = true;
          break;
        }
      }
      if (!unchecked) throw new Error(`Cannot find the control row for unresolved label "${label}".`);
    }
  }
  throw new Error(`Plan stayed NEEDS_REVIEW after unchecking every reported point: ${JSON.stringify(rejected)}`);
}

/** Compute the screen point of a WGS84 coordinate inside the OL map viewport. */
export async function mapPointForLngLat(page: Page, lnglat: number[]): Promise<[number, number]> {
  return await page.evaluate(lnglat => {
    const lon = lnglat[0] * Math.PI / 180 * 6378137;
    const lat = Math.log(Math.tan(Math.PI / 4 + lnglat[1] * Math.PI / 360)) * 6378137;
    return (window as Window).dirmap!.campusMap.map.getPixelFromCoordinate([lon, lat]);
  }, lnglat);
}
