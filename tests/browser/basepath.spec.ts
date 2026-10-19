import { test, expect } from '@playwright/test';
import { auditPage, isAllowedExternal, waitForIdle, waitForTeacherApp } from './support';

/**
 * Gate 6: the same application built with `DIRMAP_BASE=/DIRMAP/` and served from
 * a GitHub-Pages-style project sub-path works and keeps every application URL
 * under the deployment base (no leading-slash URLs leaking to the site root).
 *
 * This spec only runs in the `subpath` Playwright project.
 */
test('sub-path build keeps the whole application under /DIRMAP/', async ({ page }) => {
  test.setTimeout(120_000);
  const base = 'http://127.0.0.1:4174/DIRMAP/';
  const audit = auditPage(page);
  const requests: string[] = [];
  page.on('request', request => { if (/^https?:\/\//.test(request.url())) requests.push(request.url()); });

  // Navigate to the GitHub-Pages-style project URL explicitly: in this project
  // `baseURL` is http://127.0.0.1:4174/DIRMAP/ (a relative '/' would resolve to
  // the site root, which is exactly what a project page must never use).
  await page.goto(base);
  await waitForTeacherApp(page);

  expect(await page.evaluate(() => window.location.pathname)).toMatch(/^\/DIRMAP\//);
  await expect(page.locator('[data-map]')).toHaveCount(3);
  await expect(page.locator('.control-row')).toHaveCount(11);
  await expect(page.locator('#plan-button')).toBeVisible();

  // A real interaction, to prove the build uses base-aware URLs for data,
  // images and static assets (the overlay/terrain images below the base).
  await waitForIdle(page);
  await page.locator('#stitch-maps').check();
  await expect(page.locator('#stitch-maps')).toBeChecked();
  await waitForIdle(page);
  await page.locator('[data-tab="terrain"]').click();
  const terrainResponse = page.waitForResponse(response => response.url().includes('/DIRMAP/data/maps/szu-1/terrain-warp.png') && response.ok(), { timeout: 30_000 });
  await page.locator('#show-terrain').check();
  await terrainResponse;

  // Nothing may be requested from the site root (`/assets/...`, `/data/...`) —
  // only the /DIRMAP/ prefix is legitimate, plus cross-origin OSM raster tiles.
  const foreign = requests.filter(url => !url.startsWith(base) && !isAllowedExternal(url));
  expect(foreign, `resources fetched outside ${base}: ${foreign.join(', ')}`).toEqual([]);
  expect(requests.some(url => url.startsWith(`${base}data/catalog.json`))).toBe(true);
  expect(requests.some(url => url.startsWith(`${base}data/maps/szu-1/state.json`))).toBe(true);
  expect(requests.some(url => url.startsWith(`${base}assets/`))).toBe(true);
  expect(requests.some(url => url.includes('/DIRMAP/data/maps/szu-1/overlay.png'))).toBe(true);

  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
});
