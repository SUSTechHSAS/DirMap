import { test, expect } from '@playwright/test';
import { auditPage, isAllowedExternal, waitForIdle, waitForTeacherApp } from './support';

/**
 * Gate 1: the pure-static application loads from the root deployment.
 *
 * No `/api/` request may happen (they are also hard-blocked), nothing may throw
 * an uncaught error, and the teacher can switch all three baked map sheets and
 * toggle the overlays that are part of the workbench.
 */
test('pure static app loads all three maps at / without any API request', async ({ page }) => {
  const audit = auditPage(page);
  const requests: string[] = [];
  page.on('request', request => { if (/^https?:\/\//.test(request.url())) requests.push(request.url()); });

  await page.goto('/');
  await waitForTeacherApp(page);

  // Three baked map sheets and a usable teacher workbench.
  await expect(page.locator('[data-map]')).toHaveCount(3);
  expect(await page.locator('.control-row').count()).toBeGreaterThan(5);
  await expect(page.locator('.control-row')).toHaveCount(11);
  await expect(page.locator('#build-tag')).toBeVisible();
  await expect(page.locator('#build-tag')).toContainText('静态版');
  await expect(page.locator('#release-badge')).toHaveText('备课草稿');
  await expect(page.locator('#map-title')).toHaveText('深大定向图 1');

  // Static-only build: the registration evidence and both committed legends are
  // present in the page (the tabs are display:none until selected, but rendered).
  await expect(page.locator('#registration-quality')).toContainText('P95');
  await expect(page.locator('#legend-links a')).toHaveCount(2);

  // ---- switch map sheets (each click must finish its async task) -------------
  await waitForIdle(page);
  await page.locator('[data-map="szu-2"]').click();
  await expect(page.locator('#map-title')).toHaveText('深大定向图 2');
  await expect(page.locator('.control-row')).toHaveCount(7);
  await waitForIdle(page);

  await page.locator('[data-map="szu-3"]').click();
  await expect(page.locator('#map-title')).toHaveText('深大定向图 3');
  await expect(page.locator('.control-row')).toHaveCount(17);
  await waitForIdle(page);

  await page.locator('[data-map="szu-1"]').click();
  await expect(page.locator('#map-title')).toHaveText('深大定向图 1');
  await expect(page.locator('.control-row')).toHaveCount(11);
  await waitForIdle(page);

  // ---- overlay toggles -------------------------------------------------------
  // "叠合三图" merges the other two baked overlays over the active sheet.
  await page.locator('#stitch-maps').check();
  await expect(page.locator('#stitch-maps')).toBeChecked();
  await waitForIdle(page);

  // "在地图上显示障碍候选" lives in the 通行核验 tab and must be reachable by a
  // real user click; selecting it proves the terrain overlay layer is built.
  await page.locator('[data-tab="terrain"]').click();
  await expect(page.locator('#tab-terrain')).toBeVisible();
  await expect(page.locator('#tab-course')).toBeHidden();
  const terrainResponse = page.waitForResponse(response => response.url().includes('/terrain-warp.png') && response.ok(), { timeout: 30_000 });
  await page.locator('#show-terrain').check();
  await expect(page.locator('#show-terrain')).toBeChecked();
  await terrainResponse;
  await page.locator('#show-terrain').uncheck();

  // Switching back to the course tab restores the planning workbench.
  await page.locator('[data-tab="course"]').click();
  await expect(page.locator('#tab-course')).toBeVisible();
  await expect(page.locator('#tab-terrain')).toBeHidden();
  await expect(page.locator('#plan-button')).toBeVisible();

  // ---- static hosting invariants --------------------------------------------
  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
  const base = 'http://127.0.0.1:4173/';
  const foreign = requests.filter(url => !url.startsWith(base) && !isAllowedExternal(url));
  expect(foreign, `requests that escaped the static deployment base: ${foreign.join(', ')}`).toEqual([]);
  // Everything the app needs is served as a plain file from the deployment root.
  expect(requests.some(url => url.startsWith(`${base}data/catalog.json`))).toBe(true);
  expect(requests.some(url => url.startsWith(`${base}data/maps/szu-1/state.json`))).toBe(true);
  expect(requests.some(url => url.startsWith(`${base}data/maps/szu-1/overlay.png`))).toBe(true);
  expect(requests.some(url => url.startsWith(`${base}assets/`))).toBe(true);
});
