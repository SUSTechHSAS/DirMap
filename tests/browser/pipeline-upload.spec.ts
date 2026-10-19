import { test, expect } from '@playwright/test';
import { auditPage, gotoTeacherWithHook, planReachableSubset, waitForJob } from './support';

/**
 * Addendum C gate: a teacher imports a new scan through the file input; the
 * browser recognizes + registers it, adds it to the local catalog as a new map
 * sheet, and a reachable subset of its targets can be planned into real routes.
 *
 * The fixture upload deliberately reuses a bundled sheet (szu-2's lossless
 * PNG) so a duplicate local sheet exercises the full upload pipeline end to end.
 */
test('uploaded scan is processed in-browser and produces a plannable local map', async ({ page }) => {
  test.setTimeout(300_000);
  const audit = auditPage(page);
  await gotoTeacherWithHook(page);
  await expect(page.locator('[data-map]')).toHaveCount(3);

  const startedAt = Date.now();
  await page.setInputFiles('#upload-input', 'web/public/data/maps/szu-2/original.png');
  await waitForJob(page);
  const seconds = (Date.now() - startedAt) / 1000;
  test.info().annotations.push({ type: 'upload-process-duration', description: `${seconds.toFixed(1)} s (Chromium, in-browser)` });

  // New local catalog entry with a processed state, automatically selected.
  await expect(page.locator('[data-map]')).toHaveCount(4);
  const lastCard = page.locator('.map-card').last();
  await expect(lastCard).toContainText('本机处理');
  const activeCard = page.locator('.map-card.active');
  await expect(activeCard).toContainText('本机处理');
  await expect(page.locator('#release-badge')).toHaveText('本机处理');
  await expect(page.locator('#map-title')).toHaveText('original');
  expect(await page.locator('.control-row').count()).toBeGreaterThan(0);

  const record = await page.evaluate(() => {
    const value = window.dirmap!.getLocalRecord();
    const catalog = window.dirmap!.getCatalog();
    return value ? {
      assetId: value.summary.assetId,
      kind: value.summary.kind,
      controls: value.state.controls.length,
      registration: Boolean(value.state.registration),
      fitCount: value.state.quality?.fitCount as number | undefined,
      p95: value.state.quality?.p95ErrorM as number | null | undefined,
      graphHash: value.state.graphHash as string | null,
      catalogLength: catalog.maps.length,
      catalogLocal: catalog.maps.filter(map => map.local).length,
    } : null;
  });
  expect(record).not.toBeNull();
  expect(record!.kind).toBe('upload');
  expect(record!.assetId).toMatch(/^upload-/);
  expect(record!.controls).toBeGreaterThan(0);
  expect(record!.registration).toBe(true);
  expect(record!.fitCount).toBeGreaterThan(0);
  expect(record!.p95).not.toBeNull();
  expect(record!.graphHash).toMatch(/^client-/);
  expect(record!.catalogLength).toBe(4);
  expect(record!.catalogLocal).toBe(1);

  // A reachable subset of the recognized targets can be planned; the helper
  // unchecks the points the client planner reports as unresolved (replan loop).
  const plan = await planReachableSubset(page, { people: 6, teams: 2 });
  expect(['OPTIMAL', 'FEASIBLE']).toContain(plan.status);
  expect(plan.routes).toHaveLength(2);
  expect(plan.graphHash).toMatch(/^client-/);
  for (const route of plan.routes) {
    expect(route.controlIds.length).toBeGreaterThan(0);
    expect(route.distanceM).toBeGreaterThan(0);
    expect(route.durationSec).toBeGreaterThan(0);
  }
  await expect(page.locator('#route-results')).toContainText('队伍路线');
  await expect(page.locator('.route-card')).toHaveCount(2);
  await expect(page.locator('#plan-errors')).toBeEmpty();

  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
});
