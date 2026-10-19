import { test, expect } from '@playwright/test';
import { auditPage, gotoTeacherWithHook, selectMap, waitForJob } from './support';

/**
 * Addendum C gate: "更新通行识别" re-runs the terrain analysis in the browser
 * while keeping the already recognized/teacher-edited controls. This is the
 * cheap refresh path (no OCR), used after checking walls, doors and water.
 */
test('in-browser terrain refresh keeps controls and stores a fresh local analysis', async ({ page }) => {
  test.setTimeout(300_000);
  const audit = auditPage(page);
  await gotoTeacherWithHook(page);
  await selectMap(page, 'szu-2');

  const before = await page.evaluate(() => {
    const state = window.dirmap!.getState()!;
    return {
      controls: state.controls.map(control => ({ id: control.id, label: control.label })),
      version: state.version,
      forbidden: state.statistics?.forbiddenPixelFraction ?? null,
    };
  });
  await expect(page.locator('.control-row')).toHaveCount(7);

  // The refresh button lives in the 通行核验 tab and must be reachable there.
  await page.locator('[data-tab="terrain"]').click();
  await expect(page.locator('#tab-terrain')).toBeVisible();
  await expect(page.locator('#refresh-terrain')).toBeVisible();

  const startedAt = Date.now();
  await page.locator('#refresh-terrain').click();
  await waitForJob(page);
  const seconds = (Date.now() - startedAt) / 1000;
  test.info().annotations.push({ type: 'terrain-refresh-duration', description: `${seconds.toFixed(1)} s (Chromium, in-browser)` });

  // Controls are preserved; the analysis result becomes a local map record.
  await expect(page.locator('#release-badge')).toHaveText('本机处理');
  await expect(page.locator('.control-row')).toHaveCount(7);
  await expect(page.locator('.map-card.active')).toContainText('本机处理');
  await expect(page.locator('#terrain-summary')).toContainText('%');
  await expect(page.locator('#terrain-summary')).toContainText('本机识别/处理结果');

  const after = await page.evaluate(() => {
    const state = window.dirmap!.getState()!;
    const record = window.dirmap!.getLocalRecord()!;
    return {
      controls: state.controls.map(control => ({ id: control.id, label: control.label })),
      version: state.version,
      graphHash: record.state.graphHash as string | null,
      kind: record.summary.kind,
      storedControls: record.state.controls.length,
      forbidden: record.state.statistics?.forbiddenPixelFraction ?? null,
    };
  });
  expect(after.controls).toEqual(before.controls);
  expect(after.version).toBe(before.version + 1);
  expect(after.kind).toBe('processed');
  expect(after.storedControls).toBe(7);
  expect(after.graphHash).toMatch(/^client-/);
  expect(typeof after.forbidden).toBe('number');
  expect(after.forbidden).toBeGreaterThanOrEqual(0);
  expect(after.forbidden).toBeLessThanOrEqual(1);

  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
});
