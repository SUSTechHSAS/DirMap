import { test, expect } from '@playwright/test';
import { auditPage, gotoTeacherWithHook, processCurrentMap, selectMap, waitForJob, waitForMapIdle } from './support';

interface AnchorControl { id: string; label: string; pixel: [number, number]; lnglat: [number, number]; role?: string }

function triangleArea(a: AnchorControl, b: AnchorControl, c: AnchorControl): number {
  return Math.abs((b.pixel[0] - a.pixel[0]) * (c.pixel[1] - a.pixel[1]) - (c.pixel[0] - a.pixel[0]) * (b.pixel[1] - a.pixel[1])) / 2;
}

function insideTriangle(point: AnchorControl, a: AnchorControl, b: AnchorControl, c: AnchorControl): boolean {
  const s1 = (b.pixel[0] - a.pixel[0]) * (point.pixel[1] - a.pixel[1]) - (b.pixel[1] - a.pixel[1]) * (point.pixel[0] - a.pixel[0]);
  const s2 = (c.pixel[0] - b.pixel[0]) * (point.pixel[1] - b.pixel[1]) - (c.pixel[1] - b.pixel[1]) * (point.pixel[0] - b.pixel[0]);
  const s3 = (a.pixel[0] - c.pixel[0]) * (point.pixel[1] - c.pixel[1]) - (a.pixel[1] - c.pixel[1]) * (point.pixel[0] - c.pixel[0]);
  return (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
}

/** Pick 3 well-spread fit anchors and 1 validation anchor inside their hull. */
function pickAnchorControls(controls: AnchorControl[]): { fit: AnchorControl[]; validation: AnchorControl } {
  const usable = controls.filter(control => Array.isArray(control.pixel) && Array.isArray(control.lnglat));
  let best: { area: number; indexes: [number, number, number] } | null = null;
  for (let i = 0; i < usable.length; i += 1) {
    for (let j = i + 1; j < usable.length; j += 1) {
      for (let k = j + 1; k < usable.length; k += 1) {
        const area = triangleArea(usable[i], usable[j], usable[k]);
        if (!best || area > best.area) best = { area, indexes: [i, j, k] };
      }
    }
  }
  if (!best) throw new Error('Not enough calibrated controls for anchors');
  const fit = best.indexes.map(index => usable[index]);
  const validation = usable.find(control => !fit.includes(control) && insideTriangle(control, fit[0], fit[1], fit[2]));
  if (!validation) throw new Error('No control inside the fit triangle can be used as a validation anchor');
  return { fit, validation };
}

/**
 * Addendum C gate: 同名点校准 — a teacher picks four same-name points on the
 * original scan and the matching locations on the standard map; three fitting
 * anchors + one independent check point re-run registration in the browser.
 */
test('four same-name anchors re-calibrate szu-2 in the browser', async ({ page }) => {
  test.setTimeout(300_000);
  const audit = auditPage(page);
  await gotoTeacherWithHook(page);
  await selectMap(page, 'szu-2');

  // Anchors need the locally recognized controls/analysis, so process first.
  const processSeconds = await processCurrentMap(page);
  test.info().annotations.push({ type: 'recognition-duration', description: `${processSeconds.toFixed(1)} s (Chromium, in-browser)` });
  await waitForMapIdle(page);

  const before = await page.evaluate(() => {
    const state = window.dirmap!.getState()!;
    const record = window.dirmap!.getLocalRecord()!;
    return {
      version: state.version,
      width: state.width,
      height: state.height,
      graphHash: record.state.graphHash as string | null,
      controls: state.controls.map(control => ({ id: control.id, label: control.label, pixel: control.pixel, lnglat: control.lnglat, role: control.role })) as AnchorControl[],
    };
  });
  const { fit, validation } = pickAnchorControls(before.controls);
  const anchors = [...fit, validation].map(control => ({ pixel: control.pixel, lnglat: control.lnglat }));

  // 地图配准 tab: add 3 fitting anchors and 1 validation anchor. Each anchor is
  // one click in the original-image editor canvas + one click on the standard map.
  await page.locator('[data-tab="registration"]').click();
  await expect(page.locator('#tab-registration')).toBeVisible();
  for (const [index, anchor] of anchors.entries()) {
    await page.locator('#validation-anchor').setChecked(index === anchors.length - 1);
    await page.locator('#add-anchor').click();
    const dialog = page.locator('dialog.editor-dialog');
    await expect(dialog).toBeVisible({ timeout: 30_000 });
    const canvas = dialog.locator('canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('editor canvas has no layout box');
    await canvas.click({ position: { x: anchor.pixel[0] / before.width * box.width, y: anchor.pixel[1] / before.height * box.height } });
    await expect(page.locator('dialog.editor-dialog')).toHaveCount(0);
    const point = await page.evaluate(lnglat => {
      const lon = lnglat[0] * Math.PI / 180 * 6378137;
      const lat = Math.log(Math.tan(Math.PI / 4 + lnglat[1] * Math.PI / 360)) * 6378137;
      return window.dirmap!.campusMap.map.getPixelFromCoordinate([lon, lat]);
    }, anchor.lnglat);
    await page.locator('#map').click({ position: { x: point[0], y: point[1] } });
    await expect(page.locator('#anchor-list > div')).toHaveCount(index + 1);
  }

  const anchorTexts = await page.locator('#anchor-list > div').allInnerTexts();
  expect(anchorTexts.filter(text => text.includes('拟合点'))).toHaveLength(3);
  expect(anchorTexts.filter(text => text.includes('核查点'))).toHaveLength(1);
  await expect(page.locator('#fit-anchors')).toBeEnabled();

  const startedAt = Date.now();
  await page.locator('#fit-anchors').click();
  await waitForJob(page);
  const fitSeconds = (Date.now() - startedAt) / 1000;
  test.info().annotations.push({ type: 'anchor-fit-duration', description: `${fitSeconds.toFixed(1)} s (Chromium, in-browser)` });

  // Success path: anchors consumed, quality rebuilt from the teacher landmarks,
  // a new client graph hash, and a bumped version. (Python/TS parity:
  // fitCount = fitting anchors, validationCount = independent check points.)
  await expect(page.locator('#anchor-list > div')).toHaveCount(0);
  await expect(page.locator('#release-badge')).toHaveText('本机处理');
  await expect(page.locator('#registration-quality')).toContainText('标准地图地物匹配');
  await expect(page.locator('#toast')).toContainText('已用同名点重新校准');

  const after = await page.evaluate(() => {
    const state = window.dirmap!.getState()!;
    const record = window.dirmap!.getLocalRecord()!;
    return {
      version: state.version,
      graphHash: record.state.graphHash as string | null,
      fitCount: record.state.quality?.fitCount as number | undefined,
      validationCount: record.state.quality?.validationCount as number | undefined,
      p95: record.state.quality?.p95ErrorM as number | null | undefined,
      landmarks: ((record.state.quality?.landmarks ?? []) as unknown[]).length,
      registration: Boolean(record.state.registration),
    };
  });
  expect(after.version).toBe(before.version + 1);
  expect(after.graphHash).toMatch(/^client-/);
  expect(after.graphHash).not.toBe(before.graphHash);
  expect(after.registration).toBe(true);
  expect(after.fitCount).toBe(3);
  expect(after.validationCount).toBe(1);
  expect(after.landmarks).toBe(4);
  expect(typeof after.p95).toBe('number');
  expect(after.p95!).toBeLessThan(5);

  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
});
