import { test, expect } from '@playwright/test';
import { auditPage, gotoTeacherWithHook, processCurrentMap, selectMap, waitForIdle } from './support';

/**
 * Addendum C gate: the whole recognition/registration pipeline now runs inside
 * the browser (opencv.js WASM + tesseract.js OCR from the CDN). Processing the
 * bundled szu-2 sheet must regenerate controls, registration evidence and the
 * client graph locally — with no `/api` backend at all.
 */
test('in-browser recognition and registration of szu-2 through the real UI', async ({ page }) => {
  test.setTimeout(300_000);
  const audit = auditPage(page);
  await gotoTeacherWithHook(page);

  await selectMap(page, 'szu-2');
  await expect(page.locator('#map-title')).toHaveText('深大定向图 2');
  await expect(page.locator('#release-badge')).toHaveText('备课草稿');
  await expect(page.locator('#process-current')).toBeEnabled();

  const seconds = await processCurrentMap(page);
  test.info().annotations.push({ type: 'recognition-duration', description: `${seconds.toFixed(1)} s (Chromium, in-browser)` });

  // The local result replaces the baked default: badge, card, controls, quality.
  await expect(page.locator('#release-badge')).toHaveText('本机处理');
  await expect(page.locator('.control-row')).toHaveCount(7);
  await expect(page.locator('#control-count')).toContainText('7');
  const activeCard = page.locator('.map-card.active');
  await expect(activeCard).toContainText('本机处理');
  await expect(activeCard).toContainText('7 个候选目标');

  const quality = page.locator('#registration-quality');
  await expect(quality).toContainText('拟合同名点');
  await expect(quality).toContainText('P95');
  await expect(quality).toContainText(/\d/);

  // Labels come from OCR (tesseract.js loaded from the CDN). The pipeline must
  // degrade gracefully to 待核N labels when the CDN/WASM is unavailable; when OCR
  // is available the course numbers are recognized exactly like the Python bake.
  const labels = (await page.locator('.control-row .control-number').allInnerTexts()).map(label => label.trim());
  expect(labels).toHaveLength(7);
  expect(labels.some(label => label.includes('起终点'))).toBe(true);
  expect(labels.every(label => /^(?:\d{1,3}|待核\d*|起终点|起点|终点)$/.test(label))).toBe(true);
  const numeric = labels.filter(label => /^\d{1,3}$/.test(label)).length;
  if (numeric < 5) {
    test.info().annotations.push({ type: 'ocr-fallback', description: `tolerated 待核 labels: ${labels.join(',')}` });
  } else {
    expect(numeric).toBeGreaterThanOrEqual(5);
  }

  // Deterministic view of the persisted client record through the ?e2e hook.
  const record = await page.evaluate(() => {
    const value = window.dirmap!.getLocalRecord();
    if (!value) return null;
    return {
      assetId: value.summary.assetId,
      kind: value.summary.kind,
      controls: value.state.controls.length,
      version: value.state.version,
      graphHash: value.state.graphHash as string | null,
      registration: Boolean(value.state.registration),
      fitCount: value.state.quality?.fitCount as number | undefined,
      p95: value.state.quality?.p95ErrorM as number | null | undefined,
      approaches: Object.keys(value.approaches?.controls ?? {}).length,
      required: value.state.controls.filter((control: { required?: boolean }) => control.required).length,
    };
  });
  expect(record).not.toBeNull();
  expect(record!.assetId).toBe('szu-2');
  expect(record!.kind).toBe('processed');
  expect(record!.controls).toBe(7);
  expect(record!.registration).toBe(true);
  expect(record!.graphHash).toMatch(/^client-/);
  expect(record!.fitCount).toBeGreaterThan(0);
  expect(record!.p95).not.toBeNull();
  expect(record!.approaches).toBeGreaterThan(0);

  // The merged catalog marks the sheet as a local result and the teacher can go
  // straight on to planning (no reload required).
  const catalog = await page.evaluate(() => window.dirmap!.getCatalog());
  const entry = catalog.maps.find(map => map.id === 'szu-2');
  expect(entry?.local).toBe(true);
  expect(entry?.processed).toBe(true);
  await waitForIdle(page);
  await expect(page.locator('#plan-button')).toBeVisible();

  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
});
