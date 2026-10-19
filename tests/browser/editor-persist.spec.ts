import { test, expect } from '@playwright/test';
import { auditPage, waitForIdle, waitForTeacherApp } from './support';

/**
 * Gate 3: a teacher edit performed through the real source editor is persisted
 * in this browser (localStorage), survives a reload, and "恢复构建时数据"
 * restores the immutable baked state.
 */
test('teacher label edit persists locally across reload and restores the baked data', async ({ page }) => {
  const audit = auditPage(page);
  await page.goto('/');
  await waitForTeacherApp(page);
  await expect(page.locator('#map-title')).toHaveText('深大定向图 1');
  await expect(page.locator('#release-badge')).toHaveText('备课草稿');

  const firstRow = page.locator('.control-row').first();
  const originalLabel = (await firstRow.locator('.control-number').innerText()).trim();
  expect(originalLabel.length).toBeGreaterThan(0);

  // Open the editor for the first control through the control list's "↗" button.
  await firstRow.locator('[data-control-edit]').click();
  const dialog = page.locator('dialog.editor-dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog.locator('canvas')).toBeVisible();
  const labelInput = dialog.locator('#control-label');
  await expect(labelInput).toBeVisible();
  await expect(labelInput).toHaveValue(originalLabel);

  await labelInput.fill('E2E-SZU1');
  await labelInput.blur();
  await dialog.locator('#save-edits').click();
  await expect(page.locator('dialog.editor-dialog')).toHaveCount(0);

  // Saved into this browser; the baked base stays untouched behind the override.
  await expect(page.locator('#release-badge')).toHaveText('本机草稿');
  await expect(page.locator('.control-row .control-number').first()).toHaveText('E2E-SZU1');
  const stored = await page.evaluate(() => localStorage.getItem('dirmap-static-state-v1:szu-1'));
  expect(stored, 'local state override').toBeTruthy();
  expect(JSON.parse(stored!)).toMatchObject({ controls: expect.arrayContaining([expect.objectContaining({ label: 'E2E-SZU1' })]) });
  await waitForIdle(page);

  // Reload: the local draft is merged over the baked state again.
  await page.reload();
  await waitForTeacherApp(page);
  await expect(page.locator('.control-row .control-number').first()).toHaveText('E2E-SZU1');
  await expect(page.locator('#release-badge')).toHaveText('本机草稿');

  // "恢复构建时数据" lives in the 通行核验 tab; restore the baked labels.
  await page.locator('[data-tab="terrain"]').click();
  await expect(page.locator('#tab-terrain')).toBeVisible();
  const reset = page.locator('#reset-static-data');
  await expect(reset).toBeEnabled();
  await reset.click();
  await expect(page.locator('#release-badge')).toHaveText('备课草稿');
  await expect(page.locator('.control-row .control-number').first()).toHaveText(originalLabel);
  await expect(reset).toBeDisabled();
  expect(await page.evaluate(() => localStorage.getItem('dirmap-static-state-v1:szu-1'))).toBeNull();

  await waitForIdle(page);
  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
});
