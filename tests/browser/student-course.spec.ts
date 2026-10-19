import { test, expect } from '@playwright/test';
import { auditPage, firstBakedRelease, waitForStudentApp } from './support';

/**
 * Gate 4: the baked student course (`szu-3` release committed under
 * `web/public/data/`) opens without a backend, supports team selection,
 * check-in and persistent local progress, and does not call any API.
 */
test('baked student course renders and check-in progress persists locally', async ({ page }) => {
  const release = firstBakedRelease();
  const routeCount = release.plan.routes.length;
  const teamId = release.plan.routes[0].teamId;
  const controlCount = release.plan.routes[0].controlIds.length;
  const progressKey = `dirmap-progress:${release.id}:${teamId}`;
  const firstControlId = release.plan.routes[0].controlIds[0];
  const audit = auditPage(page);

  await page.goto(`/?course=${release.id}`);
  await waitForStudentApp(page, routeCount);

  // Student shell replaces the teacher workbench.
  await expect(page.locator('#teacher-panel')).toBeHidden();
  await expect(page.locator('#student-panel')).toBeVisible();
  await expect(page.locator('#header-mode')).toHaveText('学生导航');
  await expect(page.locator('#course-title')).toHaveText(release.title);
  await expect(page.locator('#release-badge')).toHaveText('课堂导航');
  await expect(page.locator('#student-team option')).toHaveCount(routeCount);
  await expect(page.locator('#student-team')).toHaveValue(teamId);
  await expect(page.locator('#next-control')).toBeVisible();
  await expect(page.locator('#check-in')).toBeEnabled();
  await expect(page.locator('#student-progress')).toHaveText(`0 / ${controlCount}`);
  await expect(page.locator('.student-stop')).toHaveCount(controlCount);

  // First check-in is stored under the documented per-course/per-team key.
  await page.locator('#check-in').click();
  await expect(page.locator('#student-progress')).toHaveText(`1 / ${controlCount}`);
  await expect(page.locator('.student-stop.done')).toHaveCount(1);
  expect(await page.evaluate(key => localStorage.getItem(key), progressKey)).toBe(JSON.stringify([firstControlId]));

  // Reload keeps the progress for this browser.
  await page.reload();
  await waitForStudentApp(page, routeCount);
  await expect(page.locator('#student-progress')).toHaveText(`1 / ${controlCount}`);
  await expect(page.locator('.student-stop.done')).toHaveCount(1);
  expect(await page.evaluate(key => localStorage.getItem(key), progressKey)).toBe(JSON.stringify([firstControlId]));

  // Undo removes the last check-in again.
  await page.locator('#undo-check-in').click();
  await expect(page.locator('#student-progress')).toHaveText(`0 / ${controlCount}`);
  await expect(page.locator('.student-stop.done')).toHaveCount(0);
  expect(await page.evaluate(key => localStorage.getItem(key), progressKey)).toBe('[]');

  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
});
