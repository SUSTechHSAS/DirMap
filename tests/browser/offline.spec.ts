import { test, expect } from '@playwright/test';
import { auditPage, firstBakedRelease, waitForStudentApp } from './support';

/**
 * Gate 7: a saved course opens offline from the service-worker cache.
 *
 * The service worker intentionally keeps the shell and the course package
 * (release JSON + overlay + shared documents), not the large graph bundles;
 * offline the student page must still render the course, check in and persist
 * progress. Re-planning can degrade to the cached route offline, so this test
 * only asserts the student-facing course flow.
 */
test('saved course shell and student panel load from the service-worker cache offline', async ({ page, context }) => {
  test.slow();
  const release = firstBakedRelease();
  const routeCount = release.plan.routes.length;
  const controlCount = release.plan.routes[0].controlIds.length;
  const audit = auditPage(page);

  await page.goto(`/?course=${release.id}`);
  await waitForStudentApp(page, routeCount);
  // `cacheStudentCourse()` runs automatically on boot in the production build;
  // the explicit "保存本课供离线查看" button must work too (same SW message path).
  await expect(page.locator('#offline-status')).toContainText('本课已保存', { timeout: 30_000 });
  await page.locator('#save-offline').click();
  await expect(page.locator('#offline-status')).toContainText('本课已保存', { timeout: 30_000 });
  await page.waitForFunction(() => Boolean(navigator.serviceWorker.controller), undefined, { timeout: 30_000 });

  await context.setOffline(true);
  // This is a real gate now: a reload that cannot be served by the worker means
  // the offline shell regressed, so it must fail (not silently skip).
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 30_000 });

  // The cached application shell booted without any network.
  await expect(page.locator('#app')).not.toBeEmpty({ timeout: 30_000 });
  await expect(page.locator('#build-tag')).toBeVisible();
  await expect(page.locator('#student-panel')).toBeVisible();
  await expect(page.locator('#header-mode')).toHaveText('学生导航');
  await waitForStudentApp(page, routeCount);
  await expect(page.locator('#release-badge')).toContainText('离线');
  await expect(page.locator('#course-title')).toHaveText(release.title);
  await expect(page.locator('#student-progress')).toHaveText(`0 / ${controlCount}`);

  // Check-ins still work offline and are persisted locally.
  await page.locator('#check-in').click();
  await expect(page.locator('#student-progress')).toHaveText(`1 / ${controlCount}`);
  expect(await page.evaluate(id => localStorage.getItem(`dirmap-progress:${id}:${document.querySelector<HTMLSelectElement>('#student-team')!.value}`), release.id)).not.toBeNull();

  // Back online the page refreshes from the static deployment again.
  await context.setOffline(false);
  await page.reload();
  await waitForStudentApp(page, routeCount);
  await expect(page.locator('#release-badge')).toHaveText('课堂导航');
  await expect(page.locator('#student-progress')).toHaveText(`1 / ${controlCount}`);

  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
});
