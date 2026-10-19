import { test, expect } from '@playwright/test';
import { auditPage, gotoTeacherWithHook, planReachableSubset, processCurrentMap, selectMap } from './support';

/**
 * Addendum C gate: publish a client-processed course locally (no backend).
 *
 * The teacher processes szu-2 in the browser, plans a reachable subset, confirms
 * the three review items and gets a `?course=<32 hex>` link. The course JSON is
 * saved in this browser's localStorage, so the student page opens from local data
 * in the same context (`本机课程`), check-in works, and the frozen client graph can
 * re-plan navigation without any server.
 */
test('locally published client-processed course opens for the student and re-plans navigation', async ({ page }) => {
  test.setTimeout(300_000);
  const audit = auditPage(page);
  await gotoTeacherWithHook(page);
  await selectMap(page, 'szu-2');

  const processSeconds = await processCurrentMap(page);
  test.info().annotations.push({ type: 'recognition-duration', description: `${processSeconds.toFixed(1)} s (Chromium, in-browser)` });

  // Plan a reachable subset (the client planner reports unreachable targets in
  // "这些位置需要核对后再规划"; the helper unchecks exactly those rows).
  const plan = await planReachableSubset(page, { people: 6, teams: 2 });
  expect(['OPTIMAL', 'FEASIBLE']).toContain(plan.status);
  expect(plan.routes).toHaveLength(2);
  expect(plan.graphHash).toMatch(/^client-/);
  await expect(page.locator('#plan-errors')).toBeEmpty();
  await expect(page.locator('#review-button')).toBeVisible();

  // Review + publish. For a local map the dialog explains that the course is
  // previewable in this browser only until its data is committed to the build.
  await page.locator('#review-button').click();
  const dialog = page.locator('dialog.info-dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('本机处理的地图仅能在本浏览器预览');
  for (const name of ['registration', 'controls', 'passages']) await page.locator(`#review-${name}`).check();
  if (await page.locator('#review-approaches').count()) await page.locator('#review-approaches').check();
  await page.locator('#publish-confirm').click();

  const linkInput = page.locator('#share-result input[aria-label="学生课程链接"]');
  await expect(linkInput).toBeVisible({ timeout: 120_000 });
  const link = await linkInput.inputValue();
  const courseId = new URL(link).searchParams.get('course');
  expect(courseId).toMatch(/^[0-9a-f]{32}$/);
  expect(new URL(link).searchParams.get('e2e')).toBeNull();

  // The release is stored in this browser (local-first static publishing).
  const storedIds = await page.evaluate(() => {
    const raw = JSON.parse(localStorage.getItem('dirmap-static-releases-v1') ?? '[]') as { id: string }[];
    return raw.map(release => release.id);
  });
  expect(storedIds).toContain(courseId);

  // Open the student page in the same context (localStorage/IndexedDB shared).
  const studentUrl = new URL(link);
  studentUrl.searchParams.set('e2e', '1');
  await page.goto(studentUrl.href);
  await expect(page.locator('#student-panel')).toBeVisible({ timeout: 60_000 });
  await expect(page.locator('#header-mode')).toHaveText('学生导航');
  await expect(page.locator('#release-badge')).toHaveText('本机课程');
  await expect(page.locator('#student-team option')).toHaveCount(plan.routes.length);
  const controlCount = plan.routes[0].controlIds.length;
  await expect(page.locator('#student-progress')).toHaveText(`0 / ${controlCount}`);
  await expect(page.locator('#check-in')).toBeEnabled();

  await page.locator('#check-in').click();
  await expect(page.locator('#student-progress')).toHaveText(`1 / ${controlCount}`);

  // Client-side re-plan on the frozen local course: READY when starting from the
  // published approach of the first remaining target.
  const navigation = await page.evaluate(async () => {
    const hook = window.dirmap!;
    const release = hook.getRelease()!;
    const record = hook.getLocalRecord()!;
    const planner = await hook.getClientPlanner(record);
    const teamId = (document.querySelector('#student-team') as HTMLSelectElement).value;
    const route = (release.plan.routes as any[]).find(candidate => candidate.teamId === teamId);
    const firstId = route.controlIds[0];
    const approach = ((release.plan.approaches ?? []) as any[]).find(item => item.controlId === firstId && item.lnglat);
    const control = ((release.controls ?? []) as any[]).find(item => item.id === firstId);
    const result = await planner.navigate(release, {
      teamId, lnglat: approach?.lnglat ?? control?.lnglat, accuracyM: 5, timestampMs: Date.now(), completedControlIds: [firstId],
    }, release.controls);
    return {
      status: result.status as string,
      routeControlIds: (result.route?.controlIds ?? null) as string[] | null,
      expectedRemaining: route.controlIds.slice(1) as string[],
      geometryLength: (result.route?.geometry?.length ?? 0) as number,
      graphHash: result.graphHash as string,
      planGraphHash: release.plan.graphHash as string,
      usedApproach: Boolean(approach),
      releaseRoutes: (release.plan.routes as any[]).map(candidate => candidate.controlIds),
    };
  });
  expect(navigation.status).toBe('READY');
  expect(navigation.usedApproach).toBe(true);
  expect(navigation.releaseRoutes[0]).toEqual(plan.routes[0].controlIds);
  expect(navigation.routeControlIds).toEqual(navigation.expectedRemaining);
  expect(navigation.graphHash).toBe(navigation.planGraphHash);
  expect(navigation.graphHash).toBe(plan.graphHash);
  expect(navigation.geometryLength).toBeGreaterThan(1);

  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
});
