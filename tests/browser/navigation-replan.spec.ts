import { test, expect } from '@playwright/test';
import { auditPage, firstBakedRelease, waitForDirmapHook } from './support';

/**
 * Gate 5: client-side navigation re-planning in the real browser, with no
 * backend at all. The `?e2e=1` hook exposes the same planner `main.ts` uses,
 * so this exercises the real TypeScript routing core (graph load + navigate
 * validation + matrix re-plan) in production-build Chromium.
 */
test('student navigation replans client-side and rejects illegal positions', async ({ page }) => {
  test.setTimeout(180_000);
  const release = firstBakedRelease();
  const audit = auditPage(page);

  await page.goto(`/?course=${release.id}&e2e=1`);
  await waitForDirmapHook(page);

  const result = await page.evaluate(async () => {
    const hook = (window as unknown as { dirmap: Record<string, (...args: any[]) => any> }).dirmap;
    const state = hook.getState();
    const release = hook.getRelease();
    const planner = await hook.getPlanner(state);
    const teamId = (document.querySelector('#student-team') as HTMLSelectElement).value;
    const published = (release.plan.routes as any[]).find(route => route.teamId === teamId);
    if (!published) throw new Error(`no published route for selected team ${teamId}`);
    const controlIds = published.controlIds as string[];
    const first = (release.controls as any[]).find(control => control.id === controlIds[0]);
    if (!first?.lnglat) throw new Error(`control ${controlIds[0]} has no lnglat`);
    const legal = { teamId, lnglat: first.lnglat as [number, number], accuracyM: 5, timestampMs: Date.now(), completedControlIds: [] as string[] };

    const ready = await planner.navigate(release, legal, release.controls);
    const errors: Record<string, string> = {};
    const attempt = async (name: string, patch: Record<string, unknown>) => {
      try {
        await planner.navigate(release, { ...legal, ...patch }, release.controls);
        errors[name] = '';
      } catch (error) {
        errors[name] = error instanceof Error ? error.message : String(error);
      }
    };
    // Illegal accuracy, a non-prefix completed set (skips the first target) and
    // a stale timestamp must all be rejected by the planner itself.
    await attempt('accuracy', { accuracyM: 25 });
    await attempt('completedOrder', { completedControlIds: [controlIds[1]] });
    await attempt('staleTimestamp', { timestampMs: Date.now() - 60_000 });

    return {
      teamId,
      publishedControlIds: controlIds,
      releaseGraphHash: release.plan.graphHash as string,
      plannerGraphHash: planner.graphHash as string,
      status: ready.status as string,
      routeControlIds: (ready.route?.controlIds ?? null) as string[] | null,
      geometryLength: (ready.route?.geometry?.length ?? 0) as number,
      responseGraphHash: ready.graphHash as string,
      errors,
    };
  });

  expect(result.status).toBe('READY');
  expect(result.plannerGraphHash).toBe(result.releaseGraphHash);
  expect(result.responseGraphHash).toBe(result.releaseGraphHash);
  expect(result.routeControlIds).toEqual(result.publishedControlIds);
  expect(result.geometryLength).toBeGreaterThan(1);

  expect(result.errors.accuracy, 'accuracy > 20 m').toContain('定位精度');
  expect(result.errors.completedOrder, 'completed ids must be a prefix').toContain('课程顺序');
  expect(result.errors.staleTimestamp, 'timestamp older than 20 s').toContain('过期');

  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
});
