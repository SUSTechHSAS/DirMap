import { test, expect } from '@playwright/test';
import { auditPage, labelsOf, readWorkspaceJSON, waitForIdle, waitForTeacherApp } from './support';

interface OptimizerFixture {
  assetId: string;
  graphHash: string;
  request: {
    controlIds: string[];
    mode: string;
    objective: string;
    startPixel: number[];
    finishPixel: number[];
    teams: { id: string; people: number; speedMps: number; departureSec: number }[];
    timeLimitSec: number;
    version: number;
  };
  expected: {
    status: string;
    proof: { algorithm: string; globalOptimalityProven: boolean };
    routes: { teamId: string; controlIds: string[]; order: number[] }[];
  };
}

interface BakedStateFixture {
  controls: { id: string; label: string; role?: string }[];
}

/**
 * Gate 2: a teacher plans a real course on szu-1 (10 controls → the exact
 * Held-Karp branch) in the browser, and the rendered routes match the baked
 * Python goldens in `tests/fixtures/static/optimizer.json`.
 */
test('teacher plans the szu-1 all-visit course and it matches the baked exact-optimizer goldens', async ({ page }) => {
  // Exact Held-Karp on 10 controls plus first graph load; 90 s is the task budget.
  test.setTimeout(90_000);
  const golden = readWorkspaceJSON<OptimizerFixture>('tests/fixtures/static/optimizer.json');
  const state = readWorkspaceJSON<BakedStateFixture>('web/public/data/maps/szu-1/state.json');
  const labels = labelsOf(state);
  const audit = auditPage(page);

  await page.goto('/?e2e=1');
  await waitForTeacherApp(page);

  // Default sheet is szu-1; its single recognized start/finish symbol is used
  // automatically ("已识别原图起点"/"已识别原图终点"), no manual map click needed.
  await expect(page.locator('#map-title')).toHaveText('深大定向图 1');
  await expect(page.locator('#pick-start')).toContainText('已识别原图起点');
  await expect(page.locator('#pick-finish')).toContainText('已识别原图终点');
  await expect(page.locator('#return-start')).toBeChecked();

  // Recreate the golden fixture's request through the real UI:
  // all_visit, 15 people / 3 teams, default walk speed and 20 s stagger.
  await page.locator('#course-mode').selectOption('all_visit');
  await page.locator('#people').fill('15');
  await page.locator('#teams').fill('3');
  await page.locator('#stagger').fill('20');
  await expect(page.locator('#group-preview')).toContainText('每队 5 人');
  await expect(page.locator('[data-required]:checked')).toHaveCount(golden.request.controlIds.length);

  await waitForIdle(page);
  await page.locator('#plan-button').click();
  await expect(page.locator('.route-card')).toHaveCount(3, { timeout: 110_000 });
  await expect(page.locator('#route-results')).toContainText('模型内已证明最优');
  await expect(page.locator('#plan-errors')).toBeEmpty();
  await expect(page.locator('#review-button')).toBeVisible();
  await expect(page.locator('#publish-section')).toBeVisible();

  // Read the real plan object through the documented static E2E hook.
  const plan = await page.evaluate(() => {
    const app = (window as unknown as { dirmap: { getPlan: () => Record<string, any> } }).dirmap;
    const value = app.getPlan();
    return {
      status: value.status as string,
      graphHash: value.graphHash as string,
      algorithm: value.proof?.algorithm as string,
      globalOptimalityProven: value.proof?.globalOptimalityProven as boolean,
      routes: (value.routes as any[]).map(route => ({
        teamId: route.teamId as string,
        controlIds: route.controlIds as string[],
        people: route.people as number,
        distanceM: route.distanceM as number,
        durationSec: route.durationSec as number,
      })),
    };
  });

  expect(plan.status).toBe(golden.expected.status);
  expect(plan.algorithm).toBe(golden.expected.proof.algorithm);
  expect(plan.globalOptimalityProven).toBe(true);
  expect(plan.graphHash).toBe(golden.graphHash);

  // Same teams, same team ids, same per-team visiting order as Python's DP.
  expect(plan.routes.map(route => route.teamId)).toEqual(golden.expected.routes.map(route => route.teamId));
  for (const [index, expected] of golden.expected.routes.entries()) {
    expect(plan.routes[index].controlIds, `team ${expected.teamId} control order`).toEqual(expected.controlIds);
    expect(plan.routes[index].people).toBe(golden.request.teams[index].people);
    expect(plan.routes[index].distanceM).toBeGreaterThan(0);
    expect(plan.routes[index].durationSec).toBeGreaterThan(0);
  }

  // The rendered cards show the same routes, labels and plausible metrics.
  for (const [index, expected] of golden.expected.routes.entries()) {
    const card = page.locator('.route-card').nth(index);
    await expect(card).toHaveAttribute('data-team', expected.teamId);
    const stops = expected.controlIds.map(id => labels.get(id) ?? id).join(' → ');
    await expect(card.locator('.route-stops')).toHaveText(stops);
    const metrics = (await card.locator('.route-metrics').innerText()).replace(/\s+/g, ' ').trim();
    expect(metrics, 'route distance/time metrics').toMatch(/(?:km|\bm\b)/);
    expect(metrics, 'route duration in minutes').toMatch(/分/);
    expect(metrics, 'route metrics start with a distance number').toMatch(/^\d/);
  }

  expect(audit.pageErrors).toEqual([]);
  expect(audit.apiRequests).toEqual([]);
});
