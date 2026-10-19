/** Optimizer port tests: exact Held-Karp branches against brute-force
 * enumeration, mandatory/shared coverage rules, the labelled fallback and the
 * Python validation messages. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { solve, subsetTours, assignSubsets, routeCost } from '../../web/src/optimizer.ts';
import type { Control } from '../../web/src/types.ts';
import { readJSON } from './helpers.ts';

type Matrix = number[][];

function pseudoRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

function randomMatrix(size: number, seed: number, maxValue = 10): Matrix {
  const random = pseudoRandom(seed);
  const matrix: Matrix = [];
  for (let row = 0; row < size; row++) {
    const values: number[] = [];
    for (let column = 0; column < size; column++) {
      values.push(row === column ? 0 : 1 + random() * maxValue);
    }
    matrix.push(values);
  }
  return matrix;
}

function control(id: string, serviceBaseSec = 5, servicePerPersonSec = 1): Control {
  return {
    id, label: id, pixel: [0, 0], radiusPx: 15, required: true, state: 'accepted',
    serviceBaseSec, servicePerPersonSec,
  };
}

const controls = (n: number): Control[] => Array.from({ length: n }, (_, index) => control(`c${index + 1}`));

function permutations(values: number[]): number[][] {
  if (values.length <= 1) return [values.slice()];
  const result: number[][] = [];
  for (let index = 0; index < values.length; index++) {
    const rest = [...values.slice(0, index), ...values.slice(index + 1)];
    for (const tail of permutations(rest)) result.push([values[index], ...tail]);
  }
  return result;
}

function flat(matrix: Matrix): Float64Array {
  const values = new Float64Array(matrix.length * matrix.length);
  matrix.forEach((row, rowIndex) => row.forEach((value, columnIndex) => { values[rowIndex * matrix.length + columnIndex] = value; }));
  return values;
}

function sequenceCost(matrix: Matrix, sequence: number[]): number {
  let total = 0;
  for (let index = 0; index + 1 < sequence.length; index++) total += matrix[sequence[index]][sequence[index + 1]];
  return total;
}

function bruteForceAllVisit(matrix: Matrix): number {
  const n = matrix.length - 2;
  const nodes = Array.from({ length: n }, (_, index) => index + 1);
  let best = Infinity;
  for (const permutation of permutations(nodes)) {
    const cost = sequenceCost(matrix, [0, ...permutation, n + 1]);
    if (cost < best) best = cost;
  }
  return best;
}

function teamDuration(matrix: Matrix, sequence: number[], people: number, speed: number, service: number[]): number {
  let duration = sequenceCost(matrix, sequence) / speed;
  for (const node of sequence.slice(1, -1)) duration += service[node - 1];
  return duration;
}

function tupleCompare(left: number[], right: number[]): number {
  for (let index = 0; index < left.length; index++) if (left[index] !== right[index]) return left[index] - right[index];
  return 0;
}

test('Held-Karp subset_tours matches brute-force permutations', () => {
  for (const seed of [3, 11, 27]) {
    const matrix = randomMatrix(8, seed);
    const tours = subsetTours(flat(matrix), 8);
    const best = bruteForceAllVisit(matrix);
    const route = tours.route((1 << 6) - 1);
    assert.equal(route[0], 0);
    assert.equal(route[route.length - 1], 7);
    assert.ok(Math.abs(sequenceCost(matrix, route) - best) < 1e-9, `seed ${seed}: ${sequenceCost(matrix, route)} vs ${best}`);
    assert.ok(Math.abs(tours.costs[(1 << 6) - 1] - best) < 1e-9);
  }
});

test('all_visit solve is exactly optimal and identical for every team', () => {
  const matrix = randomMatrix(8, 5);
  const teams = [
    { id: 'A', people: 4, speedMps: 1.2, departureSec: 0 },
    { id: 'B', people: 6, speedMps: 1.2, departureSec: 20 },
  ];
  const result = solve(flat(matrix), controls(6), teams, 'all_visit', 'min_total_time', 5);
  assert.equal(result.status, 'OPTIMAL');
  assert.equal(result.proof.algorithm, 'held_karp');
  assert.equal(result.proof.globalOptimalityProven, true);
  assert.equal(result.routes.length, 2);
  assert.deepEqual(result.routes[0].order, result.routes[1].order);
  const expectedTravel = bruteForceAllVisit(matrix) / 1.2;
  assert.ok(Math.abs(result.routes[0].travelSec - expectedTravel) < 1e-9);
  assert.ok(Math.abs(result.routes[1].travelSec - expectedTravel) < 1e-9);
});

function bruteForceSplit(matrix: Matrix, teams: { people: number; speedMps: number }[], service: number[][]): number[] {
  const n = matrix.length - 2;
  const nodes = Array.from({ length: n }, (_, index) => index + 1);
  let best: number[] | null = null;
  for (let mask = 1; mask < (1 << n) - 1; mask++) {
    const first = nodes.filter(node => mask & (1 << (node - 1)));
    const second = nodes.filter(node => !(mask & (1 << (node - 1))));
    for (const firstOrder of permutations(first)) {
      const firstDuration = teamDuration(matrix, [0, ...firstOrder, n + 1], teams[0].people, teams[0].speedMps, service[0]);
      for (const secondOrder of permutations(second)) {
        const secondDuration = teamDuration(matrix, [0, ...secondOrder, n + 1], teams[1].people, teams[1].speedMps, service[1]);
        const value = [Math.max(firstDuration, secondDuration), firstDuration + secondDuration];
        if (best === null || tupleCompare(value, best) < 0) best = value;
      }
    }
  }
  assert.ok(best);
  return best;
}

test('split_coverage DP matches brute-force assignment for min_max_finish_then_sum', () => {
  const matrix = randomMatrix(6, 9);
  const teams = [{ id: 'A', people: 3, speedMps: 1.4, departureSec: 0 }, { id: 'B', people: 5, speedMps: 1.4, departureSec: 0 }];
  const all = controls(4);
  const service = teams.map(team => all.map(item => (item.serviceBaseSec ?? 5) + team.people * (item.servicePerPersonSec ?? 1)));
  const expected = bruteForceSplit(matrix, teams, service);
  const result = solve(flat(matrix), all, teams, 'split_coverage', 'min_max_finish_then_sum', 5);
  assert.equal(result.status, 'OPTIMAL');
  assert.equal(result.proof.algorithm, 'held_karp_subset_assignment_two_stage');
  assert.deepEqual(result.proof.objectiveValue.map(value => Math.round(value * 1e6) / 1e6), expected.map(value => Math.round(value * 1e6) / 1e6));
  const seen = new Set<string>();
  for (const route of result.routes) for (const id of route.controlIds) seen.add(id);
  assert.equal(seen.size, 4, 'shared controls must be split exactly once');
  assert.equal(result.coverage.mode, 'split_coverage');
  assert.equal(result.coverage.sharedControlIds.length, 4);
});

test('mixed_coverage keeps mandatory nodes on every team and splits the rest', () => {
  const matrix = randomMatrix(6, 21);
  const teams = [{ id: 'A', people: 3, speedMps: 1.4, departureSec: 0 }, { id: 'B', people: 5, speedMps: 1.4, departureSec: 0 }];
  const all = controls(4);
  const result = solve(flat(matrix), all, teams, 'mixed_coverage', 'min_max_finish_then_sum', 5, ['c4', 'c2'], false);
  assert.equal(result.status, 'OPTIMAL');
  assert.equal(result.proof.algorithm, 'held_karp_mandatory_shared_two_stage');
  for (const route of result.routes) {
    assert.ok(route.controlIds.includes('c2'));
    assert.ok(route.controlIds.includes('c4'));
  }
  const sharedCount = new Map<string, number>();
  for (const route of result.routes) for (const id of route.controlIds) sharedCount.set(id, (sharedCount.get(id) ?? 0) + 1);
  assert.equal(sharedCount.get('c2'), 2);
  assert.equal(sharedCount.get('c4'), 2);
  assert.equal(sharedCount.get('c1'), 1);
  assert.equal(sharedCount.get('c3'), 1);
});

function bruteForceMixed(
  matrix: Matrix, teams: { people: number; speedMps: number }[], service: number[][],
  mandatory: number[], ordered: number[],
): number[] {
  const n = matrix.length - 2;
  const shared: number[] = [];
  for (let node = 1; node <= n; node++) if (!mandatory.includes(node)) shared.push(node);
  let best: number[] | null = null;
  for (let mask = 0; mask < (1 << shared.length); mask++) {
    const first = [...mandatory, ...shared.filter((_, index) => mask & (1 << index))];
    const second = [...mandatory, ...shared.filter((_, index) => !(mask & (1 << index)))];
    for (const firstOrder of permutations(first)) {
      if (ordered.length && firstOrder.filter(node => ordered.includes(node)).join(',') !== ordered.join(',')) continue;
      const firstDuration = teamDuration(matrix, [0, ...firstOrder, n + 1], teams[0].people, teams[0].speedMps, service[0]);
      for (const secondOrder of permutations(second)) {
        if (ordered.length && secondOrder.filter(node => ordered.includes(node)).join(',') !== ordered.join(',')) continue;
        const secondDuration = teamDuration(matrix, [0, ...secondOrder, n + 1], teams[1].people, teams[1].speedMps, service[1]);
        const value = [Math.max(firstDuration, secondDuration), firstDuration + secondDuration];
        if (best === null || tupleCompare(value, best) < 0) best = value;
      }
    }
  }
  assert.ok(best);
  return best;
}

test('mixed_coverage exact branch matches brute force with mandatory order', () => {
  const matrix = randomMatrix(6, 31);
  const teams = [{ id: 'A', people: 3, speedMps: 1.4, departureSec: 0 }, { id: 'B', people: 5, speedMps: 1.4, departureSec: 0 }];
  const all = controls(4);
  const service = teams.map(team => all.map(item => (item.serviceBaseSec ?? 5) + team.people * (item.servicePerPersonSec ?? 1)));
  const expected = bruteForceMixed(matrix, teams, service, [3, 1], [3, 1]);
  const result = solve(flat(matrix), all, teams, 'mixed_coverage', 'min_max_finish_then_sum', 5, ['c3', 'c1'], true);
  assert.equal(result.status, 'OPTIMAL');
  assert.equal(result.proof.algorithm, 'held_karp_mandatory_shared_two_stage');
  assert.deepEqual(result.proof.objectiveValue.map(value => Math.round(value * 1e6) / 1e6), expected.map(value => Math.round(value * 1e6) / 1e6));
  for (const route of result.routes) {
    const ordered = route.controlIds.filter(id => id === 'c3' || id === 'c1');
    assert.deepEqual(ordered, ['c3', 'c1'], 'mandatory order preserved');
  }
});

test('fixed_order uses the selected control order for every team', () => {
  const matrix = randomMatrix(5, 41);
  const result = solve(flat(matrix), controls(3), [{ id: 'A', people: 2, speedMps: 1.2 }], 'fixed_order', 'min_total_time', 5);
  assert.equal(result.status, 'OPTIMAL');
  assert.equal(result.proof.algorithm, 'dijkstra_fixed_order');
  for (const route of result.routes) assert.deepEqual(route.order, [0, 1, 2, 3, 4]);
});

test('large all_visit and split_coverage problems fall back to FEASIBLE', () => {
  const allVisit = randomMatrix(16, 51);
  const all = controls(14);
  const allResult = solve(flat(allVisit), all, [{ id: 'A', people: 2, speedMps: 1.2 }, { id: 'B', people: 4, speedMps: 1.2 }], 'all_visit', 'min_max_finish_then_sum', 0.4);
  assert.equal(allResult.status, 'FEASIBLE');
  assert.equal(allResult.proof.algorithm, 'insertion_2opt_relocation');
  assert.equal(allResult.proof.globalOptimalityProven, false);
  assert.equal(allResult.proof.bestBound, null);
  assert.equal(allResult.proof.relativeGap, null);
  assert.equal(allResult.proof.scope, '当前图上的静态通行代价、打点时长和指定出发时间；不包含动态拥堵。');
  for (const route of allResult.routes) assert.equal(route.controlIds.length, 14);
  const split = randomMatrix(13, 61);
  const splitNodes = controls(11);
  const splitResult = solve(flat(split), splitNodes, [{ id: 'A', people: 2, speedMps: 1.2 }, { id: 'B', people: 4, speedMps: 1.2 }], 'split_coverage', 'min_max_finish_then_sum', 0.4);
  assert.equal(splitResult.status, 'FEASIBLE');
  const seen = new Map<string, number>();
  for (const route of splitResult.routes) for (const id of route.controlIds) seen.set(id, (seen.get(id) ?? 0) + 1);
  assert.equal(seen.size, 11);
  for (const count of seen.values()) assert.equal(count, 1);
});

test('solve validates its inputs with the Python Chinese messages', () => {
  const matrix = randomMatrix(4, 71);
  const teams = [{ id: 'A', people: 2, speedMps: 1.4 }];
  const items = controls(2);
  const call = (overrides: {
    matrix?: Matrix; items?: Control[]; teams?: { id: string; people: number; speedMps: number; departureSec?: number }[];
    mode?: string; objective?: string; timeLimit?: number; mandatoryIds?: string[]; mandatoryOrder?: boolean;
  } = {}) => solve(
    flat(overrides.matrix ?? matrix), overrides.items ?? items, overrides.teams ?? teams,
    overrides.mode ?? 'all_visit', overrides.objective ?? 'min_max_finish_then_sum',
    overrides.timeLimit ?? 5, overrides.mandatoryIds, overrides.mandatoryOrder,
  );
  assert.throws(() => call({ timeLimit: 0 }), /求解预算必须大于零且不超过 30 秒/);
  assert.throws(() => call({ timeLimit: 31 }), /求解预算必须大于零且不超过 30 秒/);
  assert.throws(() => call({ matrix: [[0, 1], [1, 0]] }), /必到点之间存在不可达路段或无效代价/);
  assert.throws(() => call({ matrix: [[0, -1, 2], [1, 0, 2], [1, 2, 0]] }), /必到点之间存在不可达路段或无效代价/);
  assert.throws(() => call({ mode: 'unknown' }), /课程模式无效/);
  assert.throws(() => call({ objective: 'unknown' }), /目标函数无效/);
  assert.throws(() => call({ items: [], matrix: randomMatrix(2, 72) }), /每队至少需要一个目标，分担模式下队数不能超过目标数/);
  assert.throws(() => call({ teams: [] }), /每队至少需要一个目标，分担模式下队数不能超过目标数/);
  assert.throws(() => call({ mode: 'split_coverage', items: controls(1), matrix: randomMatrix(3, 73), teams: [{ id: 'A', people: 2, speedMps: 1.4 }, { id: 'B', people: 2, speedMps: 1.4 }] }),
    /每队至少需要一个目标，分担模式下队数不能超过目标数/);
  assert.throws(() => call({ items: [control('x'), control('x')] }), /每队必到点必须是已选择的唯一目标/);
  assert.throws(() => call({ mandatoryIds: ['missing'] }), /每队必到点必须是已选择的唯一目标/);
  assert.throws(() => call({ mode: 'mixed_coverage' }), /混合模式至少需要一个每队必到点/);
  assert.throws(() => call({ mandatoryIds: ['c2'] }), /每队必到点设置仅用于混合模式/);
  assert.throws(() => call({ teams: [{ id: 'A', people: 0, speedMps: 1.4 }] }), /队伍人数或速度无效/);
  assert.throws(() => call({ teams: [{ id: 'A', people: 2, speedMps: 0.1 }] }), /队伍人数或速度无效/);
  assert.throws(() => call({ teams: [{ id: 'A', people: 2, speedMps: 1.4, departureSec: -1 }] }), /出发时间无效/);
  assert.throws(() => call({ items: [control('x', -100, 1), control('y')] }), /打点时长必须非负/);
});
