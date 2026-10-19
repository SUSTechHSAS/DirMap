/** Browser port of `dirmap/optimizer.py::solve` and its exact-DP helpers.
 *
 * CP-SAT is deliberately not ported: all_visit/split_coverage/mixed_coverage
 * within the Python exact branches reproduce the Held-Karp DP exactly, every
 * other case falls back to the deterministic greedy + 2-opt + relocation
 * heuristic and is honestly labelled `FEASIBLE`.
 */
import type { Control } from './types.ts';

export interface OptimizerTeam {
  id: string;
  people: number;
  speedMps: number;
  departureSec?: number;
}

export interface OptimizerRoute {
  teamId: string;
  people: number;
  order: number[];
  controlIds: string[];
  travelSec: number;
  serviceSec: number;
  durationSec: number;
  departureSec: number;
  finishSec: number;
}

export interface OptimizerCoverage {
  mode: string;
  mandatoryControlIds: string[];
  mandatoryOrder: boolean;
  sharedControlIds: string[];
}

export interface OptimizerProof {
  algorithm: string;
  globalOptimalityProven: boolean;
  objective: string;
  objectiveValue: number[];
  bestBound: number[] | null;
  relativeGap: number | null;
  elapsedSec: number;
  scope: string;
}

export interface OptimizerResult {
  status: 'OPTIMAL' | 'FEASIBLE';
  routes: OptimizerRoute[];
  coverage: OptimizerCoverage;
  proof: OptimizerProof;
}

export type SolveMatrix = Float64Array | readonly (readonly number[])[];

const MODES = ['all_visit', 'split_coverage', 'fixed_order', 'mixed_coverage'];
const OBJECTIVES = ['min_max_finish_then_sum', 'min_total_time', 'min_distance'];
const SOLVE_SCOPE = '当前图上的静态通行代价、打点时长和指定出发时间；不包含动态拥堵。';

class DeadlineReached extends Error {}

const nowSeconds = (): number => performance.now() / 1000;

function readMatrix(value: SolveMatrix): { matrix: Float64Array; size: number } {
  if (value instanceof Float64Array) {
    return { matrix: value, size: Math.round(Math.sqrt(value.length)) };
  }
  const rows = value.length;
  const matrix = new Float64Array(rows * rows);
  for (let row = 0; row < rows; row++) {
    const source = value[row];
    if (!source || source.length !== rows) throw new Error('必到点之间存在不可达路段或无效代价');
    for (let column = 0; column < rows; column++) matrix[row * rows + column] = source[column];
  }
  return { matrix, size: rows };
}

export function routeCost(matrix: Float64Array, size: number, sequence: readonly number[]): number {
  let total = 0;
  for (let i = 0; i + 1 < sequence.length; i++) total += matrix[sequence[i] * size + sequence[i + 1]];
  return total;
}

function tupleLess(left: readonly number[], right: readonly number[]): boolean {
  for (let i = 0; i < left.length; i++) {
    if (left[i] !== right[i]) return left[i] < right[i];
  }
  return false;
}

function orderedMatches(sequence: readonly number[], orderedNodes: Int32Array): boolean {
  if (!orderedNodes.length) return true;
  let position = 0;
  for (const node of sequence) {
    if (position < orderedNodes.length && node === orderedNodes[position]) position++;
    else if (orderedNodes.includes(node)) return false;
  }
  return position === orderedNodes.length;
}

interface SubsetTours {
  costs: Float64Array;
  route: (mask: number) => number[];
}

/** Held-Karp over every subset with distinct fixed start/finish. */
export function subsetTours(matrix: Float64Array, size: number, deadline = Infinity, orderedNodes: Int32Array = new Int32Array(0)): SubsetTours {
  const n = size - 2;
  const maskCount = 1 << n;
  const values = new Float64Array(maskCount * n).fill(Infinity);
  const parent = new Int16Array(maskCount * n).fill(-1);
  const costs = new Float64Array(maskCount).fill(Infinity);
  const end = new Int16Array(maskCount).fill(-1);
  costs[0] = matrix[size - 1];
  const requiredBefore = new Int32Array(n);
  let before = 0;
  for (const node of orderedNodes) {
    requiredBefore[node - 1] = before;
    before |= 1 << (node - 1);
  }
  for (let last = 0; last < n; last++) {
    if (!requiredBefore[last]) values[(1 << last) * n + last] = matrix[last + 1];
  }
  for (let mask = 1; mask < maskCount; mask++) {
    if (mask % 128 === 0 && nowSeconds() >= deadline) throw new DeadlineReached('精确动态规划达到时间预算');
    for (let last = 0; last < n; last++) {
      if (!(mask & (1 << last))) continue;
      const previousMask = mask ^ (1 << last);
      if ((previousMask & requiredBefore[last]) !== requiredBefore[last]) continue;
      if (previousMask) {
        const base = previousMask * n;
        let bestValue = Infinity;
        let bestPrevious = -1;
        for (let i = 0; i < n; i++) {
          const option = values[base + i] + matrix[(i + 1) * size + (last + 1)];
          if (option < bestValue) {
            bestValue = option;
            bestPrevious = i;
          }
        }
        if (bestPrevious < 0) bestPrevious = 0;
        values[mask * n + last] = bestValue;
        parent[mask * n + last] = bestPrevious;
      }
      const total = values[mask * n + last] + matrix[(last + 1) * size + (size - 1)];
      if (total < costs[mask]) {
        costs[mask] = total;
        end[mask] = last;
      }
    }
  }
  const route = (mask: number): number[] => {
    const sequence = [n + 1];
    let remaining = mask;
    let last = end[mask];
    while (remaining) {
      if (last < 0) throw new Error('必到点不连通');
      sequence.push(last + 1);
      const previous = parent[remaining * n + last];
      remaining ^= 1 << last;
      last = previous;
    }
    sequence.reverse();
    return [0, ...sequence];
  };
  return { costs, route };
}

/** Two-stage DP assignment of shared subsets to teams (exact branches). */
export function assignSubsets(
  matrix: Float64Array, size: number, teams: readonly OptimizerTeam[], service: readonly (readonly number[])[],
  objective: string, deadline: number, mandatoryNodes: Int32Array = new Int32Array(0),
  orderedNodes: Int32Array = new Int32Array(0),
): number[][] {
  const { costs: allCosts, route: reconstruct } = subsetTours(matrix, size, deadline, orderedNodes);
  const n = size - 2;
  const k = teams.length;
  const subsetService = new Float64Array(k * allCosts.length);
  for (let mask = 1; mask < allCosts.length; mask++) {
    const bit = mask & -mask;
    const nodeIndex = Math.log2(bit) | 0;
    for (let team = 0; team < k; team++) {
      subsetService[team * allCosts.length + mask] = subsetService[team * allCosts.length + (mask ^ bit)] + service[team][nodeIndex];
    }
  }
  let mandatoryMask = 0;
  const mandatorySet = new Set<number>();
  for (const node of mandatoryNodes) {
    mandatoryMask |= 1 << (node - 1);
    mandatorySet.add(node);
  }
  const shared: number[] = [];
  for (let node = 1; node <= n; node++) if (!mandatorySet.has(node)) shared.push(node);
  const sizeShared = 1 << shared.length;
  const expanded = new Int32Array(sizeShared);
  expanded[0] = mandatoryMask;
  for (let mask = 1; mask < sizeShared; mask++) {
    const bit = mask & -mask;
    const nodeIndex = Math.log2(bit) | 0;
    expanded[mask] = expanded[mask ^ bit] | (1 << (shared[nodeIndex] - 1));
  }
  const costs = new Float64Array(sizeShared);
  for (let mask = 0; mask < sizeShared; mask++) costs[mask] = allCosts[expanded[mask]];
  const durations = new Float64Array(k * sizeShared);
  for (let team = 0; team < k; team++) {
    for (let mask = 0; mask < sizeShared; mask++) {
      durations[team * sizeShared + mask] = costs[mask] / teams[team].speedMps
        + subsetService[team * allCosts.length + expanded[mask]]
        + (teams[team].departureSec ?? 0);
    }
  }
  const allowEmpty = mandatoryNodes.length > 0;
  let limit = Infinity;
  if (objective === 'min_max_finish_then_sum') {
    const primary = new Float64Array((k + 1) * sizeShared).fill(Infinity);
    primary[0] = 0;
    for (let team = 0; team < k; team++) {
      for (let mask = allowEmpty ? 0 : 1; mask < sizeShared; mask++) {
        if (mask % 64 === 0 && nowSeconds() >= deadline) throw new DeadlineReached('分队动态规划达到时间预算');
        let sub = mask;
        for (;;) {
          if (sub !== 0 || allowEmpty) {
            const left = primary[team * sizeShared + (mask ^ sub)];
            const right = durations[team * sizeShared + sub];
            const value = left > right ? left : right;
            if (value < primary[(team + 1) * sizeShared + mask]) primary[(team + 1) * sizeShared + mask] = value;
          }
          if (sub === 0) break;
          sub = (sub - 1) & mask;
        }
      }
    }
    limit = primary[k * sizeShared + (sizeShared - 1)];
  }
  const dp = new Float64Array((k + 1) * sizeShared).fill(Infinity);
  const previous = new Int32Array((k + 1) * sizeShared);
  dp[0] = 0;
  for (let team = 0; team < k; team++) {
    for (let mask = allowEmpty ? 0 : 1; mask < sizeShared; mask++) {
      if (mask % 64 === 0 && nowSeconds() >= deadline) throw new DeadlineReached('分队动态规划达到时间预算');
      let sub = mask;
      for (;;) {
        const duration = durations[team * sizeShared + sub];
        if ((sub !== 0 || allowEmpty) && duration <= limit + 1e-8) {
          const add = objective === 'min_distance' ? costs[sub] : duration;
          const value = dp[team * sizeShared + (mask ^ sub)] + add;
          if (value < dp[(team + 1) * sizeShared + mask]) {
            dp[(team + 1) * sizeShared + mask] = value;
            previous[(team + 1) * sizeShared + mask] = sub;
          }
        }
        if (sub === 0) break;
        sub = (sub - 1) & mask;
      }
    }
  }
  if (!Number.isFinite(dp[k * sizeShared + (sizeShared - 1)])) throw new Error('没有满足分担规则的路线');
  const masks: number[] = [];
  let remaining = sizeShared - 1;
  for (let team = k; team > 0; team--) {
    const mask = previous[team * sizeShared + remaining];
    masks.push(mask);
    remaining ^= mask;
  }
  masks.reverse();
  return masks.map(mask => reconstruct(expanded[mask]));
}

/** Iterated 2-opt reversal; works for asymmetric costs like the Python helper. */
export function improveOrder(
  matrix: Float64Array, size: number, sequence: readonly number[], deadline: number,
  orderedNodes: Int32Array = new Int32Array(0),
): number[] {
  let best = sequence.slice();
  let cost = routeCost(matrix, size, best);
  let changed = true;
  while (changed && nowSeconds() < deadline) {
    changed = false;
    for (let i = 1; i < best.length - 2; i++) {
      for (let j = i + 1; j < best.length - 1; j++) {
        const candidate = [...best.slice(0, i), ...best.slice(i, j + 1).reverse(), ...best.slice(j + 1)];
        if (!orderedMatches(candidate, orderedNodes)) continue;
        const value = routeCost(matrix, size, candidate);
        if (value < cost - 1e-8) {
          best = candidate;
          cost = value;
          changed = true;
        }
      }
      if (nowSeconds() >= deadline) break;
    }
  }
  return best;
}

/** Greedy nearest-target insertion followed by 2-opt. */
export function greedyRoute(matrix: Float64Array, size: number, targets: readonly number[], deadline: number): number[] {
  const remaining = new Set(targets);
  const sequence = [0];
  while (remaining.size) {
    let bestNode = -1;
    let bestCost = Infinity;
    for (const node of remaining) {
      const cost = matrix[sequence[sequence.length - 1] * size + node];
      if (cost < bestCost || (cost === bestCost && (bestNode < 0 || node < bestNode))) {
        bestCost = cost;
        bestNode = node;
      }
    }
    remaining.delete(bestNode);
    sequence.push(bestNode);
  }
  sequence.push(size - 1);
  return improveOrder(matrix, size, sequence, deadline);
}

/** Compare every insertion position and keep the best lexicographic total. */
function insertPending(
  start: number[][], pending: readonly number[], totals: (sequences: number[][]) => number[],
): number[][] {
  let sequences = start;
  for (const node of pending) {
    let bestSequences = sequences;
    let bestValue: number[] | null = null;
    for (let team = 0; team < sequences.length; team++) {
      for (let position = 1; position < sequences[team].length; position++) {
        const candidate = sequences.map(sequence => sequence.slice());
        candidate[team].splice(position, 0, node);
        const value = totals(candidate);
        if (bestValue === null || tupleLess(value, bestValue)) {
          bestValue = value;
          bestSequences = candidate;
        }
      }
    }
    sequences = bestSequences;
  }
  return sequences;
}

interface HeuristicOptions {
  matrix: Float64Array;
  size: number;
  teams: readonly OptimizerTeam[];
  mode: string;
  n: number;
  k: number;
  mandatorySet: Set<number>;
  mandatoryNodes: Int32Array;
  orderedNodes: Int32Array;
  deadline: number;
  timeLimit: number;
  started: number;
  totals: (sequences: readonly number[][]) => number[];
}

/** Greedy insertion + 2-opt + cross-team relocation (the Python fallback). */
function heuristicSequences(options: HeuristicOptions): number[][] {
  const { matrix, size, mode, n, k, mandatorySet, mandatoryNodes, orderedNodes, deadline, timeLimit, started, totals } = options;
  const heuristicDeadline = Math.min(deadline, started + Math.max(0.03, timeLimit * 0.2));
  const targets: number[] = [];
  let sequences: number[][];
  if (mode === 'all_visit') {
    const sequence = greedyRoute(matrix, size, Array.from({ length: n }, (_, index) => index + 1), heuristicDeadline);
    return options.teams.map(() => sequence.slice());
  }
  for (let node = 1; node <= n; node++) if (!mandatorySet.has(node)) targets.push(node);
  targets.sort((left, right) => matrix[right] - matrix[left] || left - right);
  let pending: number[];
  if (mandatoryNodes.length) {
    const ordered = orderedNodes.length
      ? Array.from(orderedNodes)
      : greedyRoute(matrix, size, Array.from(mandatoryNodes), heuristicDeadline).slice(1, -1);
    const base = [0, ...ordered, n + 1];
    sequences = options.teams.map(() => base.slice());
    pending = targets;
  } else {
    sequences = [];
    for (let index = 0; index < k; index++) sequences.push([0, targets[index], n + 1]);
    pending = targets.slice(k);
  }
  sequences = insertPending(sequences, pending, totals);
  sequences = sequences.map(sequence => improveOrder(matrix, size, sequence, heuristicDeadline, orderedNodes));
  let improved = true;
  while (improved && nowSeconds() < heuristicDeadline) {
    improved = false;
    let old = totals(sequences);
    let done = false;
    for (let a = 0; a < k && !done; a++) {
      for (let b = 0; b < k && !done; b++) {
        if (a === b || sequences[a].length <= 3) continue;
        for (let i = 1; i < sequences[a].length - 1 && !done; i++) {
          if (mandatorySet.has(sequences[a][i])) continue;
          for (let j = 1; j < sequences[b].length; j++) {
            const candidate = sequences.map(sequence => sequence.slice());
            const [moved] = candidate[a].splice(i, 1);
            candidate[b].splice(j, 0, moved);
            const value = totals(candidate);
            if (tupleLess(value, old)) {
              sequences = candidate;
              old = value;
              improved = true;
              done = true;
              break;
            }
          }
          if (nowSeconds() >= heuristicDeadline) done = true;
        }
      }
    }
  }
  return sequences;
}

/**
 * Solve the course exactly like `dirmap.optimizer.solve`.
 */
export function solve(
  matrixValue: SolveMatrix, controls: readonly Control[], teams: readonly OptimizerTeam[],
  mode = 'all_visit', objective = 'min_max_finish_then_sum', timeLimit = 5,
  mandatoryIds?: readonly string[], mandatoryOrder = false,
): OptimizerResult {
  const started = nowSeconds();
  if (!Number.isFinite(timeLimit) || !(timeLimit > 0 && timeLimit <= 30)) throw new Error('求解预算必须大于零且不超过 30 秒');
  const deadline = started + timeLimit;
  const { matrix, size } = readMatrix(matrixValue);
  const n = controls.length;
  const k = teams.length;
  if (size !== n + 2) throw new Error('必到点之间存在不可达路段或无效代价');
  for (let i = 0; i < matrix.length; i++) {
    if (!(matrix[i] >= 0) || !Number.isFinite(matrix[i])) throw new Error('必到点之间存在不可达路段或无效代价');
  }
  if (!MODES.includes(mode)) throw new Error('课程模式无效');
  if (!OBJECTIVES.includes(objective)) throw new Error('目标函数无效');
  if (!n || !k || (mode === 'split_coverage' && k > n)) throw new Error('每队至少需要一个目标，分担模式下队数不能超过目标数');
  const mandatory = Array.isArray(mandatoryIds) ? mandatoryIds.slice() : [];
  const indexes = new Map<string, number>();
  for (let i = 0; i < controls.length; i++) indexes.set(controls[i].id, i + 1);
  if (indexes.size !== n || new Set(mandatory).size !== mandatory.length || mandatory.some(id => !indexes.has(id))) {
    throw new Error('每队必到点必须是已选择的唯一目标');
  }
  if (mode === 'mixed_coverage' && !mandatory.length) throw new Error('混合模式至少需要一个每队必到点');
  if (mode !== 'mixed_coverage' && (mandatory.length || mandatoryOrder)) throw new Error('每队必到点设置仅用于混合模式');
  const mandatoryNodes = Int32Array.from(mandatory.map(id => indexes.get(id) as number));
  const orderedNodes = mandatoryOrder ? mandatoryNodes : new Int32Array(0);
  for (const team of teams) {
    if (!Number.isInteger(team.people) || team.people < 1 || !Number.isFinite(team.speedMps) || !(team.speedMps >= 0.2 && team.speedMps <= 10)) {
      throw new Error('队伍人数或速度无效');
    }
    if (!Number.isFinite(team.departureSec ?? 0) || (team.departureSec ?? 0) < 0) throw new Error('出发时间无效');
  }
  const service: number[][] = [];
  for (const team of teams) {
    const row: number[] = [];
    for (const control of controls) {
      row.push((control.serviceBaseSec ?? 5) + team.people * (control.servicePerPersonSec ?? 1));
    }
    service.push(row);
  }
  for (const row of service) {
    for (const value of row) if (!Number.isFinite(value) || value < 0) throw new Error('打点时长必须非负');
  }
  const mandatorySet = new Set<number>(Array.from(mandatoryNodes));
  const totals = (sequences: readonly number[][]): number[] => {
    const travel = sequences.map(sequence => routeCost(matrix, size, sequence));
    const times = sequences.map((sequence, index) => {
      let serviceTime = 0;
      for (let position = 1; position < sequence.length - 1; position++) serviceTime += service[index][sequence[position] - 1];
      return travel[index] / teams[index].speedMps + serviceTime + (teams[index].departureSec ?? 0);
    });
    if (objective === 'min_max_finish_then_sum') return [Math.max(...times), times.reduce((left, right) => left + right, 0)];
    if (objective === 'min_distance') return [travel.reduce((left, right) => left + right, 0)];
    return [times.reduce((left, right) => left + right, 0)];
  };

  let proven = true;
  let sequences: number[][] | null = null;
  let algorithm = '';
  if (mode === 'fixed_order') {
    sequences = teams.map(() => Array.from({ length: n + 2 }, (_, index) => index));
    algorithm = 'dijkstra_fixed_order';
  } else if (mode === 'all_visit' && n <= 13) {
    try {
      const tours = subsetTours(matrix, size, deadline);
      const full = (1 << n) - 1;
      sequences = teams.map(() => tours.route(full));
      algorithm = 'held_karp';
    } catch (error) {
      if (!(error instanceof DeadlineReached)) throw error;
    }
  } else if (mode === 'split_coverage' && n <= 10) {
    try {
      sequences = assignSubsets(matrix, size, teams, service, objective, deadline);
      algorithm = 'held_karp_subset_assignment_two_stage';
    } catch (error) {
      if (!(error instanceof DeadlineReached)) throw error;
    }
  } else if (mode === 'mixed_coverage' && n <= 13 && n - mandatoryNodes.length <= 9) {
    try {
      sequences = assignSubsets(matrix, size, teams, service, objective, deadline, mandatoryNodes, orderedNodes);
      algorithm = 'held_karp_mandatory_shared_two_stage';
    } catch (error) {
      if (!(error instanceof DeadlineReached)) throw error;
    }
  }
  if (sequences === null) {
    proven = false;
    algorithm = 'insertion_2opt_relocation';
    sequences = heuristicSequences({
      matrix, size, teams: teams as OptimizerTeam[], mode, n, k, mandatorySet, mandatoryNodes,
      orderedNodes, deadline, timeLimit, started, totals,
    });
  }
  const chosen = sequences as number[][];
  if (mode === 'split_coverage' || mode === 'mixed_coverage') {
    const actual = new Map<number, number>();
    for (const route of chosen) {
      for (let position = 1; position < route.length - 1; position++) {
        const node = route[position];
        actual.set(node, (actual.get(node) ?? 0) + 1);
      }
    }
    let valid = actual.size === n;
    for (let node = 1; node <= n; node++) {
      const expected = mandatorySet.has(node) ? k : 1;
      if ((actual.get(node) ?? 0) !== expected) valid = false;
    }
    for (const route of chosen) {
      const inner = route.slice(1, -1);
      if (inner.length !== new Set(inner).size) valid = false;
      for (const node of mandatorySet) if (!route.includes(node)) valid = false;
    }
    if (!valid) throw new Error('路线未满足每队必到与全班分担规则');
    if (orderedNodes.length && chosen.some(route => !orderedMatches(route, orderedNodes))) {
      throw new Error('路线未满足必到点顺序');
    }
  }
  const routes: OptimizerRoute[] = [];
  for (let index = 0; index < teams.length; index++) {
    const team = teams[index];
    const sequence = chosen[index];
    const travel = routeCost(matrix, size, sequence) / team.speedMps;
    let serviceTime = 0;
    for (let position = 1; position < sequence.length - 1; position++) serviceTime += service[index][sequence[position] - 1];
    routes.push({
      teamId: team.id,
      people: team.people,
      order: sequence.slice(),
      controlIds: sequence.slice(1, -1).map(node => controls[node - 1].id),
      travelSec: travel,
      serviceSec: serviceTime,
      durationSec: travel + serviceTime,
      departureSec: team.departureSec ?? 0,
      finishSec: travel + serviceTime + (team.departureSec ?? 0),
    });
  }
  const value = totals(chosen);
  const sharedControlIds = mode === 'split_coverage' || mode === 'mixed_coverage'
    ? controls.filter(control => !mandatory.includes(control.id)).map(control => control.id)
    : [];
  return {
    status: proven ? 'OPTIMAL' : 'FEASIBLE',
    routes,
    coverage: { mode, mandatoryControlIds: mandatory, mandatoryOrder, sharedControlIds },
    proof: {
      algorithm,
      globalOptimalityProven: proven,
      objective,
      objectiveValue: value,
      bestBound: proven ? value.slice() : null,
      relativeGap: proven ? 0 : null,
      elapsedSec: nowSeconds() - started,
      scope: SOLVE_SCOPE,
    },
  };
}
