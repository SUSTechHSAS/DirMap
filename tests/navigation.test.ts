import test from 'node:test';
import assert from 'node:assert/strict';
import { OffRouteMonitor, pathDistance, remainingPath } from '../web/src/navigation.ts';
import { localScales, type Pixel, type Route } from '../web/src/types.ts';

const origin: Pixel = [113.9304, 22.535];
const scale = localScales(origin);
const point = (x: number, y: number): Pixel => [origin[0] + x / scale[0], origin[1] + y / scale[1]];
const path = [point(0, 0), point(100, 0)];
test('polyline distance uses the closest segment, including duplicate vertices', () => {
  assert.ok(Math.abs(pathDistance(point(50, 10), path) - 10) < .001);
  assert.ok(Math.abs(pathDistance(point(-20, 0), path) - 20) < .001);
  assert.equal(pathDistance(point(0, 0), [path[0], path[0]]), 0);
  assert.equal(pathDistance(point(0, 0), []), Infinity);
});
test('off-route detection needs sustained accurate fresh fixes and a cooldown', () => {
  const monitor = new OffRouteMonitor();
  const fix = (timestamp: number, accuracy = 5, y = 50) => ({ lnglat: point(50, y), accuracy, timestamp });
  assert.equal(monitor.update(fix(1000), path, 1000), false);
  assert.equal(monitor.update(fix(3000), path, 3000), false);
  assert.equal(monitor.update(fix(4000), path, 4000), true);
  assert.equal(monitor.update(fix(5000), path, 5000), false);
  assert.equal(monitor.update(fix(9000), path, 9000), false);
  assert.equal(monitor.update(fix(19000), path, 19000), true);
  monitor.reset();
  assert.equal(monitor.update(fix(1000, 50), path, 1000), false);
  assert.equal(monitor.update(fix(5000, 50), path, 5000), false);
  assert.equal(monitor.update(fix(1000), path, 20000), false);
  assert.equal(monitor.update(fix(50000), path, 20000), false);
  assert.equal(monitor.update(fix(21000, 5, 3), path, 21000), false);
});
test('completed legs no longer count as the remaining route; return leg survives all check-ins', () => {
  const route = { controlIds: ['a', 'b'], geometry: [point(0, 0), point(20, 0), point(20, 0), point(20, 20), point(20, 20), point(0, 0)],
    legs: [{ fromId: 'start', toId: 'a', startIndex: 0, endIndex: 1, distanceM: 20 },
      { fromId: 'a', toId: 'b', startIndex: 2, endIndex: 3, distanceM: 20 },
      { fromId: 'b', toId: 'finish', startIndex: 4, endIndex: 5, distanceM: 28 }] } as Route;
  assert.deepEqual(remainingPath(route, new Set(['a'])), route.geometry.slice(2));
  assert.deepEqual(remainingPath(route, new Set(['a', 'b'])), route.geometry.slice(4));
});
