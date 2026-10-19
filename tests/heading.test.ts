import test from 'node:test';
import assert from 'node:assert/strict';
import { orientationHeading, smoothHeading, compassSample, courseSample, HeadingFilter, NavigationSensors, angleDelta } from '../web/src/heading.ts';

const close = (a: number, b: number) => assert.ok(Math.abs(angleDelta(a, b)) < 1e-8, `${a} vs ${b}`);
test('north crossing uses the short arc', () => {
  close(smoothHeading(359, 1, Math.log(2) * 250), 0);
  close(smoothHeading(1, 359, Math.log(2) * 250), 0);
  assert.throws(() => smoothHeading(NaN, 0, 20));
});
test('absolute W3C orientation across cardinal directions and screen rotation', () => {
  for (const [alpha, expected] of [[0, 0], [90, 270], [180, 180], [270, 90]]) close(orientationHeading(alpha, 0, 0, 0, true)!, expected);
  close(orientationHeading(90, 0, 0, 90, true)!, 0);
  close(orientationHeading(90, 45, 20, 0, true)!, 270);
  assert.equal(orientationHeading(0, 90, 0, 0, true), null);
  assert.equal(orientationHeading(270, 0, 0, 0, false), null);
  assert.equal(orientationHeading(null, 0, 0, 0, true), null);
});
test('iOS magnetic north correction, accuracy and single landscape compensation', () => {
  const event = { webkitCompassHeading: 5, webkitCompassAccuracy: 8 };
  const sample = compassSample(event, 90, 1000, -3.3)!;
  close(sample.degrees, 91.7); assert.equal(sample.reference, 'true');
  assert.equal(compassSample({ ...event, webkitCompassAccuracy: -1 }, 0, 1000, 0), null);
  assert.equal(compassSample({ ...event, webkitCompassAccuracy: 35 }, 0, 1000, 0), null);
});
test('GNSS movement fallback needs stable fresh motion, static drift does not count', () => {
  assert.equal(courseSample({ heading: null, speed: 3, accuracy: 5 }, 1), null);
  assert.equal(courseSample({ heading: 80, speed: .5, accuracy: 4 }, 1), null);
  const filter = new HeadingFilter();
  filter.push(courseSample({ heading: 90, speed: 2, accuracy: 5 }, 1000)!, 1000);
  assert.equal(filter.value(1000), null);
  filter.push(courseSample({ heading: 93, speed: 2, accuracy: 5 }, 2000)!, 2000);
  assert.equal(filter.value(2000)?.source, 'course');
  filter.push({ degrees: 5, timestamp: 2100, source: 'compass', reference: 'true' }, 2100);
  assert.equal(filter.value(2200)?.source, 'compass');
  assert.equal(filter.value(4201), null);
  filter.reset(); assert.equal(filter.value(4201), null);
});
test('reject future and stale sensor events', () => {
  const filter = new HeadingFilter();
  filter.push({ degrees: 1, timestamp: 5000, source: 'compass', reference: 'true' }, 2000);
  filter.push({ degrees: 1, timestamp: 0, source: 'compass', reference: 'true' }, 3000);
  assert.equal(filter.value(3000), null);
});
test('iOS landscape compensation accounts for tilt and rejects a vertical top vector', () => {
  const event = { webkitCompassHeading: 20, webkitCompassAccuracy: 5, beta: 45, gamma: 30 };
  close(compassSample(event, 90, 1000, 0)!.degrees, 20 + orientationHeading(0, 45, 30, 90, true)!);
  assert.equal(compassSample({ ...event, beta: 90 }, 0, 1000, 0), null);
  assert.equal(compassSample({ ...event, webkitCompassHeading: -1 }, 0, 1000, 0), null);
});
test('a late sample cannot replace a fresher heading', () => {
  const filter = new HeadingFilter();
  filter.push({ degrees: 20, timestamp: 2000, source: 'compass', reference: 'true' }, 2000);
  filter.push({ degrees: 80, timestamp: 1800, source: 'compass', reference: 'true' }, 2100);
  assert.equal(filter.value(2100)!.degrees, 20);
});
test('known-bearing calibration needs a fresh compass and adjusts only its offset', () => {
  const sensors = new NavigationSensors();
  assert.throws(() => sensors.calibrate(0, 1000), /罗盘/);
  sensors.filter.push({ degrees: 358, timestamp: 1000, source: 'compass', reference: 'magnetic' }, 1000);
  sensors.calibrate(0, 1000); assert.equal(sensors.offset, 2); assert.equal(sensors.calibrated, true);
  assert.equal(sensors.filter.value(1000), null);
  sensors.filter.push({ degrees: 10, timestamp: 2000, source: 'compass', reference: 'true' }, 2000);
  sensors.calibrate(5, 2000); assert.equal(sensors.offset, -3);
  sensors.clearCalibration(); assert.equal(sensors.offset, 0); assert.equal(sensors.calibrated, false);
});
