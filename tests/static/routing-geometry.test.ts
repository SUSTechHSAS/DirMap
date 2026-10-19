/** Geometry/frame/mesh unit tests: frame conversions, mesh forward/inverse and
 * the bulk path against Python `Registration.forward` fixtures. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Mesh, frameLngLat, frameLocal, localScales } from '../../web/src/geometry.ts';
import { localScales as baseLocalScales, type Pixel } from '../../web/src/types.ts';
import { readJSON } from './helpers.ts';

interface MeshSample { pixel: Pixel; local: Pixel | null; inside: boolean; affineLocal: Pixel }

const reference = readJSON<Record<string, { meshSamples: MeshSample[] }>>('tests/static/fixtures/python-reference.json');
const origin: Pixel = [113.9304, 22.535];

test('local scales match the existing shared formula', () => {
  const scales = localScales(origin);
  const base = baseLocalScales(origin);
  assert.equal(scales[0], base[0]);
  assert.equal(scales[1], base[1]);
});

test('frame conversions round-trip in both directions', () => {
  const samples: Pixel[] = [[113.9304, 22.535], [113.9321, 22.5341], [113.9288, 22.5367]];
  for (const lnglat of samples) {
    const local = frameLocal(lnglat, origin);
    const back = frameLngLat(local, origin);
    assert.ok(Math.abs(back[0] - lnglat[0]) < 1e-14);
    assert.ok(Math.abs(back[1] - lnglat[1]) < 1e-14);
  }
  // 100 m north must be ~100 m in the local frame.
  const hundredNorth = frameLngLat([frameLocal([origin[0], origin[1] + 0.001], origin)[0], frameLocal([origin[0], origin[1] + 0.001], origin)[1]], origin);
  assert.ok(Math.abs(hundredNorth[1] - (origin[1] + 0.001)) < 1e-14);
});

test('mesh forward matches the Python registration for control/interior points', async () => {
  for (const assetId of ['szu-1', 'szu-2', 'szu-3']) {
    const state = readJSON<{ registration: ConstructorParameters<typeof Mesh>[0] }>(`web/public/data/maps/${assetId}/state.json`);
    const mesh = new Mesh(state.registration);
    let insideCount = 0;
    for (const sample of reference[assetId].meshSamples) {
      const forward = mesh.forward(sample.pixel);
      assert.equal(forward.inside, sample.inside, `${assetId} inside flag for ${sample.pixel}`);
      if (sample.inside) {
        insideCount++;
        assert.ok(sample.local);
        assert.ok(Math.abs(forward.local[0] - sample.local[0]) < 1e-7, `${assetId} local x ${forward.local[0]} vs ${sample.local[0]}`);
        assert.ok(Math.abs(forward.local[1] - sample.local[1]) < 1e-7, `${assetId} local y ${forward.local[1]} vs ${sample.local[1]}`);
        const inverse = mesh.inverse(forward.local);
        assert.ok(inverse);
        assert.ok(Math.abs(inverse[0] - sample.pixel[0]) < 1e-6, `${assetId} inverse x ${inverse[0]} vs ${sample.pixel[0]}`);
        assert.ok(Math.abs(inverse[1] - sample.pixel[1]) < 1e-6, `${assetId} inverse y ${inverse[1]} vs ${sample.pixel[1]}`);
        const lnglat = mesh.pixelToLngLat(sample.pixel, origin);
        assert.ok(lnglat);
        const back = mesh.lngLatToPixel(lnglat, origin);
        assert.ok(back);
        assert.ok(Math.abs(back[0] - sample.pixel[0]) < 1e-5);
        assert.ok(Math.abs(back[1] - sample.pixel[1]) < 1e-5);
      } else {
        assert.ok(sample.affineLocal);
        assert.ok(Math.abs(forward.local[0] - sample.affineLocal[0]) < 1e-9);
        assert.ok(Math.abs(forward.local[1] - sample.affineLocal[1]) < 1e-9);
        assert.equal(mesh.pixelToLngLat(sample.pixel, origin), null);
      }
    }
    assert.ok(insideCount >= 10, `${assetId}: only ${insideCount} inside samples`);
  }
});

test('forwardMany agrees with forward point-by-point and stays inside', async () => {
  const state = readJSON<{ registration: ConstructorParameters<typeof Mesh>[0] }>('web/public/data/maps/szu-2/state.json');
  const mesh = new Mesh(state.registration);
  const pixels: Pixel[] = reference['szu-2'].meshSamples.map(sample => sample.pixel);
  const flat = new Float64Array(pixels.length * 2);
  pixels.forEach((pixel, index) => { flat[index * 2] = pixel[0]; flat[index * 2 + 1] = pixel[1]; });
  const bulk = mesh.forwardMany(flat);
  for (let index = 0; index < pixels.length; index++) {
    const single = mesh.forward(pixels[index]);
    assert.equal(Boolean(bulk.inside[index]), single.inside);
    assert.ok(Math.abs(bulk.local[index * 2] - single.local[0]) < 1e-12);
    assert.ok(Math.abs(bulk.local[index * 2 + 1] - single.local[1]) < 1e-12);
  }
});

test('mesh without registration degrades to affine/outside and null inverse', () => {
  const mesh = new Mesh(null);
  const forward = mesh.forward([12, 34]);
  assert.deepEqual(forward.local, [12, 34]);
  assert.equal(forward.inside, false);
  assert.equal(mesh.inverse([0, 0]), null);
  assert.equal(mesh.pixelToLngLat([12, 34], origin), null);
  assert.equal(mesh.lngLatToPixel(origin, origin), null);
});

test('mesh rejects an invalid affine matrix', () => {
  assert.throws(() => new Mesh({ affine: [[0, 0, 0], [0, 0, 0]], source: [], target: [], triangles: [] }), /配准矩阵无效/);
});
