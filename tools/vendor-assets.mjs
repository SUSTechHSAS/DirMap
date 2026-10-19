#!/usr/bin/env node
/**
 * Copy large WASM payloads that are only fetched on demand by the browser.
 *
 * opencv.js (~11 MB) powers the in-browser recognition/registration/terrain
 * pipeline. It is intentionally NOT bundled into the application chunk and NOT
 * committed: this script copies it from node_modules into web/public/vendor/ so
 * both `vite dev` and `vite build` can serve it lazily. The GitHub Pages
 * workflow runs `npm ci` (which triggers `prepare`) so the artifact exists there
 * too. tesseract.js worker/core/model files are loaded from the tesseract.js
 * CDN at OCR time and degrade gracefully when offline.
 */
import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const vendor = join(root, 'web', 'public', 'vendor');
mkdirSync(vendor, { recursive: true });

const assets = [
  {
    from: join(root, 'node_modules', '@techstark', 'opencv-js', 'dist', 'opencv.js'),
    to: join(vendor, 'opencv.js'),
    label: 'OpenCV.js WASM (in-browser CV pipeline)',
    required: true,
  },
];

let copied = 0;
for (const asset of assets) {
  if (!existsSync(asset.from)) {
    if (asset.required) {
      console.error(`[vendor-assets] missing ${asset.from}\nRun \`npm ci\` first.`);
      process.exitCode = 1;
    }
    continue;
  }
  const before = existsSync(asset.to) ? statSync(asset.to).size : -1;
  const after = statSync(asset.from).size;
  if (before === after) continue; // already copied by a previous prepare/build
  copyFileSync(asset.from, asset.to);
  console.log(`[vendor-assets] ${asset.label}: ${(after / 1048576).toFixed(1)} MiB -> ${asset.to}`);
  copied++;
}
if (!copied) console.log('[vendor-assets] vendor assets up to date');
