/** Packed analysis semantics exchanged between the browser pipeline, the
 * baker and the tests.
 *
 * Wire format (used by `web/public/data/maps/<id>/semantics.bin.gz` and by the
 * IndexedDB workspace record after gzip/inflate):
 *
 *   magic "DMSEM1" (6 bytes) | uint16 version=1 | uint32 width | uint32 height
 *   | uint8 maskCount | maskCount × (uint8 nameLength, UTF-8 name)
 *   | maskCount × bit-packed rows (LSB first, ceil(width*height/8) bytes each)
 *
 * Bit-packed masks are 0/1; the last byte of each mask is zero padded.
 */

export interface PackedSemantics {
  width: number;
  height: number;
  names: string[];
  bits: Uint8Array; // concatenated per-mask bit arrays
}

export type SemanticsMaskMap = Record<string, Uint8Array>;

const MAGIC = [0x44, 0x4d, 0x53, 0x45, 0x4d, 0x31]; // DMSEM1
export const SEMANTICS_VERSION = 1;

export function packBits(mask: Uint8Array): Uint8Array {
  const bits = new Uint8Array(Math.ceil(mask.length / 8));
  for (let index = 0; index < mask.length; index++) {
    if (mask[index]) bits[index >> 3] |= 1 << (index & 7);
  }
  return bits;
}

export function unpackBits(bits: Uint8Array, count: number, offset = 0): Uint8Array {
  const mask = new Uint8Array(count);
  for (let index = 0; index < count; index++) {
    mask[index] = (bits[offset + (index >> 3)] >> (index & 7)) & 1;
  }
  return mask;
}

/** Pack named 0/1 masks; names are written in the given order. */
export function packSemantics(width: number, height: number, masks: SemanticsMaskMap, names?: string[]): Uint8Array {
  const order = names ?? Object.keys(masks);
  const count = width * height;
  const encodedNames = order.map(name => new TextEncoder().encode(name));
  const header = new Uint8Array(6 + 2 + 4 + 4 + 1 + encodedNames.reduce((total, name) => total + 1 + name.length, 0));
  const view = new DataView(header.buffer);
  header.set(MAGIC, 0);
  view.setUint16(6, SEMANTICS_VERSION, true);
  view.setUint32(8, width, true);
  view.setUint32(12, height, true);
  view.setUint8(16, order.length);
  let offset = 17;
  for (const name of encodedNames) { header[offset++] = name.length; header.set(name, offset); offset += name.length; }
  const bitsPerMask = Math.ceil(count / 8);
  const packed = new Uint8Array(header.length + bitsPerMask * order.length);
  packed.set(header, 0);
  order.forEach((name, index) => {
    const mask = masks[name];
    if (!mask || mask.length !== count) throw new Error(`semantics mask "${name}" 尺寸不匹配`);
    packed.set(packBits(mask), header.length + index * bitsPerMask);
  });
  return packed;
}

export function unpackSemantics(buffer: ArrayBuffer | Uint8Array): PackedSemantics & { masks: SemanticsMaskMap } {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  for (let index = 0; index < MAGIC.length; index++) {
    if (bytes[index] !== MAGIC[index]) throw new Error('semantics 数据格式无效');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint16(6, true);
  if (version !== SEMANTICS_VERSION) throw new Error(`不支持的 semantics 版本：${version}`);
  const width = view.getUint32(8, true), height = view.getUint32(12, true);
  const count = view.getUint8(16);
  let offset = 17;
  const names: string[] = [];
  for (let index = 0; index < count; index++) {
    const length = bytes[offset++];
    names.push(new TextDecoder().decode(bytes.subarray(offset, offset + length)));
    offset += length;
  }
  const bitsPerMask = Math.ceil((width * height) / 8);
  const bits = bytes.slice(offset, offset + bitsPerMask * count);
  if (bits.length !== bitsPerMask * count) throw new Error('semantics 数据不完整');
  const masks: SemanticsMaskMap = {};
  names.forEach((name, index) => { masks[name] = unpackBits(bits, width * height, index * bitsPerMask); });
  return { width, height, names, bits, masks };
}

/** Ensure required kinds exist; missing kinds are treated like Python's `None`. */
export function pickSemantics(masks: SemanticsMaskMap, kinds: string[]): Record<string, Uint8Array | undefined> {
  const result: Record<string, Uint8Array | undefined> = {};
  for (const kind of kinds) result[kind] = masks[kind];
  return result;
}
