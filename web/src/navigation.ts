import { localScales, type Pixel, type Route } from './types.ts';

/** Distance to a polyline in local metres, never distance to its vertices. */
export function pathDistance(point: Pixel, path: Pixel[]): number {
  if (!path.length || !point.every(Number.isFinite)) return Infinity;
  const scale = localScales(point);
  const local = path.map(p => [(p[0] - point[0]) * scale[0], (p[1] - point[1]) * scale[1]]);
  let best = Math.hypot(...local[0]);
  for (let i = 1; i < local.length; i++) {
    const [ax, ay] = local[i - 1], [bx, by] = local[i], dx = bx - ax, dy = by - ay;
    const t = dx || dy ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / (dx * dx + dy * dy))) : 0;
    best = Math.min(best, Math.hypot(ax + dx * t, ay + dy * t));
  }
  return best;
}

export function remainingPath(route: Route, completed: Set<string>): Pixel[] {
  const next = route.controlIds.find(id => !completed.has(id)) ?? 'finish';
  const leg = route.legs?.find(leg => leg.toId === next);
  return leg ? route.geometry.slice(leg.startIndex) : route.geometry;
}

export class OffRouteMonitor {
  private firstOutside: number | null = null;
  private lastSample = -Infinity;
  private lastAttempt = -Infinity;

  update(position: { lnglat: Pixel; accuracy: number; timestamp: number }, path: Pixel[], now: number): boolean {
    if (!Number.isFinite(position.accuracy) || position.accuracy < 0 || position.accuracy > 20 ||
        !Number.isFinite(position.timestamp) || position.timestamp > now + 250 || now - position.timestamp > 10000 ||
        position.timestamp <= this.lastSample || !path.length) { this.firstOutside = null; return false; }
    this.lastSample = position.timestamp;
    const outside = pathDistance(position.lnglat, path) > Math.max(15, position.accuracy * 2);
    if (!outside) { this.firstOutside = null; return false; }
    this.firstOutside ??= now;
    if (now - this.firstOutside < 3000 || now - this.lastAttempt < 15000) return false;
    this.lastAttempt = now;
    this.firstOutside = null;
    return true;
  }

  reset() { this.firstOutside = null; this.lastSample = this.lastAttempt = -Infinity; }
}
