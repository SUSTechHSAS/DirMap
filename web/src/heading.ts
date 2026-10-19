/** Device angles and motion course are distinct sources, never interchanged. */
export type HeadingSource = 'compass' | 'course';
export type HeadingSample = { degrees: number; timestamp: number; source: HeadingSource; reference: 'true' | 'magnetic' };
export const normalize = (degrees: number) => (degrees % 360 + 360) % 360;
export const angleDelta = (from: number, to: number) => (to - from + 540) % 360 - 180;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

export function smoothHeading(previous: number, next: number, elapsedMs: number, tauMs = 250): number {
  if (![previous, next, elapsedMs, tauMs].every(finite) || elapsedMs < 0 || tauMs <= 0) throw new Error('Invalid heading');
  return normalize(previous + (1 - Math.exp(-elapsedMs / tauMs)) * angleDelta(previous, next));
}

export function orientationHeading(alpha: unknown, beta: unknown, gamma: unknown, screenAngle = 0, absolute = false): number | null {
  if (!absolute || !finite(alpha) || !finite(beta) || !finite(gamma) || !finite(screenAngle)) return null;
  const [a, b, g, s] = [alpha, beta, gamma, screenAngle].map(value => value * Math.PI / 180);
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), cg = Math.cos(g), sg = Math.sin(g);
  // W3C Rz(alpha) Rx(beta) Ry(gamma), projected screen-top vector.
  // Screen compensation occurs here exactly once (also in the iOS branch).
  const x = Math.sin(s), y = Math.cos(s);
  const east = (ca * cg - sa * sb * sg) * x - sa * cb * y;
  const north = (sa * cg + ca * sb * sg) * x + ca * cb * y;
  if (Math.hypot(east, north) < .25) return null;
  return normalize(Math.atan2(east, north) * 180 / Math.PI);
}

export function compassSample(event: { alpha?: unknown; beta?: unknown; gamma?: unknown; absolute?: boolean; type?: string; webkitCompassHeading?: unknown; webkitCompassAccuracy?: unknown }, screenAngle: number, timestamp: number, declination: number | null): HeadingSample | null {
  if (!finite(screenAngle) || !finite(timestamp) || (declination !== null && !finite(declination))) return null;
  if (finite(event.webkitCompassHeading)) {
    if (event.webkitCompassHeading < 0 || event.webkitCompassHeading >= 360 || !finite(event.webkitCompassAccuracy) || event.webkitCompassAccuracy < 0 || event.webkitCompassAccuracy > 25) return null;
    let compensation = screenAngle;
    if (finite(event.beta) && finite(event.gamma)) {
      const portrait = orientationHeading(0, event.beta, event.gamma, 0, true);
      const rotated = orientationHeading(0, event.beta, event.gamma, screenAngle, true);
      if (portrait === null || rotated === null) return null;
      compensation = angleDelta(portrait, rotated);
    }
    return { degrees: normalize(event.webkitCompassHeading + compensation + (declination ?? 0)), timestamp, source: 'compass', reference: declination === null ? 'magnetic' : 'true' };
  }
  const degrees = orientationHeading(event.alpha, event.beta, event.gamma, screenAngle, event.absolute === true);
  // Absolute orientation's north semantics vary across browser/sensor stacks.
  // Keep the reference visible until a known-north calibration is supplied.
  return degrees === null ? null : { degrees, timestamp, source: 'compass', reference: 'magnetic' };
}

export function courseSample(coords: { heading?: unknown; speed?: unknown; accuracy?: unknown }, timestamp: number): HeadingSample | null {
  if (!finite(coords.heading) || !finite(coords.speed) || !finite(coords.accuracy) || coords.speed < 1.2 || coords.accuracy < 0 || coords.accuracy > 15) return null;
  return { degrees: normalize(coords.heading), timestamp, source: 'course', reference: 'true' };
}

export class HeadingFilter {
  compass: HeadingSample | null = null;
  course: HeadingSample | null = null;
  private previousCourse: HeadingSample | null = null;
  private lastOutput: number | null = null;
  private lastTime: number | null = null;

  push(sample: HeadingSample, now: number) {
    if (!finite(sample.degrees) || !finite(sample.timestamp) || sample.timestamp > now + 250 || now - sample.timestamp > 2000) return;
    if (sample.source === 'compass') {
      if (this.compass && sample.timestamp <= this.compass.timestamp) return;
      this.compass = sample;
    }
    else {
      const previous = this.previousCourse;
      if (previous && sample.timestamp <= previous.timestamp) return;
      if (previous && sample.timestamp > previous.timestamp && sample.timestamp - previous.timestamp < 3000 && Math.abs(angleDelta(previous.degrees, sample.degrees)) < 35) this.course = sample;
      else this.course = null;
      this.previousCourse = sample;
    }
  }

  value(now: number): HeadingSample | null {
    const sample = this.compass && now - this.compass.timestamp <= 2000 ? this.compass :
      this.course && now - this.course.timestamp <= 2000 ? this.course : null;
    if (!sample) { this.lastTime = null; return null; }
    this.lastOutput = this.lastTime === null || this.lastOutput === null ? sample.degrees : smoothHeading(this.lastOutput, sample.degrees, now - this.lastTime);
    this.lastTime = now;
    return { ...sample, degrees: this.lastOutput };
  }

  reset() { this.compass = this.course = this.previousCourse = null; this.lastOutput = this.lastTime = null; }
}

type IOSOrientation = typeof DeviceOrientationEvent & { requestPermission?: () => Promise<string> };
export class NavigationSensors {
  filter = new HeadingFilter();
  private watch: number | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private startedAt = 0;
  active = false;
  declination: number | null = null;
  offset = 0;
  calibrated = false;
  private permissionGranted = false;
  onHeading: (sample: HeadingSample | null) => void = () => {};
  onPosition: (position: GeolocationPosition) => void = () => {};
  onStatus: (message: string) => void = () => {};

  private orientation = (event: DeviceOrientationEvent) => {
    if (document.hidden) return;
    const timestamp = event.timeStamp > 1e12 ? event.timeStamp : performance.timeOrigin + event.timeStamp;
    if (timestamp < this.startedAt) return;
    const angle = screen.orientation?.angle ?? Number((window as Window & { orientation?: number }).orientation ?? 0);
    const sample = compassSample(event, angle, timestamp, this.declination);
    if (sample) {
      sample.degrees = normalize(sample.degrees + this.offset);
      if (this.calibrated) sample.reference = 'true';
      this.filter.push(sample, Date.now());
    }
  };
  private visibility = () => {
    this.filter.reset();
    this.startedAt = Date.now();
    if (document.hidden) {
      this.clearWatch();
      this.onHeading(null);
    } else if (this.active) this.subscribeLocation();
  };

  async start() {
    if (this.active) return;
    if (!window.isSecureContext) { this.onStatus('手机定位与罗盘需要 HTTPS 连接'); return; }
    // iOS permission must run directly within the start-button gesture.
    const Orientation = window.DeviceOrientationEvent as IOSOrientation | undefined;
    if (Orientation?.requestPermission && !this.permissionGranted) {
      try { this.permissionGranted = await Orientation.requestPermission() === 'granted'; if (!this.permissionGranted) this.onStatus('罗盘权限未开启，行走时可使用 GPS 航向'); }
      catch { this.onStatus('罗盘暂不可用，行走时可使用 GPS 航向'); }
    }
    this.active = true;
    this.startedAt = Date.now();
    this.filter.reset();
    window.addEventListener('deviceorientation', this.orientation);
    window.addEventListener('deviceorientationabsolute', this.orientation as EventListener);
    document.addEventListener('visibilitychange', this.visibility);
    this.subscribeLocation();
    this.timer = setInterval(() => { if (!document.hidden) this.onHeading(this.filter.value(Date.now())); }, 50);
  }

  private subscribeLocation() {
    this.clearWatch();
    if (!navigator.geolocation) { this.onStatus('此浏览器没有定位接口'); return; }
    this.watch = navigator.geolocation.watchPosition(position => {
      if (!finite(position.timestamp) || position.timestamp > Date.now() + 250 || position.timestamp < this.startedAt || Date.now() - position.timestamp > 10000 ||
          !finite(position.coords.latitude) || !finite(position.coords.longitude) || Math.abs(position.coords.latitude) > 90 || Math.abs(position.coords.longitude) > 180 ||
          !finite(position.coords.accuracy) || position.coords.accuracy < 0) return;
      this.onPosition(position);
      const sample = courseSample(position.coords, position.timestamp);
      if (sample) this.filter.push(sample, Date.now());
      else this.filter.course = null;
    }, error => this.onStatus(error.code === 1 ? '定位权限未开启，请在浏览器设置中允许' : '暂未获取可靠位置，请移至开阔区域'),
    { enableHighAccuracy: true, maximumAge: 0, timeout: 15000 });
  }

  private clearWatch() { if (this.watch !== null) navigator.geolocation.clearWatch(this.watch); this.watch = null; }
  calibrate(bearing: number, now = Date.now()) {
    if (!finite(bearing) || bearing < 0 || bearing >= 360) throw new Error('方位角须在 0–359° 之间');
    const sample = this.filter.compass;
    if (!sample || now - sample.timestamp > 1500 || sample.timestamp > now + 250) throw new Error('请先开启跟随，等待可靠的罗盘方向');
    this.offset += angleDelta(sample.degrees, bearing);
    this.calibrated = true; this.filter.reset();
  }
  clearCalibration() { this.offset = 0; this.calibrated = false; this.filter.reset(); }
  stop() {
    this.active = false; this.clearWatch(); this.filter.reset();
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    window.removeEventListener('deviceorientation', this.orientation);
    window.removeEventListener('deviceorientationabsolute', this.orientation as EventListener);
    document.removeEventListener('visibilitychange', this.visibility);
    this.onHeading(null);
  }
}
