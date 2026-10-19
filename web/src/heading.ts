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
  private locationTimer: ReturnType<typeof setTimeout> | null = null;
  private headingTimer: ReturnType<typeof setTimeout> | null = null;
  private headingSeen = false;
  private startedAt = 0;
  /** Diagnostic counters (also surfaced by the in-app 定位诊断 dialog). */
  rawPositionCount = 0;
  acceptedPositionCount = 0;
  lastGeoError: string | null = null;
  lastRawTimestamp = 0;
  lastRawReceivedAt = 0;
  lastAcceptedReceivedAt = 0;
  orientationEventCount = 0;
  absoluteEventCount = 0;
  headingSampleCount = 0;
  locationTimeoutMs = 12000;
  active = false;
  declination: number | null = null;
  offset = 0;
  calibrated = false;
  private permissionGranted = false;
  onHeading: (sample: HeadingSample | null) => void = () => {};
  onPosition: (position: GeolocationPosition) => void = () => {};
  onStatus: (message: string) => void = () => {};
  /** Location-specific status shown in the map's location line. */
  onLocationStatus: (message: string) => void = () => {};

  private orientation = (event: DeviceOrientationEvent) => {
    if (document.hidden) return;
    const timestamp = event.timeStamp > 1e12 ? event.timeStamp : performance.timeOrigin + event.timeStamp;
    if (timestamp < this.startedAt) return;
    this.orientationEventCount++;
    if (event.absolute === true || event.type === 'deviceorientationabsolute') this.absoluteEventCount++;
    const angle = screen.orientation?.angle ?? Number((window as Window & { orientation?: number }).orientation ?? 0);
    const sample = compassSample(event, angle, timestamp, this.declination);
    if (sample) {
      sample.degrees = normalize(sample.degrees + this.offset);
      if (this.calibrated) sample.reference = 'true';
      this.filter.push(sample, Date.now());
      this.headingSampleCount++;
      if (!this.headingSeen) {
        this.headingSeen = true;
        if (this.headingTimer !== null) clearTimeout(this.headingTimer);
        this.headingTimer = null;
      }
    }
  };
  private visibility = () => {
    this.filter.reset();
    this.startedAt = Date.now();
    if (document.hidden) {
      this.clearWatch();
      this.clearLocationTimer();
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
    this.rawPositionCount = 0; this.acceptedPositionCount = 0; this.lastGeoError = null;
    this.lastRawTimestamp = 0; this.lastRawReceivedAt = 0; this.lastAcceptedReceivedAt = 0;
    this.orientationEventCount = 0; this.absoluteEventCount = 0; this.headingSampleCount = 0;
    if (typeof window.DeviceOrientationEvent === 'undefined') {
      this.onStatus('此浏览器不支持手机朝向，将仅使用 GPS 行进方向');
    } else {
      window.addEventListener('deviceorientation', this.orientation);
      window.addEventListener('deviceorientationabsolute', this.orientation as EventListener);
    }
    document.addEventListener('visibilitychange', this.visibility);
    this.subscribeLocation();
    if (!this.headingTimer) {
      this.headingTimer = setTimeout(() => {
        if (!this.active || this.headingSeen) return;
        this.onStatus('尚未收到罗盘数据：请允许“动作与方向”权限（iOS：设置 › Safari › 动作与方向访问），或在校准后使用行进方向');
      }, 6000);
    }
    this.timer = setInterval(() => { if (!document.hidden) this.onHeading(this.filter.value(Date.now())); }, 50);
  }

  private subscribeLocation() {
    this.clearWatch();
    this.clearLocationTimer();
    if (!navigator.geolocation) {
      const message = '此浏览器没有定位接口';
      this.onStatus(message); this.onLocationStatus(message);
      return;
    }
    this.onLocationStatus('正在获取位置…');
    const options: PositionOptions = { enableHighAccuracy: true, maximumAge: 5000, timeout: 20000 };
    try {
      this.watch = navigator.geolocation.watchPosition(position => this.acceptPosition(position), error => this.handleLocationError(error, false), options);
    } catch (error) {
      const message = `浏览器拒绝启动定位：${error instanceof Error ? error.message : String(error)}`;
      this.onStatus(message); this.onLocationStatus(message);
    }
    // Give the user an actionable message instead of an eternal
    // “位置尚未开启” when the first fix takes long or the browser blocks it.
    this.locationTimer = setTimeout(() => {
      if (!this.active || this.watch === null) return;
      if (this.rawPositionCount === 0) {
        // Some embedded WebViews deliver only getCurrentPosition, not a watch
        // stream. Probe exactly once before giving up.
        navigator.geolocation.getCurrentPosition(position => this.acceptPosition(position),
          error => this.handleLocationError(error, true),
          { enableHighAccuracy: false, maximumAge: 30000, timeout: 20000 });
      }
      const message = '仍未收到位置：请允许定位、到开阔处重试，并使用系统浏览器打开（微信等内嵌浏览器可能禁用定位）';
      this.onStatus(message); this.onLocationStatus(message);
    }, this.locationTimeoutMs);
  }

  /** Validate one geolocation fix and forward it; shared by watch/getCurrentPosition. */
  private acceptPosition(position: GeolocationPosition) {
    const receivedAt = Date.now();
    this.rawPositionCount++;
    const rawTimestamp = (position as GeolocationPosition).timestamp;
    // Providers may timestamp the first fix slightly BEFORE the watch started
    // (cached/emulated fixes, clock skew on iOS and embedded WebViews). Accept
    // it instead of silently rejecting every fix and leaving the app at
    // “位置尚未开启”. Only clearly stale or future samples are discarded.
    const timestamp = finite(rawTimestamp) && rawTimestamp > 0 ? rawTimestamp : receivedAt;
    this.lastRawTimestamp = timestamp;
    this.lastRawReceivedAt = receivedAt;
    if (timestamp > receivedAt + 5000 || receivedAt - timestamp > 60000) return;
    if (!finite(position.coords.latitude) || !finite(position.coords.longitude) ||
        Math.abs(position.coords.latitude) > 90 || Math.abs(position.coords.longitude) > 180) return;
    const accuracy = position.coords.accuracy;
    if (finite(accuracy) && (accuracy < 0 || accuracy > 10000)) return;
    this.acceptedPositionCount++;
    this.lastAcceptedReceivedAt = receivedAt;
    this.clearLocationTimer();
    try { this.onPosition(position); } catch { /* never break the geolocation stream for UI errors */ }
    const sample = courseSample(position.coords, timestamp);
    if (sample) this.filter.push(sample, receivedAt);
    else this.filter.course = null;
  }

  private handleLocationError(error: GeolocationPositionError, probe: boolean) {
    this.lastGeoError = `${error.code}: ${error.message || ''}`.trim();
    const message = error.code === 1 ? '定位权限被拒绝：请在浏览器设置中允许定位后重试'
      : error.code === 3 ? '定位超时：请到开阔区域后重试'
      : '暂未获取可靠位置：请检查 GPS/网络后移至开阔区域';
    this.onStatus(message); this.onLocationStatus(message);
    // Some devices fail high-accuracy mode indoors; retry once with the
    // network/wifi provider instead of leaving the UI without any position.
    if (!probe && (error.code === 2 || error.code === 3) && this.active) {
      this.clearWatch();
      try {
        this.watch = navigator.geolocation.watchPosition(position => this.acceptPosition(position),
          retryError => this.handleLocationError(retryError, true),
          { enableHighAccuracy: false, maximumAge: 10000, timeout: 15000 });
      } catch { /* reported by the next error/timeout message */ }
    }
  }

  /** Snapshot for the in-app 定位诊断 panel. */
  diagnostics() {
    return {
      active: this.active,
      secureContext: typeof window !== 'undefined' && window.isSecureContext,
      geolocationAvailable: typeof navigator !== 'undefined' && !!navigator.geolocation,
      startedAt: this.startedAt,
      watchId: this.watch,
      rawPositionCount: this.rawPositionCount,
      acceptedPositionCount: this.acceptedPositionCount,
      lastGeoError: this.lastGeoError,
      lastRawTimestamp: this.lastRawTimestamp,
      lastRawReceivedAt: this.lastRawReceivedAt,
      lastAcceptedReceivedAt: this.lastAcceptedReceivedAt,
      locationTimeoutMs: this.locationTimeoutMs,
      orientationSupported: typeof window !== 'undefined' && typeof window.DeviceOrientationEvent !== 'undefined',
      orientationPermissionGranted: this.permissionGranted,
      orientationEventCount: this.orientationEventCount,
      absoluteEventCount: this.absoluteEventCount,
      headingSampleCount: this.headingSampleCount,
      heading: this.filter.value(Date.now())?.degrees ?? null,
    };
  }

  private clearLocationTimer() {
    if (this.locationTimer !== null) clearTimeout(this.locationTimer);
    this.locationTimer = null;
  }

  private clearWatch() {
    if (this.watch !== null) navigator.geolocation.clearWatch(this.watch);
    this.watch = null;
    this.clearLocationTimer();
  }
  calibrate(bearing: number, now = Date.now()) {
    if (!finite(bearing) || bearing < 0 || bearing >= 360) throw new Error('方位角须在 0–359° 之间');
    const sample = this.filter.compass;
    if (!sample || now - sample.timestamp > 1500 || sample.timestamp > now + 250) throw new Error('请先开启跟随，等待可靠的罗盘方向');
    this.offset += angleDelta(sample.degrees, bearing);
    this.calibrated = true; this.filter.reset();
  }
  clearCalibration() { this.offset = 0; this.calibrated = false; this.filter.reset(); }
  stop() {
    this.active = false; this.clearWatch(); this.clearLocationTimer(); this.filter.reset();
    if (this.headingTimer !== null) clearTimeout(this.headingTimer);
    this.headingTimer = null; this.headingSeen = false;
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    window.removeEventListener('deviceorientation', this.orientation);
    window.removeEventListener('deviceorientationabsolute', this.orientation as EventListener);
    document.removeEventListener('visibilitychange', this.visibility);
    this.onHeading(null);
  }
}
