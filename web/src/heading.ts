/** Device angles and motion course are distinct sources, never interchanged. */
export type HeadingSource = 'compass' | 'course';
export type HeadingSample = { degrees: number; timestamp: number; source: HeadingSource; reference: 'true' | 'magnetic'; relative?: boolean };
export const normalize = (degrees: number) => (degrees % 360 + 360) % 360;
export const angleDelta = (from: number, to: number) => (to - from + 540) % 360 - 180;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
/** Minimal Generic Sensor handle used by the sensor-API fallbacks. */
type SensorHandle = { start(): void; stop(): void };

export function smoothHeading(previous: number, next: number, elapsedMs: number, tauMs = 250): number {
  if (![previous, next, elapsedMs, tauMs].every(finite) || elapsedMs < 0 || tauMs <= 0) throw new Error('Invalid heading');
  return normalize(previous + (1 - Math.exp(-elapsedMs / tauMs)) * angleDelta(previous, next));
}

export function orientationHeading(alpha: unknown, beta: unknown, gamma: unknown, screenAngle = 0, absolute = false, allowRelative = false): number | null {
  if ((!absolute && !allowRelative) || !finite(alpha) || !finite(beta) || !finite(gamma) || !finite(screenAngle)) return null;
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
  lastRawTimestampValue: number | null = null;
  lastAccuracy: number | null = null;
  lastRawReceivedAt = 0;
  lastAcceptedReceivedAt = 0;
  orientationEventCount = 0;
  absoluteEventCount = 0;
  headingSampleCount = 0;
  relativeSeen = false;
  absoluteSampleCount = 0;
  lastAbsoluteSampleAt = 0;
  sensorSampleCount = 0;
  locationTimeoutMs = 12000;
  private orientationSensor: SensorHandle | null = null;
  private magnetometer: SensorHandle | null = null;
  private accelerometer: SensorHandle | null = null;
  private magReading: { x: number; y: number; z: number; at: number } | null = null;
  private accReading: { x: number; y: number; z: number; at: number } | null = null;
  private sensorState = '未启动';
  private sensorError: string | null = null;
  private sensorTimer: ReturnType<typeof setTimeout> | null = null;
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
    let sample = compassSample(event, angle, timestamp, this.declination);
    let relative = false;
    if (sample) {
      this.absoluteSampleCount++;
      this.lastAbsoluteSampleAt = Date.now();
      if (this.orientationSensor || this.magnetometer) this.stopSensorFallbacks();
    } else {
      // Both `deviceorientation` (relative) and `deviceorientationabsolute`
      // events fire on most phones. Only fall back to the relative device
      // direction when the absolute north has produced no usable heading for a
      // while; mixing the two streams made the heading (and status text) jump
      // between two unrelated directions.
      const now = Date.now();
      const gracePassed = now - this.startedAt > 1200;
      const absoluteRecently = now - this.lastAbsoluteSampleAt < 3000;
      if (gracePassed && !absoluteRecently) {
        const degrees = orientationHeading(event.alpha, event.beta, event.gamma, angle, false, true);
        if (degrees !== null) {
          sample = { degrees, timestamp, source: 'compass', reference: 'magnetic', relative: true };
          relative = true;
          if (!this.relativeSeen) {
            this.relativeSeen = true;
            this.onStatus('此设备/浏览器未提供绝对北向：已启用相对方向，请对准已知方位后点“校准罗盘”');
          }
        }
      }
    }
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
    this.absoluteSampleCount = 0; this.lastAbsoluteSampleAt = 0; this.relativeSeen = false;
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
    // Chromium-based browsers (including HarmonyOS ArkWeb) sometimes emit
    // deviceorientation events without a usable absolute heading but do expose
    // the Generic Sensor API. Try it when no absolute sample arrived.
    this.sensorTimer = setTimeout(() => {
      if (this.active && this.absoluteSampleCount === 0) this.startSensorFallbacks();
    }, 3000);
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
    // Some OpenHarmony/ArkWeb and embedded WebViews return a monotonic
    // DOMHighResTimeStamp (small value) or 0 instead of Unix epoch. Never use the
    // provider timestamp to reject a fix: validate coordinates only and use the
    // receive time for freshness/navigation. (The previous epoch comparison
    // silently rejected every fix on those devices.)
    const timestamp = finite(rawTimestamp) && rawTimestamp > 1e12 ? rawTimestamp : receivedAt;
    this.lastRawTimestamp = timestamp;
    this.lastRawTimestampValue = finite(rawTimestamp) ? rawTimestamp : null;
    this.lastRawReceivedAt = receivedAt;
    const accuracy = position.coords.accuracy;
    this.lastAccuracy = finite(accuracy) ? accuracy : null;
    if (!finite(position.coords.latitude) || !finite(position.coords.longitude) ||
        Math.abs(position.coords.latitude) > 90 || Math.abs(position.coords.longitude) > 180) return;
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

  /** Sensor-API fallbacks when DOM orientation events lack a usable north. */
  private startSensorFallbacks() {
    if (this.orientationSensor || this.magnetometer) return;
    if (this.startAbsoluteOrientationSensor()) return;
    this.startMagnetometer();
  }

  /** AbsoluteOrientationSensor gives a quaternion in the earth frame. */
  private startAbsoluteOrientationSensor(): boolean {
    const Ctor = (window as unknown as { AbsoluteOrientationSensor?: new (options?: unknown) => SensorHandle & {
      quaternion?: number[] | null; onreading: (() => void) | null; onerror: (() => void) | null;
    } }).AbsoluteOrientationSensor;
    if (typeof Ctor !== 'function') { this.sensorState = 'AbsoluteOrientationSensor 不可用'; return false; }
    try {
      const sensor = new Ctor({ frequency: 30, referenceFrame: 'device' });
      sensor.onreading = () => {
        if (!this.active) return;
        const q = sensor.quaternion;
        if (!q || q.length < 4) return;
        const angle = (screen.orientation?.angle ?? Number((window as Window & { orientation?: number }).orientation ?? 0)) * Math.PI / 180;
        const [x, y, z, w] = q;
        // Rotate the screen-top vector into the earth frame (x east, y north).
        const vx = Math.sin(angle), vy = Math.cos(angle);
        const east = (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy;
        const north = 2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy;
        if (Math.hypot(east, north) < .25) return;
        this.pushSensorHeading(Math.atan2(east, north) * 180 / Math.PI, Date.now());
      };
      sensor.onerror = () => {
        this.sensorError = 'AbsoluteOrientationSensor 启动失败'; this.sensorState = 'AbsoluteOrientationSensor 错误';
        try { sensor.stop(); } catch { /* ignore */ }
        this.orientationSensor = null;
        this.startMagnetometer();
      };
      sensor.start();
      this.orientationSensor = sensor;
      this.sensorState = 'AbsoluteOrientationSensor 已启用';
      return true;
    } catch (error) {
      this.sensorError = `AbsoluteOrientationSensor: ${error instanceof Error ? error.message : String(error)}`;
      this.sensorState = 'AbsoluteOrientationSensor 错误';
      return false;
    }
  }

  /** Magnetometer + accelerometer give a tilt-compensated compass heading. */
  private startMagnetometer() {
    if (this.magnetometer) return;
    type VectorSensor = new (options?: unknown) => SensorHandle & {
      x?: number; y?: number; z?: number; onreading: (() => void) | null; onerror: (() => void) | null;
    };
    const api = window as unknown as { Magnetometer?: VectorSensor; Accelerometer?: VectorSensor };
    if (typeof api.Magnetometer !== 'function' || typeof api.Accelerometer !== 'function') {
      this.sensorState = '浏览器不提供 Sensor API';
      this.sensorError = 'AbsoluteOrientationSensor / Magnetometer / Accelerometer 均不可用';
      this.onStatus('此浏览器未向网页提供设备朝向数据：可在浏览器设置中允许“运动与方向/传感器”，或更换浏览器');
      return;
    }
    try {
      const magnetometer = new api.Magnetometer({ frequency: 10, referenceFrame: 'device' });
      const accelerometer = new api.Accelerometer({ frequency: 10, referenceFrame: 'device' });
      magnetometer.onreading = () => {
        if (typeof magnetometer.x !== 'number') return;
        this.magReading = { x: magnetometer.x, y: magnetometer.y ?? 0, z: magnetometer.z ?? 0, at: Date.now() };
        this.pushMagnetometerHeading();
      };
      accelerometer.onreading = () => {
        if (typeof accelerometer.x !== 'number') return;
        this.accReading = { x: accelerometer.x, y: accelerometer.y ?? 0, z: accelerometer.z ?? 0, at: Date.now() };
        this.pushMagnetometerHeading();
      };
      const fail = () => {
        this.sensorState = 'Magnetometer 错误';
        this.sensorError = this.sensorError ?? '浏览器拒绝提供磁力计/加速度计数据';
        this.stopSensorFallbacks();
      };
      magnetometer.onerror = fail; accelerometer.onerror = fail;
      magnetometer.start(); accelerometer.start();
      this.magnetometer = magnetometer; this.accelerometer = accelerometer;
      this.sensorState = 'Magnetometer + Accelerometer 已启用';
    } catch (error) {
      this.sensorState = 'Magnetometer 错误';
      this.sensorError = `Magnetometer: ${error instanceof Error ? error.message : String(error)}`;
      this.onStatus('此浏览器未向网页提供设备朝向数据：可在浏览器设置中允许“运动与方向/传感器”，或更换浏览器');
    }
  }

  private pushMagnetometerHeading() {
    const mag = this.magReading, acc = this.accReading;
    if (!mag || !acc || Math.abs(mag.at - acc.at) > 500) return;
    const acceleration = Math.hypot(acc.x, acc.y, acc.z);
    if (!Number.isFinite(acceleration) || acceleration < 1e-3) return;
    const up = [acc.x / acceleration, acc.y / acceleration, acc.z / acceleration];
    const vertical = mag.x * up[0] + mag.y * up[1] + mag.z * up[2];
    const horizontal = [mag.x - vertical * up[0], mag.y - vertical * up[1], mag.z - vertical * up[2]];
    if (Math.hypot(horizontal[0], horizontal[1], horizontal[2]) < 1e-3) return;
    const project = (v: number[]) => {
      const dot = v[0] * up[0] + v[1] * up[1] + v[2] * up[2];
      return [v[0] - dot * up[0], v[1] - dot * up[1], v[2] - dot * up[2]];
    };
    const angle = (screen.orientation?.angle ?? Number((window as Window & { orientation?: number }).orientation ?? 0)) * Math.PI / 180;
    const top = project([Math.sin(angle), Math.cos(angle), 0]);
    const right = project([Math.cos(angle), -Math.sin(angle), 0]);
    if (Math.hypot(top[0], top[1], top[2]) < 1e-3) return;
    const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const heading = Math.atan2(-dot(horizontal, right), dot(horizontal, top)) * 180 / Math.PI;
    this.pushSensorHeading(heading, Date.now());
  }

  /** Feed a sensor-derived absolute heading through the same smoothing path. */
  private pushSensorHeading(rawDegrees: number, now: number) {
    if (!Number.isFinite(rawDegrees)) return;
    const degrees = normalize(rawDegrees + (this.declination ?? 0));
    const sample: HeadingSample = { degrees, timestamp: now, source: 'compass', reference: this.declination === null ? 'magnetic' : 'true' };
    this.headingSampleCount++;
    this.absoluteSampleCount++;
    this.sensorSampleCount++;
    this.lastAbsoluteSampleAt = now;
    this.filter.push(sample, now);
    if (!this.headingSeen) {
      this.headingSeen = true;
      if (this.headingTimer !== null) clearTimeout(this.headingTimer);
      this.headingTimer = null;
    }
  }

  private stopSensorFallbacks() {
    try { this.orientationSensor?.stop(); } catch { /* ignore */ }
    try { this.magnetometer?.stop(); } catch { /* ignore */ }
    try { this.accelerometer?.stop(); } catch { /* ignore */ }
    this.orientationSensor = null; this.magnetometer = null; this.accelerometer = null;
    this.magReading = null; this.accReading = null;
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
      lastRawTimestampValue: this.lastRawTimestampValue,
      lastAccuracy: this.lastAccuracy,
      lastRawReceivedAt: this.lastRawReceivedAt,
      lastAcceptedReceivedAt: this.lastAcceptedReceivedAt,
      locationTimeoutMs: this.locationTimeoutMs,
      orientationSupported: typeof window !== 'undefined' && typeof window.DeviceOrientationEvent !== 'undefined',
      orientationPermissionGranted: this.permissionGranted,
      orientationPermissionRequired: typeof window !== 'undefined' && typeof (window as unknown as { DeviceOrientationEvent?: { requestPermission?: unknown } }).DeviceOrientationEvent?.requestPermission === 'function',
      orientationEventCount: this.orientationEventCount,
      absoluteEventCount: this.absoluteEventCount,
      headingSampleCount: this.headingSampleCount,
      absoluteSampleCount: this.absoluteSampleCount,
      sensorSampleCount: this.sensorSampleCount,
      sensorState: this.sensorState,
      sensorError: this.sensorError,
      orientationSensorActive: this.orientationSensor !== null,
      magnetometerActive: this.magnetometer !== null,
      absoluteOrientationSensorAvailable: typeof (window as unknown as { AbsoluteOrientationSensor?: unknown }).AbsoluteOrientationSensor === 'function',
      magnetometerAvailable: typeof (window as unknown as { Magnetometer?: unknown }).Magnetometer === 'function' && typeof (window as unknown as { Accelerometer?: unknown }).Accelerometer === 'function',
      relativeSeen: this.relativeSeen,
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
    if (this.sensorTimer !== null) clearTimeout(this.sensorTimer);
    this.sensorTimer = null;
    this.stopSensorFallbacks();
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    window.removeEventListener('deviceorientation', this.orientation);
    window.removeEventListener('deviceorientationabsolute', this.orientation as EventListener);
    document.removeEventListener('visibilitychange', this.visibility);
    this.onHeading(null);
  }
}
