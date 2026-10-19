let registration: Promise<ServiceWorkerRegistration | null> | null = null;

/** Base URL of the deployment, always ending in `/` (`/` or `/<repo>/`). */
const BASE = import.meta.env.BASE_URL;

export function installOffline(): Promise<ServiceWorkerRegistration | null> {
  if (!import.meta.env.PROD || !('serviceWorker' in navigator) || !window.isSecureContext) return Promise.resolve(null);
  registration ??= navigator.serviceWorker.register(`${BASE}sw.js`, { scope: BASE, updateViaCache: 'none' })
    .then(() => navigator.serviceWorker.ready).catch(() => null);
  return registration;
}

export async function saveOfflineCourse(id: string): Promise<void> {
  const worker = (await installOffline())?.active;
  if (!worker) throw new Error('此浏览器暂不能保存离线课程');
  await new Promise<void>((resolve, reject) => {
    const channel = new MessageChannel();
    const timer = setTimeout(() => { channel.port1.close(); reject(new Error('课程尚未完整保存，联网后可重试')); }, 20000);
    channel.port1.onmessage = event => {
      clearTimeout(timer); channel.port1.close();
      if (event.data.ok) resolve(); else reject(new Error(event.data.message || '课程保存失败'));
    };
    worker.postMessage({ type: 'CACHE_COURSE', id }, [channel.port2]);
  });
}
