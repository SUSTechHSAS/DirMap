#!/usr/bin/env python3
"""Dependency-free static file server for verifying the DirMap static build.

It mimics the parts of GitHub Pages that matter for this project:

* serves files only below ``--dir`` (default ``web/dist``),
* strips a configurable URL base prefix, so ``--base /DIRMAP/`` reproduces
  ``https://<user>.github.io/<repo>/``,
* falls back to ``index.html`` for navigation requests whose path does not
  exist (single-page app), without any directory listing,
* sends sensible ``Content-Type`` headers and ``Cache-Control`` values
  (``no-cache`` for HTML/JSON/service worker, immutable for content-hashed
  assets under ``assets/``), and supports ``HEAD``.

There is no API and no directory listing; the server only exists to verify the
build locally and in Playwright tests. It never writes to disk.

Examples::

    python3 tools/static_server.py
    python3 tools/static_server.py --dir web/dist --base /DIRMAP/ --port 4173
    python3 tools/static_server.py --base /DIRMAP/ --port 0 --quiet
"""
from __future__ import annotations

import argparse
import http.server
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Explicit table (rather than mimetypes) so the verification server behaves the
# same on every platform, including minimal CI images.
CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.htm': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.webmanifest': 'application/manifest+json; charset=utf-8',
    '.map': 'application/json; charset=utf-8',
    '.wasm': 'application/wasm',
    '.gz': 'application/gzip',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.webp': 'image/webp',
    '.gif': 'image/gif',
    '.ico': 'image/x-icon',
    '.pdf': 'application/pdf',
    '.txt': 'text/plain; charset=utf-8',
    '.xml': 'application/xml; charset=utf-8',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.ttf': 'font/ttf',
}

# Vite content-hashed file names look like ``index-Cna85sIQ.js`` / ``x-D3f4Gh5j.css``.
HASHED_NAME = re.compile(r'-[\w-]{8,}\.[A-Za-z0-9]+$')


def normalize_base(value: str) -> str:
    """Return a URL base that starts and ends with ``/`` (root stays ``/``)."""
    value = (value or '/').strip() or '/'
    if not value.startswith('/'):
        value = '/' + value
    if not value.endswith('/'):
        value += '/'
    return value if value != '//' else '/'


def content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), 'application/octet-stream')


def cache_control(relative: str, filename: str) -> str:
    if filename == 'sw.js':
        return 'no-cache'
    if relative.endswith(('.html', '.htm', '.json', '.webmanifest')):
        return 'no-cache'
    if relative.startswith('assets/') and HASHED_NAME.search(filename):
        return 'public, max-age=31536000, immutable'
    return 'no-cache'


class StaticServer(http.server.ThreadingHTTPServer):
    """Threaded HTTP server bound to one directory and one deployment base."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type[http.server.BaseHTTPRequestHandler],
                 root: Path, base: str) -> None:
        self.root = root
        self.base = base
        super().__init__(address, handler)


class StaticRequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'DirMapStatic/1.0'
    sys_version = ''

    # -- request handling ---------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        self._serve(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(head=True)

    def do_POST(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    # -- helpers ------------------------------------------------------------
    def _relative_path(self) -> str | None:
        """Path relative to the served directory, or None if outside the base."""
        raw_path = unquote(urlsplit(self.path).path)
        base = self.server.base  # type: ignore[attr-defined]
        if base != '/':
            if raw_path.startswith(base):
                relative = raw_path[len(base):]
            elif raw_path.rstrip('/') == base.rstrip('/'):
                relative = ''
            else:
                return None
        else:
            relative = raw_path.lstrip('/')
        return relative.lstrip('/')

    def _resolve(self, relative: str) -> Path | None:
        root: Path = self.server.root  # type: ignore[attr-defined]
        try:
            candidate = (root / relative).resolve()
        except OSError:
            return None
        if candidate != root and root not in candidate.parents:
            return None
        return candidate

    def _serve(self, *, head: bool) -> None:
        relative = self._relative_path()
        if relative is None:
            self._send_error(404, 'Not Found', head=head)
            return
        resolved = self._resolve(relative)
        if resolved is None:
            self._send_error(404, 'Not Found', head=head)
            return

        target: Path | None = resolved
        if target.is_dir():
            index = target / 'index.html'
            target = index if index.is_file() else None
        if target is None or not target.is_file():
            # Single-page application: only navigation requests fall back to the
            # shell. Missing scripts/styles/data still return a real 404.
            if 'text/html' not in self.headers.get('Accept', ''):
                self._send_error(404, 'Not Found', head=head)
                return
            root: Path = self.server.root  # type: ignore[attr-defined]
            index = root / 'index.html'
            if not index.is_file():
                self._send_error(404, 'Not Found', head=head)
                return
            relative, target = 'index.html', index

        self._send_file(target, relative, head=head)

    def _send_file(self, path: Path, relative: str, *, head: bool) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            self._send_error(404, 'Not Found', head=head)
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type(path))
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', cache_control(relative, path.name))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if head:
            return
        try:
            with path.open('rb') as handle:
                while chunk := handle.read(64 * 1024):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):  # client went away (curl --head, reload)
            pass

    def _send_error(self, code: int, message: str, *, head: bool = False) -> None:
        body = f'{code} {message}\n'.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if not head:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _method_not_allowed(self) -> None:
        self.send_response(405)
        self.send_header('Allow', 'GET, HEAD')
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', '0')
        self.end_headers()


class QuietRequestHandler(StaticRequestHandler):
    """Handler that does not log every request (useful for automated tests)."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Serve the DirMap static build for verification.')
    parser.add_argument('--dir', default='web/dist', help='directory to serve (default: web/dist)')
    parser.add_argument('--base', default='/', help='URL base prefix, e.g. /DIRMAP/ (default: /)')
    parser.add_argument('--port', type=int, default=4173, help='TCP port (default: 4173)')
    parser.add_argument('--host', default='127.0.0.1', help='bind address (default: 127.0.0.1)')
    parser.add_argument('--quiet', action='store_true', help='suppress per-request logging')
    args = parser.parse_args(argv)

    root = Path(args.dir).expanduser().resolve()
    if not root.is_dir():
        print(f'error: directory not found: {root} (build first: npm run build)', file=sys.stderr)
        return 2
    base = normalize_base(args.base)

    handler = QuietRequestHandler if args.quiet else StaticRequestHandler
    server = StaticServer((args.host, args.port), handler, root=root, base=base)
    host, port = server.server_address[:2]
    print(f'Serving {root} at http://{host}:{port}{base}', flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
