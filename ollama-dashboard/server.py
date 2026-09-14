#!/usr/bin/env python3
"""Ollama Dashboard — local web UI with API proxy.

Serves the static dashboard and proxies /api/* and /v1/* to a local Ollama
daemon so the browser is not blocked by CORS.

Usage:
  python3 server.py [--host 127.0.0.1] [--port 8080] [--ollama http://127.0.0.1:11434]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
PROXY_PREFIXES = ("/api/", "/v1/")


class DashboardHandler(BaseHTTPRequestHandler):
    ollama_base: str = DEFAULT_OLLAMA
    server_version = "OllamaDashboard/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str, extra_headers: Optional[dict] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _json_error(self, code: int, message: str) -> None:
        payload = json.dumps({"error": message}).encode("utf-8")
        self._send(code, payload, "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:
        self._handle_request(include_body=False)

    def do_GET(self) -> None:
        self._handle_request(include_body=True)

    def do_POST(self) -> None:
        self._handle_request(include_body=True)

    def do_DELETE(self) -> None:
        self._handle_request(include_body=True)

    def do_PUT(self) -> None:
        self._handle_request(include_body=True)

    def _handle_request(self, include_body: bool) -> None:
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        if path == "/healthz":
            body = json.dumps(
                {
                    "status": "ok",
                    "ollama": self.ollama_base,
                }
            ).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return

        if any(path.startswith(prefix) for prefix in PROXY_PREFIXES):
            self._proxy(path, parsed.query, include_body=include_body)
            return

        self._serve_static(path)

    def _serve_static(self, path: str) -> None:
        if path in ("/", ""):
            rel = "index.html"
        else:
            rel = path.lstrip("/")

        # Prevent path traversal
        target = (STATIC_DIR / rel).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._json_error(403, "Forbidden")
            return

        if not target.is_file():
            # SPA-style fallback to index for unknown routes
            target = STATIC_DIR / "index.html"
            if not target.is_file():
                self._json_error(404, "Not found")
                return

        data = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in (
            "application/javascript",
            "application/json",
        ):
            content_type = f"{content_type}; charset=utf-8"
        self._send(200, data, content_type)

    def _proxy(self, path: str, query: str, include_body: bool) -> None:
        target = self.ollama_base.rstrip("/") + path
        if query:
            target = f"{target}?{query}"

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else None

        headers = {}
        content_type = self.headers.get("Content-Type")
        if content_type:
            headers["Content-Type"] = content_type
        # Prefer identity so we can stream cleanly
        headers["Accept"] = self.headers.get("Accept", "*/*")
        headers["Accept-Encoding"] = "identity"

        req = urllib.request.Request(
            target,
            data=body if self.command in ("POST", "PUT", "DELETE", "PATCH") else None,
            headers=headers,
            method=self.command,
        )

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                upstream_ct = resp.headers.get("Content-Type", "application/octet-stream")
                # Stream response to support Ollama NDJSON streams
                self.send_response(resp.status)
                self.send_header("Content-Type", upstream_ct)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                # Do not set Content-Length so chunked/streamed body is fine
                self.end_headers()
                if not include_body:
                    return
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as exc:
            err_body = exc.read() if exc.fp is not None else b""
            if not err_body:
                err_body = json.dumps({"error": exc.reason}).encode("utf-8")
            self.send_response(exc.code)
            self.send_header(
                "Content-Type",
                exc.headers.get("Content-Type", "application/json; charset=utf-8"),
            )
            self.send_header("Content-Length", str(len(err_body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if include_body:
                self.wfile.write(err_body)
        except urllib.error.URLError as exc:
            self._json_error(
                502,
                f"Cannot reach Ollama at {self.ollama_base}: {exc.reason}",
            )
        except (TimeoutError, socket.timeout):
            self._json_error(504, "Ollama request timed out")
        except BrokenPipeError:
            # Client disconnected mid-stream
            return


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama Dashboard local server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument(
        "--ollama",
        default=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA),
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA} or $OLLAMA_HOST)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not STATIC_DIR.is_dir():
        print(f"error: static directory missing: {STATIC_DIR}", file=sys.stderr)
        return 1

    ollama = args.ollama
    if not ollama.startswith("http://") and not ollama.startswith("https://"):
        ollama = "http://" + ollama

    DashboardHandler.ollama_base = ollama.rstrip("/")
    httpd = ThreadingHTTPServer((args.host, args.port), DashboardHandler)

    def _shutdown_on_signal() -> None:
        httpd.shutdown()

    print(f"Ollama Dashboard listening on http://{args.host}:{args.port}")
    print(f"Proxying API requests to {DashboardHandler.ollama_base}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")
        threading.Thread(target=_shutdown_on_signal, daemon=True).start()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
