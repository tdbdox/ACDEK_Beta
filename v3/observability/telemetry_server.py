"""
Telemetry server stub for ACDEK v3: minimal in-process HTTP server.
"""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

class TelemetryRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for telemetry endpoints."""
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format, *args):
        # Suppress default logging
        return

class TelemetryServer:
    """Scaffold for an in-process telemetry HTTP server."""
    def __init__(self, host='127.0.0.1', port=8000):
        self._server = HTTPServer((host, port), TelemetryRequestHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        """Start the telemetry server in a background thread."""
        self._thread.start()

    def stop(self):
        """Stop the telemetry server."""
        self._server.shutdown()
        self._thread.join()