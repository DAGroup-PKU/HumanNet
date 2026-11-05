"""Small HTTP adapter with explicit image and action payloads."""

import base64
import binascii
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

MAX_BODY_BYTES = 32 * 1024 * 1024


def decode_observation(payload):
    def images(views):
        result = {}
        for camera, encoded in views.items():
            content = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(content)) as source:
                result[camera] = np.array(source.convert("RGB"), copy=True)
        return result

    result = dict(payload)
    result["images"] = images(payload["images"])
    result["goal_images"] = images(payload["goal_images"])
    result["history"] = [
        dict(capture, images=images(capture["images"])) for capture in payload.get("history", [])
    ]
    return result


def serve(model, host="127.0.0.1", port=8000):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/healthz":
                self.reply(200, {"status": "ok"})
            elif self.path == "/metadata":
                self.reply(
                    200,
                    {
                        "model": "GoalWAM",
                        "horizon": model.config.horizon,
                        "output_action_dim": model.config.output_action_dim,
                        "rtc_prefix_length": model.sampling.prefix_length,
                        "physical_inputs": model.robot is not None,
                    },
                )
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/infer":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= MAX_BODY_BYTES:
                    self.reply(413, {"error": "Request body must be between 1 byte and 32 MiB"})
                    return
                payload = json.loads(self.rfile.read(length))
                result = model.predict(decode_observation(payload))
                self.reply(
                    200,
                    {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in result.items()
                    },
                )
            except (ValueError, KeyError, TypeError, binascii.Error, OSError) as error:
                self.reply(400, {"error": str(error)})
            except Exception:
                self.reply(500, {"error": "Inference failed; inspect the server log"})
                import logging

                logging.exception("GoalWAM inference failed")

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"GoalWAM listening on http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
