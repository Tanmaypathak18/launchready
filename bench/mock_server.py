"""Tiny OpenAI-compatible mock server for testing the bench scripts without a GPU or weights.

Serves /health, /version, /v1/models and /v1/chat/completions (streaming and not). Each
completion returns `max_completion_tokens` (or `max_tokens`) words, one per SSE chunk, with
a fixed delay between them. Never used for published numbers.

    python bench/mock_server.py --port 8000 --token-delay-ms 2
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock-model"


def make_handler(token_delay: float, model: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep stdout quiet
            pass

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/version":
                self._json(200, {"version": "mock-0"})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            n = int(req.get("max_completion_tokens") or req.get("max_tokens") or 16)
            prompt = " ".join(str(m.get("content", "")) for m in req.get("messages", []))
            usage = {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": n,
                "total_tokens": len(prompt.split()) + n,
            }
            base = {"id": "cmpl-mock", "created": int(time.time()), "model": model}
            if not req.get("stream"):
                time.sleep(token_delay * n)
                text = " ".join(["tok"] * n)
                choice = {"index": 0, "message": {"role": "assistant", "content": text}}
                choice["finish_reason"] = "length"
                self._json(
                    200, {**base, "object": "chat.completion", "choices": [choice], "usage": usage}
                )
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for i in range(n):
                time.sleep(token_delay)
                delta = {"content": "tok "} if i else {"role": "assistant", "content": "tok "}
                chunk = {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            final = {
                **base,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                "usage": usage,
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
            self.close_connection = True

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token-delay-ms", type=float, default=2.0)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    handler = make_handler(args.token_delay_ms / 1000, args.model)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"mock server on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
