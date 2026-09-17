"""Protocol fixture, not a language model or an accuracy benchmark."""
import http.server
import json
import math
import os
import pathlib
import subprocess
import tempfile
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
seen = []
mode = "ok"
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/chat/completions"
        assert body["max_tokens"] == 1 and body["logprobs"] is True
        payload = json.loads(body["messages"][1]["content"])
        seen.append(payload)
        if mode == "http":
            self.send_response(503); self.end_headers(); return
        tokens = [{"token": "0", "logprob": math.log(.2)}, {"token": "1", "logprob": math.log(.8)}]
        if mode == "missing": tokens.pop()
        if mode == "sentinel": tokens[0]["logprob"] = -9999
        data = json.dumps({"choices": [{"logprobs": {"content": [{"top_logprobs": tokens}]}}]}).encode()
        self.send_response(200); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)

server = http.server.ThreadingHTTPServer(("127.0.0.1",0),Handler)
thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
env = dict(os.environ, JAX_ENDPOINT=f"http://127.0.0.1:{server.server_port}/v1/chat/completions", JAX_TRACE="1", JAX_API_KEY="", JAX_WORKERS="4")
def run(*args):
    return subprocess.run([str(ROOT/"build/jax"),*map(str,args)],capture_output=True,text=True,env=env,timeout=20)
try:
    request = ROOT/"examples/ticket.json"
    result = run("infer",request)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert len(seen) == 7
    assert all("department" not in p and "urgency" not in p for p in seen)
    a = out["answers"]
    assert abs(a["refund_requested"]["noul"]-.8)<1e-10
    assert "confidence" not in a["refund_requested"]
    assert a["department"]["choice"] == "billing" # alphabetical tie
    assert abs(sum(a["department"]["probabilities"].values())-1)<1e-10
    assert abs(a["urgency"]["score"]-1)<1e-10
    assert a["urgency"]["legend"]["0"].startswith("Routine")
    assert out["raw_logits"]["refund_requested"]["labels"] == ["false","true"]
    for p in seen:
        if "urgent" in p["instructions"]:
            assert "alternatives" not in p
            assert p["criterion"] in json.loads(request.read_text())["questions"]["urgency"]["criteria"]
    for failure in ("missing","sentinel","http"):
        mode = failure
        result = run("infer",request)
        assert result.returncode == 1 and result.stdout == "", (failure,result)
    mode = "ok"
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp)/"bad.json"
        for contents in ('{} garbage', '{"state": 1,"questions":{}}', '{"state": NaN,"questions":{}}'):
            before = len(seen); p.write_text(contents)
            assert run("infer",p).returncode == 1
            assert len(seen) == before
        r = run("calibrate",ROOT/"examples/calibration.json")
        assert r.returncode == 0, r.stderr
        calibrated = json.loads(r.stdout)
        assert calibrated["fit_nll_after"] < calibrated["fit_nll_before"]
        p.write_text(r.stdout)
        assert run("infer",request,p).returncode == 0
        metrics = run("evaluate",ROOT/"examples/calibration.json",p)
        assert metrics.returncode == 0, metrics.stderr
        assert abs(json.loads(metrics.stdout)["nll"]-calibrated["fit_nll_after"]) < 1e-10
        p.write_text('[{"logits":[0,0],"label":1}]')
        metrics = json.loads(run("evaluate",p).stdout)
        assert abs(metrics["brier"]-.5) < 1e-10
        assert abs(metrics["nll"]-math.log(2)) < 1e-10
        assert abs(metrics["ece_10"]-.5) < 1e-10
    print("integration: HTTP, schema, raw logits, failures, calibration CLI OK (fixture only)")
finally:
    server.shutdown(); server.server_close(); thread.join()
