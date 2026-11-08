#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise every model of a LiteLLM gateway, per USAGE, fallbacks off -- and know who depends on it.

Why: a vision model had been returning 404 for an unknown time. It was found by
chance, while building something else. Nothing had seen it because:
  - the gateway was "active": a live service is not a live model;
  - the agents "answered": they answered THROUGH THE FALLBACK. A working fallback
    HIDES a dead model. This probe therefore calls each model with fallbacks
    DISABLED (`disable_fallbacks: true`): it judges the model, not the chain.

The test depends on the usage
  text        "answer with the single word READY"          -> the word must come back
  vision      an image showing a random 4-digit number     -> the number must be read
  embeddings  a non-empty vector
Also measures latency. DEAD = error or empty; WRONG = answers but fails the test;
SLOW = above PROBE_SLOW_S seconds.

Severity, from who depends on what
  SEVERE    a USED model is DEAD/WRONG and NONE of its fallbacks is alive,
            or a used model is absent from the gateway  -> exit 1
  DEGRADED  a used model is dead but a fallback answers  -> reported, exit 0
            (calling that "severe" every morning would teach everyone to ignore the alarm)
  noted     a dead model nobody uses                     -> reported, exit 0
  BLIND     gateway unreachable, no key, or EVERY model dead (that is the gateway
            or the key, not the models)                  -> exit 2

    python model_probe.py               one pass
    python model_probe.py --self-test   witnesses, no network

Environment:
    PROBE_GATEWAY      gateway base URL (default http://127.0.0.1:4000)
    PROBE_KEY_FILE     file holding the gateway key: either the bare key, or KEY=VALUE lines
                       with LITELLM_MASTER_KEY (read from a FILE, never from argv)
    PROBE_CONFIG       LiteLLM config.yaml: model list and router_settings.fallbacks
                       (optional: without it, models come from GET /v1/models and no fallback is known)
    PROBE_CONSUMERS    JSON file {"model": ["who uses it", ...]}
    PROBE_AGENT_STORE  JSON file with agents[].name and agents[].llm.model (e.g. an agent platform store)
    PROBE_NEEDS        extra "model:who,model:who" declarations
    PROBE_KINDS        "model:vision,model:embeddings" overrides (default: guessed from the name)
    PROBE_OUT_DIR      where the JSON result is written (default ./model-probe)
    PROBE_SLOW_S       default 60

Standard library only. Python 3.8+.
"""
import datetime
import io
import json
import os
import random
import re
import socket
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

GATEWAY = os.environ.get("PROBE_GATEWAY", "http://127.0.0.1:4000").rstrip("/")
KEY_FILE = os.environ.get("PROBE_KEY_FILE", "")
CONFIG = os.environ.get("PROBE_CONFIG", "")
CONSUMERS = os.environ.get("PROBE_CONSUMERS", "")
AGENT_STORE = os.environ.get("PROBE_AGENT_STORE", "")
NEEDS = os.environ.get("PROBE_NEEDS", "")
KINDS = os.environ.get("PROBE_KINDS", "")
OUT_DIR = os.environ.get("PROBE_OUT_DIR", "model-probe")
SLOW_S = float(os.environ.get("PROBE_SLOW_S", "60"))
ALIVE = ("OK", "SLOW")


def gateway_key(path=None):
    path = path or KEY_FILE
    try:
        text = io.open(path, encoding="utf-8").read()
    except (OSError, TypeError):
        return ""
    for line in text.splitlines():
        if line.strip().startswith("LITELLM_MASTER_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    return lines[0] if len(lines) == 1 and "=" not in lines[0] else ""


def models_from_config(text):
    return list(dict.fromkeys(re.findall(r"^\s*-\s*model_name:\s*[\"']?([\w.\-/:]+)", text, re.M)))


def fallbacks_from_config(text):
    """{model: [fallbacks]} from router_settings.fallbacks (list-of-mappings form, YAML block or flow)."""
    out = {}
    block = re.search(r"^\s*fallbacks:\s*(.*?)(?=^\S|\Z)", text, re.M | re.S)
    if not block:
        return out
    body = block.group(1)
    for m in re.finditer(r"[\"']?([\w.\-/:]+)[\"']?\s*:\s*\[([^\]]*)\]", body):       # - {a: [b, c]}
        out[m.group(1)] = [x.strip().strip("\"'") for x in m.group(2).split(",") if x.strip()]
    current = None
    for line in body.splitlines():                                                    # - a:\n    - b
        m = re.match(r"^\s*-\s*[\"']?([\w.\-/:]+)[\"']?:\s*$", line)
        if m:
            current = m.group(1)
            out.setdefault(current, [])
            continue
        m = re.match(r"^\s*-\s*[\"']?([\w.\-/:]+)[\"']?\s*$", line)
        if m and current:
            out[current].append(m.group(1))
    return out


def consumers():
    """{model: [who uses it]} from PROBE_CONSUMERS, PROBE_AGENT_STORE and PROBE_NEEDS."""
    out = {}
    if CONSUMERS:
        try:
            for m, who in json.load(io.open(CONSUMERS, encoding="utf-8")).items():
                out.setdefault(m, []).extend(who if isinstance(who, list) else [str(who)])
        except (OSError, ValueError, AttributeError):
            pass
    if AGENT_STORE:
        try:
            for a in json.load(io.open(AGENT_STORE, encoding="utf-8")).get("agents", []):
                m = (a.get("llm") or {}).get("model")
                if m:
                    out.setdefault(m, []).append("agent " + str(a.get("name", "?"))[:40])
        except (OSError, ValueError, AttributeError):
            pass
    for item in [x for x in NEEDS.split(",") if ":" in x]:
        m, who = item.rsplit(":", 1)
        out.setdefault(m.strip(), []).append(who.strip())
    return out


def kind(name, overrides=None):
    if overrides is None:
        overrides = dict(x.rsplit(":", 1) for x in KINDS.split(",") if ":" in x)
    if name in overrides:
        return overrides[name].strip()
    n = name.lower()
    if "embed" in n:
        return "embeddings"
    if "vision" in n or "-vl" in n or "ocr" in n:
        return "vision"
    return "text"


# --------------------------------------------------------------------- the vision test image
_DIGITS = {  # 5x7 bitmap font
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}


def number_png(number, scale=16, margin=3):
    """A black-on-white PNG of the digits, standard library only (no Pillow, no ffmpeg)."""
    cols = len(number) * 6 - 1 + 2 * margin
    rows = 7 + 2 * margin
    grid = [[0] * cols for _ in range(rows)]
    for i, ch in enumerate(number):
        for y, line in enumerate(_DIGITS[ch]):
            for x, bit in enumerate(line):
                if bit == "1":
                    grid[margin + y][margin + i * 6 + x] = 1
    w, h = cols * scale, rows * scale
    raw = bytearray()
    for y in range(h):
        raw.append(0)                                   # filter type: none
        row = grid[y // scale]
        raw.extend(0 if row[x // scale] else 255 for x in range(w))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


# --------------------------------------------------------------------- calls
def http(path, body, key, timeout=150):
    req = urllib.request.Request(GATEWAY + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))
        return r, None, time.time() - t0
    except urllib.error.HTTPError as e:
        return None, "HTTP %s" % e.code, time.time() - t0
    except Exception as e:                                         # noqa: BLE001
        return None, str(e)[:80], time.time() - t0


def test_model(name, key, call=http, kinds=None, number=None):
    """(state, seconds, detail) with state in OK / SLOW / WRONG / DEAD.

    `number` fixes the vision test's digits (witnesses only); by default they are random,
    so that a model cannot pass by repeating a remembered answer."""
    k = kind(name, kinds)
    if k == "embeddings":
        r, err, d = call("/v1/embeddings", {"model": name, "input": "probe"}, key)
        ok = bool(r and (r.get("data") or [{}])[0].get("embedding"))
        expected = "a vector"
    else:
        if k == "vision":
            expected = number or "%04d" % random.randint(1000, 9999)
            import base64
            content = [{"type": "text", "text": "Copy exactly the number shown in the image. Answer with the digits only."},
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                                           + base64.b64encode(number_png(expected)).decode()}}]
        else:
            content, expected = "Answer with the single word: READY", "READY"
        body = {"model": name, "temperature": 0, "max_tokens": 2000, "disable_fallbacks": True,
                "messages": [{"role": "user", "content": content}]}
        r, err, d = call("/v1/chat/completions", body, key)
        text = ((r or {}).get("choices") or [{}])[0].get("message", {}).get("content") or ""
        ok = expected in text.upper()
        if r is not None and not ok:
            return "WRONG", d, "answer without %r: %r" % (expected, text[:60])
    if err or r is None:
        return "DEAD", d, err or "empty"
    if not ok:
        return "WRONG", d, "%s test failed" % expected
    return ("SLOW" if d > SLOW_S else "OK"), d, ""


def classify(states, deps, fallbacks):
    """(severe, degraded): SEVERE = used, dead, and NO live fallback; DEGRADED = a fallback answers."""
    severe, degraded = [], []
    for name, state in states.items():
        if state not in ("DEAD", "WRONG") or not deps.get(name):
            continue
        alive = [f for f in fallbacks.get(name, []) if states.get(f) in ALIVE]
        (degraded if alive else severe).append((name, alive))
    return severe, degraded


def one_pass():
    key = gateway_key()
    if not key:
        print("BLIND: no gateway key (PROBE_KEY_FILE)")
        return 2
    cfg = ""
    if CONFIG:
        try:
            cfg = io.open(CONFIG, encoding="utf-8").read()
        except OSError as e:
            print("BLIND: config unreadable (%s)" % e)
            return 2
    names = models_from_config(cfg) if cfg else []
    if not names:
        r, err, _ = http("/v1/models", None, key, timeout=30)
        if r is None:
            print("BLIND: gateway unreachable (%s)" % err)
            return 2
        names = [m.get("id") for m in r.get("data", []) if m.get("id")]
    deps, fb = consumers(), fallbacks_from_config(cfg)
    results = []
    for name in names:
        state, d, detail = test_model(name, key)
        who = deps.get(name, [])
        results.append({"model": name, "kind": kind(name), "state": state, "seconds": round(d, 1),
                        "detail": detail, "used_by": who})
        print("  %-24s %-10s %-5s %5.1f s  %s%s" % (name, kind(name), state, d,
                                                    ("used by: " + ", ".join(who)) if who else "(no consumer)",
                                                    ("  -- " + detail) if detail else ""))
    if results and all(r["state"] == "DEAD" for r in results):
        print("BLIND: EVERY model is dead -- that is the gateway or the key, not the models")
        return 2
    states = {r["model"]: r["state"] for r in results}
    severe_t, degraded = classify(states, deps, fb)
    severe = [n for n, _ in severe_t]
    for n, alive in degraded:
        print("  DEGRADED %s: dead, served by %s" % (n, ", ".join(alive)))
    absent = sorted(set(deps) - set(names))
    for m in absent:
        print("  %-24s ABSENT from the gateway, used by: %s" % (m, ", ".join(deps[m])))
        severe.append(m)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    out = os.path.join(OUT_DIR, "probe_%s.json" % stamp)
    io.open(out, "w", encoding="utf-8").write(json.dumps(
        {"at": stamp, "host": socket.gethostname(), "results": results, "absent": absent, "severe": severe,
         "degraded": [{"model": n, "served_by": a} for n, a in degraded]}, ensure_ascii=False, indent=1))
    print("%d model(s) tested, %d severe: %s ; %d degraded ; result: %s"
          % (len(results), len(severe), ", ".join(severe) or "none", len(degraded), out))
    return 1 if severe else 0


# --------------------------------------------------------------------- witnesses
def self_test(verbose=True):
    failures = []

    def check(cond, name):
        if verbose:
            print("  %-4s %s" % ("ok" if cond else "FAIL", name))
        if not cond:
            failures.append(name)

    answer = lambda text: (lambda p, b, k: ({"choices": [{"message": {"content": text}}]}, None, 1.0))
    check(test_model("chat-free", "k", answer("READY."))[0] == "OK", "1. text: the word comes back -> OK")
    check(test_model("chat-free", "k", answer("Hello!"))[0] == "WRONG", "2. text: answers beside the point -> WRONG")
    check(test_model("chat-free", "k", lambda p, b, k: (None, "HTTP 404", 0.3))[0] == "DEAD", "3. 404 -> DEAD")
    check(test_model("chat-free", "k", lambda p, b, k: ({"choices": [{"message": {"content": "READY"}}]}, None, 75.0))[0]
          == "SLOW", "4. above the latency floor -> SLOW")
    seen = {}

    def spy(p, b, k):
        seen.update(b)
        return {"choices": [{"message": {"content": "READY"}}]}, None, 1.0
    test_model("chat-free", "k", spy)
    check(seen.get("disable_fallbacks") is True, "5. fallbacks are DISABLED: the model is judged, not the chain")
    asked = {}

    def reader(p, b, k):
        url = b["messages"][0]["content"][1]["image_url"]["url"]
        asked["png"] = url
        return {"choices": [{"message": {"content": "1234"}}]}, None, 1.0
    right = test_model("vision-model", "k", reader, number="1234")[0]
    wrong = test_model("vision-model", "k", reader, number="4729")[0]
    check(asked.get("png", "").startswith("data:image/png;base64,iVBORw0KGgo") and right == "OK" and wrong == "WRONG",
          "6. vision: a real PNG is sent; the right digits -> OK, other digits -> WRONG")
    png = number_png("4729")
    w, h = struct.unpack(">II", png[16:24])
    check(png[:8] == b"\x89PNG\r\n\x1a\n" and (w, h) == ((4 * 6 - 1 + 6) * 16, 13 * 16)
          and zlib.decompress(png[41:41 + struct.unpack(">I", png[33:37])[0]])[0] == 0,
          "7. the PNG is well-formed (signature, size, compressed rows)")
    check(kind("text-embedding-3") == "embeddings" and kind("agent-vision-2") == "vision" and kind("chat") == "text"
          and kind("chat", {"chat": "vision"}) == "vision", "8. kind guessed from the name, overridable")
    sev, deg = classify({"fast": "DEAD", "backup": "OK", "unused": "DEAD", "x": "DEAD", "y": "DEAD"},
                        {"fast": ["agent A"], "x": ["agent B"]}, {"fast": ["backup"], "x": ["y"]})
    check(deg == [("fast", ["backup"])] and sev == [("x", [])],
          "9. dead + live fallback = DEGRADED ; dead + dead fallbacks = SEVERE ; unused = nothing")
    cfg = ("model_list:\n  - model_name: fast\n  - model_name: backup\nrouter_settings:\n  fallbacks:\n"
           "    - fast:\n      - backup\n    - {x: [y, z]}\ngeneral_settings:\n  master_key: os.environ/K\n")
    check(models_from_config(cfg) == ["fast", "backup"] and fallbacks_from_config(cfg) == {"fast": ["backup"], "x": ["y", "z"]},
          "10. models and fallbacks read from a LiteLLM config (block and flow forms)")
    if verbose:
        print("  %d witness(es), %d failure(s)" % (10, len(failures)))
    return len(failures)


if __name__ == "__main__":
    sys.exit((1 if self_test() else 0) if "--self-test" in sys.argv else one_pass())
