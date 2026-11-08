#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Claude Code UserPromptSubmit hook: recall WHERE THINGS ARE and HOW A PROBLEM WAS ALREADY SOLVED,
at the moment the user's message mentions them.

Why: the operator's complaint, verbatim in substance -- "you know nothing, you don't
know where it is, you have to search, and we solve the same problem dozens of
times". The answers existed, in files the agent could read. It did not read them,
because nothing made it.

So the recall does not depend on the agent's goodwill: it starts from the MESSAGE.
Each entry of recall.json has a regular expression; when the user's prompt matches,
one short POINTER line is added to the agent's context. Pointers, not content: a
path, a count, a command -- never a value, never a secret. At most MAX_LINES lines
and MAX_BYTES bytes per message, so that a message touching everything does not
flood the context.

    (hook)  reads the hook JSON on stdin, prints the pointer lines on stdout
    python recall_hook.py --render RECURRING_PROBLEMS.md    human-readable registry from the same JSON
    python recall_hook.py --self-test

recall.json has two lists:
    assets    {id, pattern, line}                              where something is
    problems  {id, pattern, symptom, cause, solution, guard}   a problem already solved

Environment: RECALL_FILE (default: recall.json next to this script).
Never blocks: any error, a broken JSON or a broken pattern means silence, exit 0.
Standard library only. Python 3.8+.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RECALL_FILE", os.path.join(HERE, "recall.json"))
MAX_LINES = 5
MAX_BYTES = 2400
HEADER = "RECALL (%s) -- what already exists about what this message mentions:"


def load(path=None):
    try:
        return json.load(io.open(path or DATA, encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return {}


def recall(prompt, data=None):
    data = load() if data is None else data
    text = prompt or ""
    out = []
    for p in data.get("problems", []):
        try:
            if re.search(p["pattern"], text, re.I):
                out.append("[%s] %s -> %s" % (p["id"], p["symptom"], p["solution"]))
        except (re.error, KeyError, TypeError):
            continue
    for a in data.get("assets", []):
        try:
            if re.search(a["pattern"], text, re.I):
                out.append("[%s] %s" % (a["id"], a["line"]))
        except (re.error, KeyError, TypeError):
            continue
    kept, total = [], 0
    for line in out[:MAX_LINES]:
        if total + len(line) > MAX_BYTES:
            break
        kept.append(line)
        total += len(line)
    if kept:
        kept.insert(0, HEADER % os.path.basename(DATA))
    return kept


def render(data=None):
    data = load() if data is None else data
    cell = lambda v: str(v or "").replace("|", "/").replace(chr(10), " ")
    L = ["# Recurring problems -- symptom, cause, solution, guard", "",
         "*Rendered from `%s` by `recall_hook.py --render`. Do not edit here: add the entry to the JSON. "
         "Each entry is also RECALLED automatically when a user message matches its pattern.*" % os.path.basename(DATA), "",
         "| id | symptom | cause | solution | automatic guard |", "|---|---|---|---|---|"]
    for p in data.get("problems", []):
        L.append("| %s |" % " | ".join(cell(p.get(k)) for k in ("id", "symptom", "cause", "solution", "guard")))
    L += ["", "## Where things are (recalled assets)", ""]
    for a in data.get("assets", []):
        L.append("- **%s** -- %s" % (a.get("id"), a.get("line")))
    return chr(10).join(L) + chr(10)


def prompt_from_stdin(raw):
    """The hook receives JSON with a `prompt` field; plain text is accepted for manual use."""
    try:
        d = json.loads(raw)
        if isinstance(d, dict):
            return str(d.get("prompt") or "")
    except ValueError:
        pass
    return raw


# --------------------------------------------------------------------- witnesses
def self_test(verbose=True):
    failures = []

    def check(cond, name):
        if verbose:
            print("  %-4s %s" % ("ok" if cond else "FAIL", name))
        if not cond:
            failures.append(name)

    d = load(os.path.join(HERE, "recall.example.json"))
    check(bool(d.get("assets")) and bool(d.get("problems")), "1. the example data loads")
    bad = []
    for e in d.get("assets", []) + d.get("problems", []):
        try:
            re.compile(e["pattern"])
        except re.error:
            bad.append(e["id"])
    check(not bad, "2. every pattern compiles %s" % (bad or ""))
    r = recall("Where is the mail archive? Is it in the database?", d)
    check(any("A-MAIL-ARCHIVE" in l for l in r), "3. 'mail archive' recalls the asset")
    check(any("P-INVENTORY-GAP" in l for l in r), "4. 'where is ... database' recalls the known problem")
    check(recall("ok thanks", d) == [], "5. a banal message triggers nothing (an alarm that always rings gets muted)")
    check(any("P-EMPTY-START" in l for l in recall("I ran systemctl start and it returned 0", d)),
          "6. a known symptom is recognised")
    everything = ("where is the email database? the model is dead, fallback? memory, secret token, "
                  "systemctl start returned 0, it doesn't exist, bot is silent")
    raw = sum(1 for e in d.get("assets", []) + d.get("problems", []) if re.search(e["pattern"], everything, re.I))
    big = recall(everything, d)
    # Literal 5 and 2400: comparing with the module constants would make the witness pass when they are broken.
    check(raw > 5 and len(big) <= 5 + 1 and sum(map(len, big)) <= 2400 + 200,
          "7. the cap holds on a message touching everything (%d entries match, %d lines kept)" % (raw, max(len(big) - 1, 0)))
    check(recall("mail", {"assets": [{"id": "X", "pattern": "(", "line": "x"}]}) == [], "8. a broken pattern crashes nothing")
    check(prompt_from_stdin(json.dumps({"prompt": "hello", "session_id": "s"})) == "hello",
          "9. the hook JSON on stdin is parsed")
    reg = render(d)
    check("| P-EMPTY-START |" in reg and "**A-MAIL-ARCHIVE**" in reg, "10. the registry renders both lists")
    if verbose:
        print("  %d witness(es), %d failure(s)" % (10, len(failures)))
    return len(failures)


def main(argv):
    if "--self-test" in argv:
        return 1 if self_test() else 0
    if "--render" in argv:
        out = argv[argv.index("--render") + 1]
        io.open(out, "w", encoding="utf-8", newline="\n").write(render())
        print("written: %s" % out)
        return 0
    try:
        lines = recall(prompt_from_stdin(sys.stdin.read()))
        if lines:
            sys.stdout.write(chr(10).join(lines) + chr(10))
    except Exception:                                              # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
