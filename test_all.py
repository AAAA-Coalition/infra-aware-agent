#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run every witness, then BREAK each guard on purpose and check that a witness turns red.

A green that cannot turn red proves nothing.

    python test_all.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import data_catalogue  # noqa: E402
import model_probe     # noqa: E402
import recall_hook     # noqa: E402


def main():
    total = 0
    for name, mod in (("data_catalogue", data_catalogue), ("recall_hook", recall_hook), ("model_probe", model_probe)):
        print(chr(10) + "== " + name)
        total += mod.self_test()

    print(chr(10) + "== mutations: each broken guard must be caught")
    missed = 0

    def mutation(name, mod, attr, value, run):
        nonlocal missed
        saved = getattr(mod, attr)
        setattr(mod, attr, value)
        try:
            caught = run(verbose=False) > 0
        finally:
            setattr(mod, attr, saved)
        print("  %-4s %s" % ("ok" if caught else "FAIL", name + (" -> caught" if caught else " -> NOT caught")))
        missed += 0 if caught else 1

    never = re.compile(r"(?!x)x")
    mutation("secret-shape guard disabled", data_catalogue, "SECRET_SHAPE", never, data_catalogue.self_test)
    mutation("never-open list ignored", data_catalogue, "sqlite_tables",
             lambda p: {"messages": 1}, data_catalogue.self_test)
    mutation("recall cap removed", recall_hook, "MAX_LINES", 10 ** 6, recall_hook.self_test)
    mutation("fallbacks no longer disabled", model_probe, "test_model",
             (lambda real: lambda n, k, call=None, kinds=None, number=None:
              real(n, k, (lambda p, b, kk: call(p, dict(b, disable_fallbacks=False), kk)) if call else None, kinds, number))
             (model_probe.test_model), model_probe.self_test)
    mutation("a live fallback no longer downgrades to DEGRADED", model_probe, "ALIVE", (), model_probe.self_test)

    print(chr(10) + "TOTAL: %d witness failure(s), %d mutation(s) not caught" % (total, missed))
    return 1 if (total or missed) else 0


if __name__ == "__main__":
    sys.exit(main())
