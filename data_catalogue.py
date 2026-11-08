#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data catalogue measured on each host: what data exists, how big, who writes it, who can read it.

Why: a map of services, ports and tools did not list the two largest assets on
the host -- a mail archive of tens of gigabytes and a search index of hundreds of
thousands of rows. Nobody had forgotten them; the map simply never looked at
data. Agents kept asking "where is X?" and rediscovering the answer.

What it records, per host (run as root, locally or over SSH):
  1. data directories above a size threshold, under configurable roots;
  2. SQLite databases: tables and row counts;
  3. Postgres databases inside Docker containers: largest tables (estimated rows, size);
  4. for each item: the systemd unit(s) that mention its path (probable writer),
     its mode and owner, and a WORLD-READABLE flag.
What it never does: read a row's content, print a value, or open a database
whose path matches CATALOGUE_NEVER_OPEN (it is counted, not opened).

    python data_catalogue.py --local                     -> CATALOGUE.md for this host
    python data_catalogue.py host-a host-b --out CATALOGUE.md
                                                         (ssh host "sudo python3 -I - --collect")
    python data_catalogue.py --collect                   JSON facts on stdout (what SSH runs)
    python data_catalogue.py --self-test

Environment:
    CATALOGUE_ROOTS        colon-separated roots (default /opt:/srv:/var/lib:/home)
    CATALOGUE_MIN_MB       directory threshold in MB (default 200)
    CATALOGUE_DEPTH        directory depth below each root (default 3)
    CATALOGUE_NEVER_OPEN   comma-separated substrings: matching databases are counted, never opened
    CATALOGUE_UNITS_DIR    where unit files live (default /etc/systemd/system)

A host that cannot be read is reported BLIND, never empty. If any string shaped
like a secret appears in the rendered catalogue, NOTHING is written (exit 3).
Standard library only. Python 3.8+.
"""
import datetime
import glob
import io
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys

ROOTS = [r for r in os.environ.get("CATALOGUE_ROOTS", "/opt:/srv:/var/lib:/home").split(os.pathsep) if r]
MIN_MB = int(os.environ.get("CATALOGUE_MIN_MB", "200"))
DEPTH = int(os.environ.get("CATALOGUE_DEPTH", "3"))
NEVER_OPEN = tuple(s.strip() for s in os.environ.get("CATALOGUE_NEVER_OPEN", "").split(",") if s.strip())
UNITS_DIR = os.environ.get("CATALOGUE_UNITS_DIR", "/etc/systemd/system")
NOISE = ("/.venv", "/venv", "/node_modules", "/go/pkg", "site-packages", "/.cache", "/.npm", "/.git/", "/__pycache__",
         "/overlay2/")      # container image layers are not data
SQLITE_EXT = (".db", ".sqlite", ".sqlite3")
SKIP_SQLITE = ("/backups/", "/backup/", ".bak", "/.venv/", "site-packages", "/node_modules/", "/go/pkg/mod/")

# Value shapes, not field names: API-key prefixes, bot tokens, GitHub tokens, long base64 runs.
SECRET_SHAPE = re.compile(r"(?<![/\w.\-])(?:sk-[A-Za-z0-9_\-]{16,}|[0-9]{8,}:[A-Za-z0-9_\-]{30,}|"
                          r"gh[pousr]_[A-Za-z0-9]{30,}|[A-Za-z0-9+]{40,}={0,2})(?![/\w])")


def secrets_in(text):
    return SECRET_SHAPE.findall(text or "")


# --------------------------------------------------------------------- collection (runs on the host)
def _rights(path):
    try:
        s = os.stat(path)
    except OSError:
        return {}
    owner = "%s:%s" % (s.st_uid, s.st_gid)
    try:
        import grp
        import pwd
        owner = "%s:%s" % (pwd.getpwuid(s.st_uid).pw_name, grp.getgrgid(s.st_gid).gr_name)
    except Exception:                                              # noqa: BLE001
        pass
    return {"mode": oct(s.st_mode & 0o777)[2:], "owner": owner, "world_readable": bool(s.st_mode & stat.S_IROTH)}


def load_units(units_dir=None):
    units = {}
    for u in glob.glob(os.path.join(units_dir or UNITS_DIR, "*.service")):
        try:
            units[os.path.basename(u)] = io.open(u, encoding="utf-8", errors="replace").read()
        except OSError:
            pass
    return units


def writers(path, units):
    """Units that mention the path or one of its parents.

    A service usually names its root directory, rarely the data sub-directory: walk up."""
    p = path.replace("\\", "/")
    while p.count("/") >= 2:
        hits = sorted(n for n, text in units.items() if p in text)
        if hits:
            return hits[:4]
        p = p.rsplit("/", 1)[0]
    return []


def _walk(root):
    """os.walk that stays on the root's filesystem (like du -x) and skips NOISE."""
    try:
        dev = os.stat(root).st_dev
    except OSError:
        return
    for cur, dirs, files in os.walk(root, onerror=lambda e: None):
        if any(n in cur.replace("\\", "/") + "/" for n in NOISE):
            dirs[:] = []
            continue
        keep = []
        for d in dirs:
            try:
                if os.lstat(os.path.join(cur, d)).st_dev == dev:
                    keep.append(d)
            except OSError:
                pass
        dirs[:] = keep
        yield cur, dirs, files


def dir_sizes(roots, depth, min_mb):
    """{dir: bytes} for directories up to `depth` levels below each root, at or above min_mb."""
    totals = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip("/\\").count(os.sep)
        for cur, dirs, files in _walk(root):
            size = 0
            for f in files:
                try:
                    st = os.lstat(os.path.join(cur, f))
                    if stat.S_ISREG(st.st_mode):
                        size += st.st_size
                except OSError:
                    pass
            # attribute the files' size to every ancestor up to the allowed depth
            parts = cur.rstrip("/\\").split(os.sep)
            for lvl in range(base, min(len(parts) - 1, base + depth) + 1):
                anc = os.sep.join(parts[:lvl + 1]) or os.sep
                totals[anc] = totals.get(anc, 0) + size
    floor = min_mb * 1024 * 1024
    return {d: b for d, b in totals.items() if b >= floor and d.rstrip("/\\") not in [r.rstrip("/\\") for r in roots]}


def sqlite_files(roots):
    out = []
    for root in roots:
        for cur, dirs, files in _walk(root):
            for f in files:
                full = os.path.join(cur, f)
                if f.endswith(SQLITE_EXT) and not any(s in full.replace("\\", "/") for s in SKIP_SQLITE):
                    try:
                        if os.path.getsize(full) > 0:
                            out.append(full)
                    except OSError:
                        pass
    return sorted(out)


def sqlite_tables(path):
    if any(s in path for s in NEVER_OPEN):
        return "(counted, never opened)"
    try:
        c = sqlite3.connect("file:%s?mode=ro" % path.replace("\\", "/"), uri=True, timeout=5)
        try:
            t = {}
            for (n,) in c.execute("select name from sqlite_master where type='table' "
                                  "and name not like 'sqlite_%' and name not like '%_fts_%'"):
                try:
                    t[n] = c.execute('select count(*) from "%s"' % n.replace('"', '""')).fetchone()[0]
                except sqlite3.Error:
                    t[n] = "?"
            return t
        finally:
            c.close()
    except sqlite3.Error as e:
        return "unreadable: %s" % str(e)[:60]


def _sh(args, timeout=120):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def postgres_in_docker():
    out = []
    for line in _sh(["docker", "ps", "--format", "{{.Names}} {{.Image}}"]).splitlines():
        name, image = (line.split(" ", 1) + [""])[:2]
        if not re.search("postgres|pgvector|timescale", image + name, re.I):
            continue
        user = _sh(["docker", "exec", name, "printenv", "POSTGRES_USER"]).strip() or "postgres"
        dbs = [b for b in _sh(["docker", "exec", name, "psql", "-U", user, "-At", "-c",
                               "select datname from pg_database where not datistemplate"]).split() if b]
        for db in dbs:
            q = ("select n.nspname||'.'||c.relname, greatest(c.reltuples,0)::bigint, "
                 "pg_total_relation_size(c.oid)/1048576 from pg_class c join pg_namespace n on n.oid=c.relnamespace "
                 "where c.relkind='r' and n.nspname not in ('pg_catalog','information_schema') "
                 "order by pg_total_relation_size(c.oid) desc limit 12")
            rows = [l.split("|") for l in _sh(["docker", "exec", name, "psql", "-U", user, "-d", db, "-At", "-c", q]).splitlines()
                    if l.count("|") == 2]
            out.append({"container": name, "database": db,
                        "tables": [{"table": t, "rows": int(r), "mb": int(m)} for t, r, m in rows]})
    return out


def collect(roots=None, units_dir=None, depth=None, min_mb=None, with_docker=True):
    roots = roots or ROOTS
    units = load_units(units_dir)
    facts = {"directories": [], "sqlite": [], "postgres": []}
    for d, b in sorted(dir_sizes(roots, DEPTH if depth is None else depth, MIN_MB if min_mb is None else min_mb).items()):
        e = {"path": d.replace("\\", "/"), "mb": b // (1024 * 1024), "writers": writers(d, units)}
        e.update(_rights(d))
        facts["directories"].append(e)
    for f in sqlite_files(roots):
        e = {"path": f.replace("\\", "/"), "mb": round(os.path.getsize(f) / 1e6, 1),
             "writers": writers(os.path.dirname(f), units), "tables": sqlite_tables(f)}
        e.update(_rights(f))
        facts["sqlite"].append(e)
    if with_docker:
        facts["postgres"] = postgres_in_docker()
    return facts


# --------------------------------------------------------------------- remote + render (runs on the operator side)
def collect_remote(host):
    src = io.open(os.path.abspath(__file__), encoding="utf-8").read()
    p = subprocess.run(["ssh", host, "sudo python3 -I - --collect"], input=src, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=1800)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError((p.stderr or "empty output")[-200:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def _n(v):
    return format(v, ",").replace(",", " ") if isinstance(v, int) else str(v)


def _flag(e):
    return " **WORLD-READABLE**" if e.get("world_readable") else ""


def render(facts_by_host, when):
    L = ["# Data catalogue -- what exists, how big, who writes it, who can read it", "",
         "*Generated by `data_catalogue.py`, by measurement, on %s. No content and no value: paths, sizes, counts "
         "and rights. Read it BEFORE searching for \"where is X?\".*" % when, ""]
    for host, f in facts_by_host.items():
        L += ["## %s" % host, ""]
        if isinstance(f, str):
            L += ["BLIND: %s" % f, ""]
            continue
        L += ["### Postgres databases (containers)", ""]
        for e in f.get("postgres", []):
            L.append("- **%s / %s** -- largest tables: %s" % (e["container"], e["database"], ", ".join(
                "%s (%s rows, %s MB)" % (t["table"], _n(t["rows"]), t["mb"]) for t in e["tables"][:6]) or "none"))
        if not f.get("postgres"):
            L.append("- none found")
        L += ["", "### SQLite databases", "", "| database | MB | tables (rows) | written by | rights |", "|---|---|---|---|---|"]
        for e in sorted(f.get("sqlite", []), key=lambda x: -x["mb"]):
            t = e["tables"] if isinstance(e["tables"], str) else ", ".join(
                "%s %s" % (k, _n(v)) for k, v in list(e["tables"].items())[:6])
            L.append("| %s | %s | %s | %s | %s %s%s |" % (e["path"], e["mb"], t, ", ".join(e["writers"]) or "--",
                                                          e.get("mode", "?"), e.get("owner", ""), _flag(e)))
        L += ["", "### Data directories (>= threshold)", "", "| directory | size | written by | rights |", "|---|---|---|---|"]
        for d in sorted(f.get("directories", []), key=lambda x: -x["mb"]):
            size = ("%.1f GB" % (d["mb"] / 1024)) if d["mb"] >= 1024 else ("%d MB" % d["mb"])
            L.append("| %s | %s | %s | %s %s%s |" % (d["path"], size, ", ".join(d["writers"]) or "--",
                                                     d.get("mode", "?"), d.get("owner", ""), _flag(d)))
        L.append("")
    return chr(10).join(L) + chr(10)


def write_catalogue(facts_by_host, out, when=None):
    """Render and write. Returns 0, 2 if a host is blind, 3 (and writes nothing) on a secret shape."""
    when = when or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    text = render(facts_by_host, when)
    leaks = secrets_in(text)
    if leaks:
        print("REFUSED: %d secret-shaped string(s) in the catalogue -- nothing written" % len(leaks))
        return 3
    io.open(out, "w", encoding="utf-8", newline="\n").write(text)
    blind = [h for h, f in facts_by_host.items() if isinstance(f, str)]
    print("written: %s ; blind: %s" % (out, ", ".join(blind) or "none"))
    return 2 if blind else 0


# --------------------------------------------------------------------- witnesses
def self_test(verbose=True):
    import tempfile
    failures = []

    def check(cond, name):
        if verbose:
            print("  %-4s %s" % ("ok" if cond else "FAIL", name))
        if not cond:
            failures.append(name)

    sample = {"host-a": {"directories": [{"path": "/srv/mail-archive", "mb": 45056, "writers": ["mail-sync.service"],
                                          "mode": "775", "owner": "svc:svc", "world_readable": True}],
                         "sqlite": [{"path": "/var/lib/private-bot/runs.db", "mb": 2.0, "writers": [],
                                     "tables": "(counted, never opened)", "mode": "600", "owner": "root:root"}],
                         "postgres": [{"container": "pg", "database": "app",
                                       "tables": [{"table": "public.chunks", "rows": 686183, "mb": 900}]}]},
              "host-b": "ssh: connection refused"}
    t = render(sample, "x")
    check("mail-archive | 44.0 GB" in t, "1. a large directory is rendered with its size in GB")
    check("**WORLD-READABLE**" in t, "2. the world-readable flag is rendered")
    check("never opened" in t, "3. a never-open database is counted, not opened")
    check("public.chunks (686 183 rows" in t, "4. Postgres tables are rendered with their row counts")
    check("BLIND: ssh" in t, "5. an unreadable host is BLIND, not empty")
    check(not secrets_in(t), "6. no false secret on a healthy catalogue (long paths included)")
    trap = json.loads(json.dumps(sample))
    trap["host-a"]["directories"][0]["owner"] = "sk-" + "a" * 30
    out = os.path.join(tempfile.mkdtemp(), "C.md")
    check(write_catalogue(trap, out, "x") == 3 and not os.path.exists(out),
          "7. a secret shape in the facts -> nothing is written")

    # End to end on a real temporary tree.
    root = tempfile.mkdtemp()
    data = os.path.join(root, "app", "data")
    os.makedirs(data)
    with open(os.path.join(data, "blob.bin"), "wb") as fh:
        fh.write(b"0" * (2 * 1024 * 1024))
    c = sqlite3.connect(os.path.join(data, "app.sqlite"))
    c.executescript("create table items(x); insert into items values (1),(2),(3); create table logs(y);")
    c.commit()
    c.close()
    os.makedirs(os.path.join(root, "secret-bot"))
    c = sqlite3.connect(os.path.join(root, "secret-bot", "private.db"))
    c.executescript("create table messages(x); insert into messages values ('never read');")
    c.commit()
    c.close()
    units = os.path.join(root, "units")
    os.makedirs(units)
    io.open(os.path.join(units, "app-writer.service"), "w").write("[Service]\nWorkingDirectory=%s\n"
                                                                  % os.path.join(root, "app").replace("\\", "/"))
    global NEVER_OPEN
    saved, NEVER_OPEN = NEVER_OPEN, ("secret-bot",)
    try:
        f = collect([root], units_dir=units, depth=2, min_mb=1, with_docker=False)
    finally:
        NEVER_OPEN = saved
    db = [e for e in f["sqlite"] if e["path"].endswith("app.sqlite")]
    check(bool(db) and db[0]["tables"] == {"items": 3, "logs": 0}, "8. a real SQLite file: tables and row counts")
    check(bool(db) and db[0]["writers"] == ["app-writer.service"], "9. the writer is found by walking up to the unit's path")
    pv = [e for e in f["sqlite"] if e["path"].endswith("private.db")]
    check(bool(pv) and pv[0]["tables"] == "(counted, never opened)", "10. a never-open database is not opened")
    check(any(d["path"].endswith("app/data") and d["mb"] >= 2 for d in f["directories"]),
          "11. a data directory above the threshold is listed with its size")
    if verbose:
        print("  %d witness(es), %d failure(s)" % (11, len(failures)))
    return len(failures)


def main(argv):
    if "--self-test" in argv:
        return 1 if self_test() else 0
    if "--collect" in argv:
        print(json.dumps(collect()))
        return 0
    out = argv[argv.index("--out") + 1] if "--out" in argv else "CATALOGUE.md"
    hosts = [a for i, a in enumerate(argv[1:], 1) if not a.startswith("--") and argv[i - 1] != "--out"]
    facts = {}
    if "--local" in argv:
        import socket
        facts[socket.gethostname()] = collect()
    for h in hosts:
        try:
            facts[h] = collect_remote(h)
        except Exception as e:                                     # noqa: BLE001
            facts[h] = str(e)[:200]
    if not facts:
        print(__doc__)
        return 2
    return write_catalogue(facts, out)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
