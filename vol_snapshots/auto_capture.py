"""Opportunistic capture: snapshot the chains whenever the laptop is on.

The daily capture only works if something runs it at the right time, and a
fixed-time task misses every day the machine is off at that minute. This
runner is meant to be fired often (Task Scheduler every 30 minutes and at
logon, see ``register_auto_capture.ps1``) and decides for itself whether
there is anything to do:

* **close** phase - a weekday after 16:15 New York time (the US close plus
  yfinance's ~15 minute quote delay) until midnight New York. This is the
  snapshot the studies are built on.
* **intraday** phase - a weekday 10:00-16:15 New York. If the machine is on
  during the session but may not be after the close, an intraday snapshot is
  better than a missing day. It is recorded as intraday in the day's
  ``_capture_meta.json`` and REPLACED if a close-phase run happens later the
  same session.
* otherwise (pre-open, overnight, weekends) it exits at once, before
  importing pandas or touching the network.

Dates are New York session dates, not local dates: 00:00-05:00 Irish time is
still the previous US evening, and a local-date label would file that
session's close under the wrong day. Holidays are caught by the same
SPY-daily-bar check ``capture.py`` uses.

After a capture it commits and pushes ONLY the nested data repo
(``vol_snapshots/data``). It never touches the toolkit repo's branches:
the old ``daily_capture.ps1`` rebased whatever branch was checked out, which
is wrong on a machine used for development. The surface fits and studies are
left to run on demand.

Everything goes to ``vol_snapshots/capture.log`` (pythonw has no console).

    pythonw vol_snapshots/auto_capture.py          # what the task runs
    python  vol_snapshots/auto_capture.py --dry-run # say what it would do
"""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

NY = ZoneInfo("America/New_York")
INTRADAY_START = _dt.time(10, 0)     # skip the opening half hour's wide quotes
CLOSE_READY = _dt.time(16, 15)       # 16:00 close + ~15 min delayed quotes
LOG_PATH = os.path.join(_HERE, "capture.log")
LOCK_PATH = os.path.join(_HERE, ".auto_capture.lock")
LOCK_STALE_SECONDS = 45 * 60
DATA_DIR = os.path.join(_HERE, "data")
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW


# --------------------------------------------------------------------------- #
# Pure decisions                                                               #
# --------------------------------------------------------------------------- #
def capture_phase(now_ny: _dt.datetime) -> Optional[str]:
    """"close", "intraday" or None for a New York wall-clock time."""
    if now_ny.weekday() >= 5:
        return None
    t = now_ny.time()
    if t >= CLOSE_READY:
        return "close"
    if t >= INTRADAY_START:
        return "intraday"
    return None


def needs_capture(file_exists: bool, meta_entry: Optional[dict],
                  phase: str) -> bool:
    """Capture this ticker now? Missing file -> yes. An existing intraday
    file is upgraded by a close-phase run. Anything else (a close file, or a
    legacy file with no meta) is final."""
    if not file_exists:
        return True
    prior = (meta_entry or {}).get("phase", "close")
    return prior == "intraday" and phase == "close"


def plan(tickers: Iterable[str], existing: Iterable[str],
         meta: Dict[str, dict], phase: str) -> List[str]:
    """The tickers this run should capture."""
    have = {t.upper() for t in existing}
    return [t for t in tickers
            if needs_capture(t.upper() in have, meta.get(t.upper()), phase)]


# --------------------------------------------------------------------------- #
# Side effects                                                                 #
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(f"{_dt.datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def _acquire_lock() -> bool:
    try:
        if os.path.exists(LOCK_PATH):
            age = time.time() - os.path.getmtime(LOCK_PATH)
            if age < LOCK_STALE_SECONDS:
                return False
            os.remove(LOCK_PATH)               # a crashed run left it behind
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except OSError:
        return False


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", DATA_DIR, *args], capture_output=True,
                          text=True, creationflags=_NO_WINDOW, timeout=300)


def push_data(session: str, phase: str) -> None:
    """Commit + push the nested data repo only; failures are logged, the
    files stay on disk and the next run retries."""
    if not os.path.isdir(os.path.join(DATA_DIR, ".git")):
        _log("ERROR: vol_snapshots/data is not a git repo - data kept on disk, not pushed")
        return
    _git("add", "-A")
    if _git("diff", "--cached", "--quiet").returncode != 0:
        host = os.environ.get("COMPUTERNAME", "unknown")
        c = _git("commit", "-q", "-m", f"capture {session} ({phase}, {host})")
        if c.returncode != 0:
            _log(f"ERROR: data commit failed: {c.stderr.strip()}")
            return
    push_pending()


def unpushed_commits() -> int:
    """Local data commits not on origin/main (no network: last-fetched ref)."""
    r = _git("rev-list", "--count", "origin/main..HEAD")
    try:
        return int(r.stdout.strip()) if r.returncode == 0 else 0
    except ValueError:
        return 0


def push_pending() -> None:
    """Rebase onto origin and push any local data commits (retries a push
    that failed on an earlier run)."""
    if unpushed_commits() == 0:
        return
    pull = _git("pull", "-q", "--rebase", "origin", "main")
    if pull.returncode != 0:
        _git("rebase", "--abort")
        _log(f"WARNING: data pull --rebase failed ({pull.stderr.strip()}) - "
             "committed locally, push skipped")
        return
    p = _git("push", "-q", "origin", "HEAD:main")
    _log("data: pushed" if p.returncode == 0
         else f"WARNING: data push failed ({p.stderr.strip()}) - will retry next run")


def run(now_ny: Optional[_dt.datetime] = None, dry_run: bool = False) -> int:
    now_ny = now_ny or _dt.datetime.now(NY)
    phase = capture_phase(now_ny)
    if phase is None:
        return 0                               # the common, silent case
    session = now_ny.date().isoformat()

    from vol_snapshots import capture as C   # pandas/yfinance only from here

    day_dir = os.path.join(C.DATA_ROOT, session)
    existing = ([f[:-len(".csv.gz")] for f in os.listdir(day_dir)
                 if f.endswith(".csv.gz")] if os.path.isdir(day_dir) else [])
    meta = C.read_capture_meta(C.DATA_ROOT, session)
    todo = plan(C.DEFAULT_TICKERS, existing, meta, phase)
    if not todo:
        # This phase is already done; retry a push an earlier run could not
        # make (a local check first, so the usual case costs no network).
        if not dry_run and os.path.isdir(os.path.join(DATA_DIR, ".git")) \
                and unpushed_commits() > 0:
            _log(f"data: {unpushed_commits()} unpushed commit(s) - retrying push")
            push_pending()
        return 0

    last = C.last_session_date()
    if last is not None and last != now_ny.date():
        _log(f"{session} is not a trading session (last bar {last}) - skipped")
        return 0
    if dry_run:
        _log(f"[dry-run] {phase} capture for {session}: {', '.join(todo)}")
        return 0

    _log(f"=== {phase} capture for {session}: {len(todo)} ticker(s) ===")
    rf = C._fetch_riskfree()
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    n_ok = 0
    for t in todo:
        try:
            if C.capture_ticker(t, session, rf, replace=True):
                C.write_capture_meta(C.DATA_ROOT, session, t, phase, stamp)
                n_ok += 1
        except Exception as e:                 # one dead ticker can't stop the batch
            _log(f"  {t}: capture failed ({type(e).__name__}: {e})")
    _log(f"captured {n_ok}/{len(todo)}")
    if n_ok:
        push_data(session, phase)
    return 0


def main(argv: Optional[list] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    dry = "--dry-run" in args
    if sys.stdout is None or os.path.basename(sys.executable).lower() == "pythonw.exe":
        log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
    if capture_phase(_dt.datetime.now(NY)) is None:
        return 0
    if not _acquire_lock():
        return 0
    try:
        return run(dry_run=dry)
    except Exception as e:
        _log(f"ERROR: auto_capture crashed ({type(e).__name__}: {e})")
        return 1
    finally:
        try:
            os.remove(LOCK_PATH)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
