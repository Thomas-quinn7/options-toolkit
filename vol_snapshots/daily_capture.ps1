# Daily option-chain snapshot: capture, fit the surface, commit the results.
#
# Run by Windows Task Scheduler (see vol_snapshots/README.md). Appends all
# output to vol_snapshots/capture.log so silent failures are visible.
#
# Two git homes, deliberately:
#   * RAW chains  -> the nested data repo at vol_snapshots/data
#                    (github.com/Thomas-quinn7/options-toolkit-data, private) -
#                    committed AND pushed daily so the dataset is durable
#                    off-machine without bloating the public toolkit repo.
#   * DERIVED     -> the toolkit repo: surface_history.csv + the history chart
#                    (small, and part of the public portfolio). Committed, not
#                    pushed - it rides along with normal pushes.
#
# If the nested data repo is missing (first run after the split) it is
# bootstrapped automatically: init, fetch the remote history, restore any
# tracked files missing locally, adopt local-only days. On bootstrap/push
# failure the error is logged loudly and the raw files simply stay on disk
# untracked - nothing is lost, the next run retries.

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$log = Join-Path $PSScriptRoot "capture.log"
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$dataDir = Join-Path $PSScriptRoot "data"
$dataRemote = "https://github.com/Thomas-quinn7/options-toolkit-data.git"

"=== $(Get-Date -Format s) capture starting ===" | Out-File $log -Append -Encoding utf8

# 1) Capture today's chains (idempotent per day).
& $py (Join-Path $PSScriptRoot "capture.py") 2>&1 | Out-File $log -Append -Encoding utf8

# 2) Ensure the nested data repo exists; bootstrap it if not.
if (-not (Test-Path (Join-Path $dataDir ".git"))) {
    "bootstrapping nested data repo in vol_snapshots/data ..." | Out-File $log -Append -Encoding utf8
    New-Item -ItemType Directory -Force $dataDir | Out-Null
    git -C $dataDir init -q -b main 2>&1 | Out-File $log -Append -Encoding utf8
    git -C $dataDir remote add origin $dataRemote 2>&1 | Out-File $log -Append -Encoding utf8
    git -C $dataDir fetch -q origin main 2>&1 | Out-File $log -Append -Encoding utf8
    if ($LASTEXITCODE -eq 0) {
        # Adopt the remote history without touching local files, then restore
        # any tracked files the local tree is missing (earlier days).
        git -C $dataDir reset -q --mixed FETCH_HEAD 2>&1 | Out-File $log -Append -Encoding utf8
        $deleted = git -C $dataDir ls-files --deleted
        foreach ($f in $deleted) {
            git -C $dataDir checkout -- $f 2>&1 | Out-File $log -Append -Encoding utf8
        }
    } else {
        "WARNING: could not fetch $dataRemote (auth/network?) - starting fresh history" |
            Out-File $log -Append -Encoding utf8
    }
}

# 3) Commit + push the raw data in the nested repo.
git -C $dataDir add -A 2>&1 | Out-File $log -Append -Encoding utf8
git -C $dataDir diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git -C $dataDir commit -q -m "daily capture $(Get-Date -Format yyyy-MM-dd)" 2>&1 |
        Out-File $log -Append -Encoding utf8
    git -C $dataDir push -q -u origin main 2>&1 | Out-File $log -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        "ERROR: data repo push FAILED - data is committed locally only; will retry tomorrow" |
            Out-File $log -Append -Encoding utf8
    }
} else {
    "data: nothing new to commit" | Out-File $log -Append -Encoding utf8
}

# 4) Fit the day's surfaces and update the history + chart (toolkit repo).
& $py (Join-Path $PSScriptRoot "fit_history.py") 2>&1 | Out-File $log -Append -Encoding utf8

# 5) Commit the derived artifacts to the toolkit repo (stage ONLY these paths
#    so unrelated work-in-progress is never swept into an automated commit).
git add vol_snapshots/surface_history.csv charts/vol_surface/surface_history.png 2>&1 |
    Out-File $log -Append -Encoding utf8
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -q -m "vol_snapshots: daily surface fit $(Get-Date -Format yyyy-MM-dd)" 2>&1 |
        Out-File $log -Append -Encoding utf8
} else {
    "derived: nothing new to commit" | Out-File $log -Append -Encoding utf8
}

"=== $(Get-Date -Format s) done ===" | Out-File $log -Append -Encoding utf8
