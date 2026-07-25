# Daily option-chain snapshot: capture, then commit the new data.
#
# Run by Windows Task Scheduler (see vol_snapshots/README.md). Appends its
# output to vol_snapshots/capture.log so silent failures are visible. The
# commit stages ONLY vol_snapshots/data, so unrelated work-in-progress in the
# repo is never swept into an automated commit; if the capture produced
# nothing new (weekend, double-fire) no commit is made.

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$log = Join-Path $PSScriptRoot "capture.log"
"=== $(Get-Date -Format s) capture starting ===" | Out-File $log -Append -Encoding utf8

& (Join-Path $repo ".venv\Scripts\python.exe") (Join-Path $PSScriptRoot "capture.py") 2>&1 |
    Out-File $log -Append -Encoding utf8

git add vol_snapshots/data
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "vol_snapshots: daily capture $(Get-Date -Format yyyy-MM-dd)" 2>&1 |
        Out-File $log -Append -Encoding utf8
} else {
    "nothing new to commit" | Out-File $log -Append -Encoding utf8
}
"=== $(Get-Date -Format s) done ===" | Out-File $log -Append -Encoding utf8
