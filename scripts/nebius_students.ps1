<#
Nebius Token Factory students on Windows (no WSL needed): sampling only.

  $env:NEBIUS_API_KEY = "<key>"          # or set it once in the user environment
  powershell -ExecutionPolicy Bypass -File scripts\nebius_students.ps1

Steps: 0 preflight (no key sent), 1 list models (1 request), 2 smoke (1 request),
3 sample N students at each pivot from the pre-built prompts (N x pivots requests, 30 by default).
Output: out\tokenfactory\students.jsonl plus student_stats.json, requests.jsonl and cache\.
Labelling needs Linux: send students.jsonl back to the cloud lane, or in WSL run
  python -m pivots.audit.run specimens/*/ --students out/tokenfactory/students.jsonl --pivots <same> --progress --out out/tokenfactory/audit
Any 401/402/403, billing or quota error stops at once and is never retried. The key is never printed or written.
#>
param(
  [string]$Model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
  [int]$N = 5,
  [string]$Pivots = "access-log-summary:4,backup-cron-repair:3,billing-invoice-bugfix:3,ledger-git-revert:3,sensor-site-report:4,statcli-make-fix:5",
  [string]$Out = "out\tokenfactory",
  [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $env:NEBIUS_API_KEY) { throw "Set NEBIUS_API_KEY first (the key is read from the environment only)." }
$base = if ($env:TOKENFACTORY_BASE_URL) { $env:TOKENFACTORY_BASE_URL } else { "https://api.tokenfactory.nebius.com/v1" }
$env:TOKENFACTORY_BASE_URL = $base
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$cache = Join-Path $Out "cache"
$nPiv = ($Pivots.Split(",") | Where-Object { $_ }).Count
$audit = $nPiv * $N
$cap = $audit + [math]::Floor($audit / 3) + 4
Write-Host "== model $Model; $N x $nPiv pivots = $audit sample requests (+2 for list and smoke); cap $cap"

Write-Host "== 0. preflight (no key sent)"
try { $r = Invoke-WebRequest -UseBasicParsing -Method Get -Uri "$base/models" -TimeoutSec 20; $code = $r.StatusCode }
catch { if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode } else { throw "Token Factory not reachable: $($_.Exception.Message)" } }
Write-Host "   reachable (HTTP $code without a key)"

function Tf { & $Python -m pivots.students.tokenfactory --model $Model --base-url $base --cache-dir $cache @args; if ($LASTEXITCODE -ne 0) { throw "step failed (exit $LASTEXITCODE); see the message above" } }
Write-Host "== 1. models"
Tf --list-models --max-requests 2 --out (Join-Path $Out "models.json")
Write-Host "== 2. smoke"
Tf --smoke --max-requests 2 --out (Join-Path $Out "smoke.json")
Write-Host "== 3. sample"
& $Python -m pivots.students.sample_prompts --model $Model --base-url $base --cache-dir $cache `
  --pivots $Pivots --n $N --max-requests $cap --out (Join-Path $Out "students.jsonl")
if ($LASTEXITCODE -ne 0) { throw "sampling stopped (exit $LASTEXITCODE); samples so far are in $Out\students.jsonl" }
Write-Host "Done. Send $Out\students.jsonl to the cloud lane for labelling (it holds only model outputs, no key)."
