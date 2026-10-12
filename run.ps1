# run.ps1 — start NoticeIQ backend + UI on Windows (PowerShell)
#
# Usage (from project root with virtual-env already created):
#   .\run.ps1
#
# Prerequisites:
#   1. Python 3.11+ installed and on PATH
#   2. pip install -r requirements.txt  (ideally inside a .venv)
#   3. .env file present (copy .env.example and fill in LLM_API_KEY)
#
# The script starts the FastAPI backend and the Streamlit UI in two
# separate background jobs and waits. Press Ctrl+C to stop both.

param(
    [string]$BackendHost = "127.0.0.1",
    [int]$BackendPort    = 8000,
    [int]$UIPort         = 8501
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Resolve python / uvicorn / streamlit from .venv if it exists ───────────────
$venvDir = Join-Path $PSScriptRoot ".venv"
if (Test-Path $venvDir) {
    $PythonExe   = Join-Path $venvDir "Scripts\python.exe"
    $UvicornExe  = Join-Path $venvDir "Scripts\uvicorn.exe"
    $StreamlitExe = Join-Path $venvDir "Scripts\streamlit.exe"
} else {
    $PythonExe   = "python"
    $UvicornExe  = "uvicorn"
    $StreamlitExe = "streamlit"
}

# ── Verify .env exists ─────────────────────────────────────────────────────────
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Warning ".env file not found. Copying from .env.example …"
    $exampleFile = Join-Path $PSScriptRoot ".env.example"
    if (Test-Path $exampleFile) {
        Copy-Item $exampleFile $envFile
        Write-Warning "Please edit .env and set LLM_API_KEY before using the Chat page."
    } else {
        Write-Error ".env.example not found either. Cannot start."
        exit 1
    }
}

# ── Ensure data directories exist ─────────────────────────────────────────────
$dirs = @("data\inbox", "data\processed", "data\chroma", "eval\results")
foreach ($d in $dirs) {
    $full = Join-Path $PSScriptRoot $d
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
        Write-Host "Created $full"
    }
}

# ── Launch FastAPI backend ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "==> Starting FastAPI backend on http://${BackendHost}:${BackendPort} …" -ForegroundColor Cyan
$backendJob = Start-Job -ScriptBlock {
    param($exe, $host_, $port, $dir)
    Set-Location $dir
    & $exe app.main:app --host $host_ --port $port --reload
} -ArgumentList $UvicornExe, $BackendHost, $BackendPort, $PSScriptRoot

# Give the backend a moment to bind
Start-Sleep -Seconds 3

# ── Launch Streamlit UI ────────────────────────────────────────────────────────
Write-Host "==> Starting Streamlit UI on http://localhost:${UIPort} …" -ForegroundColor Cyan
$env:BACKEND_URL = "http://${BackendHost}:${BackendPort}"
$uiJob = Start-Job -ScriptBlock {
    param($exe, $port, $dir, $backendUrl)
    Set-Location $dir
    $env:BACKEND_URL = $backendUrl
    & $exe run ui/streamlit_app.py --server.port $port --server.headless true
} -ArgumentList $StreamlitExe, $UIPort, $PSScriptRoot, "http://${BackendHost}:${BackendPort}"

Write-Host ""
Write-Host "NoticeIQ is running." -ForegroundColor Green
Write-Host "  Backend API : http://${BackendHost}:${BackendPort}/docs"
Write-Host "  UI          : http://localhost:${UIPort}"
Write-Host ""
Write-Host "Press Ctrl+C to stop both services."

try {
    # Stream job output to console while both jobs are alive
    while ($true) {
        Receive-Job -Job $backendJob  -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "[backend] $_" }
        Receive-Job -Job $uiJob       -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "[ui]      $_" }
        if ($backendJob.State -eq "Failed" -or $uiJob.State -eq "Failed") {
            Write-Error "One of the processes exited unexpectedly."
            break
        }
        Start-Sleep -Milliseconds 500
    }
} finally {
    Write-Host "`nStopping services …" -ForegroundColor Yellow
    Stop-Job  -Job $backendJob, $uiJob -ErrorAction SilentlyContinue
    Remove-Job -Job $backendJob, $uiJob -Force -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Yellow
}
