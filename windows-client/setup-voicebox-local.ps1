# Voicebox fork (gemini-stt) - local backend for Windows.
# Run ONCE in PowerShell:  .\setup-voicebox-local.ps1
# - installs Python 3.12 via uv, clones the fork, installs deps
# - registers a logon task that runs the backend on 127.0.0.1:17493
# - the official Voicebox app auto-adopts this server at next launch

param(
  [Parameter(Mandatory=$true)][string]$GeminiKey,
  [string]$GeminiKey2 = "",
  [string]$NebiusKey = "",
  [string]$InstallDir = "$env:USERPROFILE\voicebox-fork"
)

$ErrorActionPreference = "Stop"
Write-Host "== Voicebox local backend setup =="

# 1) python 3.12 via uv (app wheels need 3.11/3.12; avoid python 3.14)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
uv python install 3.12

# 2) fetch the fork
if (-not (Test-Path $InstallDir)) {
  Invoke-WebRequest "https://github.com/surya17495/voicebox/archive/refs/heads/gemini-stt.zip" -OutFile "$env:TEMP\vb.zip"
  Expand-Archive "$env:TEMP\vb.zip" -DestinationPath $env:TEMP\vb-extract -Force
  Move-Item "$env:TEMP\vb-extract\voicebox-gemini-stt" $InstallDir
}

# 3) venv + deps
Set-Location $InstallDir
if (-not (Test-Path "$InstallDir\.venv")) {
  uv venv "$InstallDir\.venv" --python 3.12
  uv pip install --python "$InstallDir\.venv\Scripts\python.exe" -r windows-client\requirements-fork.txt
}

# 4) env file (local only, never commit)
@"
GEMINI_API_KEY=$GeminiKey
GEMINI_API_KEY_2=$GeminiKey2
NEBIUS_API_KEY=$NebiusKey
NEBIUS_REWRITE_MODEL=nvidia/Nemotron-3_5-Lightning
NEBIUS_REWRITE_MIN_SEC=12
"@ | Out-File "$InstallDir\.env.local" -Encoding utf8NoBOM

# 5) runner + logon task (kills stale copies of itself first)
$run = @"
`$ErrorActionPreference = 'Continue'
Set-Location '$InstallDir'
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { `$_.CommandLine -like '*backend.main:app*' -and `$_.ProcessId -ne `$PID } |
  ForEach-Object { Stop-Process -Id `$_.ProcessId -Force }
`$env:GEMINI_API_KEY  = (Get-Content '$InstallDir\.env.local' | Select-String '^GEMINI_API_KEY=' ) -replace 'GEMINI_API_KEY=',''
`$env:GEMINI_API_KEY_2 = (Get-Content '$InstallDir\.env.local' | Select-String '^GEMINI_API_KEY_2=') -replace 'GEMINI_API_KEY_2=',''
`$env:NEBIUS_API_KEY   = (Get-Content '$InstallDir\.env.local' | Select-String '^NEBIUS_API_KEY='  ) -replace 'NEBIUS_API_KEY=',''
`$env:NEBIUS_REWRITE_MODEL     = 'nvidia/Nemotron-3_5-Lightning'
`$env:NEBIUS_REWRITE_MIN_SEC   = '12'
& '.venv\Scripts\python.exe' -m uvicorn backend.main:app --host 127.0.0.1 --port 17493
"@
$run | Out-File "$InstallDir\run-backend.ps1" -Encoding utf8NoBOM
$act  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$InstallDir\run-backend.ps1`""
$trig = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$set  = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 365) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "VoiceboxLocalBackend" -Action $act -Trigger $trig -Settings $set -RunLevel Highest -Force | Out-Null
Start-ScheduledTask -TaskName "VoiceboxLocalBackend"
Start-Sleep -Seconds 12

# 6) verify
try {
  $h = Invoke-RestMethod "http://127.0.0.1:17493/health" -TimeoutSec 8
  Write-Host "backend: $($h.status), model_loaded=$($h.model_loaded), size=$($h.model_size)"
} catch { Write-Host "HEALTH CHECK FAILED - check task: Start-ScheduledTask VoiceboxLocalBackend" }

Write-Host "Done. Launch the Voicebox app - it adopts this server on localhost automatically."
