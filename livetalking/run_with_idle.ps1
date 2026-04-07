$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

if (-not $env:PYTHONUTF8) { $env:PYTHONUTF8 = "1" }
if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = "utf-8" }

$pythonCandidates = @(
  (Join-Path $projectRoot "venv\Scripts\python.exe"),
  (Join-Path $projectRoot ".venv\Scripts\python.exe"),
  "python"
)
$pythonExe = $pythonCandidates | Where-Object { $_ -eq "python" -or (Test-Path $_) } | Select-Object -First 1
if (-not $pythonExe) {
  Write-Host "Cannot find Python executable. Expected venv\.venv python or python in PATH."
  exit 1
}

$envFile = Join-Path $projectRoot ".env"
if (Test-Path $envFile) {
  Get-Content $envFile | ForEach-Object {
    if ($_ -match "^\s*#") { return }
    if ($_ -match "^\s*$") { return }
    $pair = $_ -split "=", 2
    if ($pair.Length -eq 2) {
      $k = $pair[0].Trim()
      $v = $pair[1].Trim()
      if ($k -and $v -and -not (Get-Item "Env:$k" -ErrorAction SilentlyContinue)) {
        [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
      }
    }
  }
}

$avatarId = if ($env:LIVETALKING_AVATAR_ID -and -not [string]::IsNullOrWhiteSpace($env:LIVETALKING_AVATAR_ID)) {
  $env:LIVETALKING_AVATAR_ID
} else {
  "half"
}
if ($avatarId -in @("half-avatar", "half_avatar")) {
  $avatarId = "half"
}
$requestedTts = if ($env:LIVETALKING_TTS -and -not [string]::IsNullOrWhiteSpace($env:LIVETALKING_TTS)) {
  $env:LIVETALKING_TTS.Trim().ToLower()
} else {
  "elevenlabs"
}
if ($requestedTts -in @("edge_tts", "edge")) {
  $requestedTts = "edgetts"
}
$enableTransition = $false
$transitionDuration = 0.06
$transport = if ($env:LIVETALKING_TRANSPORT) { $env:LIVETALKING_TRANSPORT } else { "webrtc" }
$pushUrl = if ($env:LIVETALKING_PUSH_URL) { $env:LIVETALKING_PUSH_URL } else { "http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream" }
$listenPort = if ($env:LIVETALKING_PORT) { $env:LIVETALKING_PORT } else { "18010" }

function Stop-PortListener {
  param(
    [Parameter(Mandatory = $true)][int]$Port
  )

  $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $listener) {
    Write-Host "Port $Port is already free"
    return
  }

  $listenerPid = [int]$listener.OwningProcess
  Write-Host "Port $Port is occupied by pid=$listenerPid. Stopping it before launch..."
  try {
    Stop-Process -Id $listenerPid -Force -ErrorAction Stop
  } catch {
    Write-Host "Failed to stop pid=${listenerPid}: $($_.Exception.Message)"
  }

  # If this is the child python, stop the direct parent too (common double-python tree).
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$listenerPid" -ErrorAction SilentlyContinue
  if ($proc -and $proc.ParentProcessId) {
    $parentPid = [int]$proc.ParentProcessId
    if ($parentPid -gt 0) {
      try {
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$parentPid" -ErrorAction SilentlyContinue
        if ($parent -and ($parent.Name -ieq "python.exe")) {
          Stop-Process -Id $parentPid -Force -ErrorAction SilentlyContinue
          Write-Host "Also stopped parent python pid=$parentPid"
        }
      } catch {
      }
    }
  }

  Start-Sleep -Milliseconds 300
  $check = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($check) {
    throw "Port $Port is still occupied by pid=$($check.OwningProcess)."
  }
  Write-Host "Port $Port is now free"
}

# Noble RAG runtime configuration used by livetalking/rag_chat_client.py
if (-not $env:NOBLE_RAG_API_URL) { $env:NOBLE_RAG_API_URL = "http://127.0.0.1:8010" }
if (-not $env:NOBLE_VISION_API_URL) { $env:NOBLE_VISION_API_URL = "http://127.0.0.1:8020" }
if (-not $env:NOBLE_WHISPER_API_URL) { $env:NOBLE_WHISPER_API_URL = "http://127.0.0.1:8001" }
if (-not $env:NOBLE_VISION_SOURCE) { $env:NOBLE_VISION_SOURCE = "browser" }
if (-not $env:RAG_CHAT_MODE) { $env:RAG_CHAT_MODE = "stream" }
if (-not $env:RAG_CHAT_ENDPOINT) { $env:RAG_CHAT_ENDPOINT = "/query/stream" }
if (-not $env:RAG_STREAM_METHOD) { $env:RAG_STREAM_METHOD = "POST" }
if (-not $env:LIGHTRAG_URL) { $env:LIGHTRAG_URL = $env:NOBLE_RAG_API_URL }
if (-not $env:RAG_STREAM_ENDPOINT) { $env:RAG_STREAM_ENDPOINT = "/query/stream" }
if (-not $env:NOBLE_USE_CAMERA_CHAT_ENDPOINT) { $env:NOBLE_USE_CAMERA_CHAT_ENDPOINT = "true" }
if (-not $env:NOBLE_FOLLOWUP_NO_CAMERA) { $env:NOBLE_FOLLOWUP_NO_CAMERA = "true" }
if (-not $env:RAG_CAMERA_CHAT_ENDPOINT) { $env:RAG_CAMERA_CHAT_ENDPOINT = "/api/v1/sales/chat-with-camera" }
if (-not $env:NOBLE_CAMERA_SESSION_BY_CUSTOMER_ID) { $env:NOBLE_CAMERA_SESSION_BY_CUSTOMER_ID = "true" }
if (-not $env:NOBLE_CAMERA_CHAT_STRICT) { $env:NOBLE_CAMERA_CHAT_STRICT = "true" }
if (-not $env:NOBLE_CAMERA_BACKENDS) { $env:NOBLE_CAMERA_BACKENDS = "CAP_MSMF,CAP_DSHOW,CAP_ANY" }
if (-not $env:NOBLE_CAMERA_INDEXES) { $env:NOBLE_CAMERA_INDEXES = "0,1,2" }
if (-not $env:NOBLE_WHISPER_EOF_RETRY) { $env:NOBLE_WHISPER_EOF_RETRY = "1" }
if (-not $env:NOBLE_PRESENCE_CAMERA) { $env:NOBLE_PRESENCE_CAMERA = "cam01" }
if (-not $env:NOBLE_PRESENCE_STALE_SEC) { $env:NOBLE_PRESENCE_STALE_SEC = "4.0" }
# Optional: VISION_STARTUP_SMOKE=1 — chỉ hợp lệ khi NOBLE_VISION_SOURCE=camera hoặc VISION_STARTUP_SMOKE_FORCE=1 (OpenCV có webcam)

Write-Host "Project root: $projectRoot"
Write-Host "Python: $pythonExe"
Write-Host "Avatar: $avatarId"
Write-Host "Hold video: disabled"
Write-Host "TTS requested: $requestedTts"
Write-Host "Transport: $transport"
Write-Host "Listen port: $listenPort"
Write-Host "Whisper API: $env:NOBLE_WHISPER_API_URL"
if ($transport -eq "rtcpush") {
  Write-Host "RTCPush URL: $pushUrl"
}

$vaeConfigCandidates = @(
  (Join-Path $projectRoot "models\sd-vae-ft-mse\config.json"),
  (Join-Path $projectRoot "models\sd-vae\config.json")
)
$vaeConfigFound = $vaeConfigCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

$requiredModelPaths = @(
  (Join-Path $projectRoot "models\musetalkV15\unet.pth"),
  (Join-Path $projectRoot "models\musetalkV15\musetalk.json"),
  (Join-Path $projectRoot "models\whisper\config.json")
)

$missingModelPaths = @()
if (-not $vaeConfigFound) {
  $missingModelPaths += $vaeConfigCandidates[0]
}
$missingModelPaths += @($requiredModelPaths | Where-Object { -not (Test-Path $_) })
if ($missingModelPaths.Count -gt 0) {
  Write-Host "Missing required MuseTalk model files:"
  $missingModelPaths | ForEach-Object { Write-Host " - $_" }
  Write-Host "Place the downloaded MuseTalk model bundle under $projectRoot\models and try again."
  exit 1
}

try {
  Stop-PortListener -Port ([int]$listenPort)
} catch {
  Write-Host "Cannot continue: $($_.Exception.Message)"
  exit 1
}

$arguments = @(
  "app.py",
  "--avatar_id", $avatarId,
  "--tts", $requestedTts,
  "--transport", $transport,
  "--listenport", "$listenPort",
  "--transition_duration", "$transitionDuration"
)

if ($transport -eq "rtcpush") {
  $arguments += @("--push_url", $pushUrl)
}

if ($enableTransition) {
  $arguments += "--enable_transition"
}

& $pythonExe @arguments
