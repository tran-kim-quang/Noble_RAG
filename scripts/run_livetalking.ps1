Param(
    [string]$Model = "musetalk",
    [string]$AvatarId = "half-avatar-bsn6",
    [string]$Tts = "edgetts",
    [string]$Transport = "webrtc",
    [int]$ListenPort = 18011,
    [int]$BatchSize = 16,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$liveTalkingDir = Join-Path $repoRoot "livetalking"
$pythonExe = Join-Path $liveTalkingDir "venv310\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Khong tim thay Python env: $pythonExe"
}

# Bridge LiveTalking -> Noble RAG (sales chat)
$env:NOBLE_RAG_CHAT_URL = "http://127.0.0.1:18081/sales/chat"
$env:NOBLE_RAG_CHAT_STREAM_URL = "http://127.0.0.1:18081/sales/chat/stream"
$env:NOBLE_RAG_TIMEOUT_SEC = "60"
$env:PYTHONUTF8 = "1"

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:18081/health" -TimeoutSec 3
    Write-Host "RAG health:" ($health.status) "- http://127.0.0.1:18081/health"
} catch {
    Write-Warning "Khong goi duoc RAG health tren 18081. LiveTalking van chay, nhung chat co the fail."
}

Push-Location $liveTalkingDir
try {
    $argsList = @(
        "app.py",
        "--model", $Model,
        "--avatar_id", $AvatarId,
        "--batch_size", "$BatchSize",
        "--tts", $Tts,
        "--transport", $Transport,
        "--listenport", "$ListenPort"
    )
    if ($ExtraArgs) {
        $argsList += $ExtraArgs
    }

    Write-Host "Running:" $pythonExe ($argsList -join " ")
    & $pythonExe @argsList
} finally {
    Pop-Location
}
