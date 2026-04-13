Param(
    [string]$Model = "wav2lip",
    [string]$AvatarId = "half-wav2lip",
    [string]$Tts = "elevenlabs",
    [string]$Transport = "webrtc",
    [int]$ListenPort = 18011,
    [int]$BatchSize = 16,
    [int]$RagPort = 18081,
    [int]$SttPort = 18001,
    [int]$HealthTimeoutSec = 120,
    [switch]$SkipDocker,
    [switch]$NoBuildRag,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

function Wait-RagHealth {
    Param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSec = 120
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-RestMethod -Uri $Url -TimeoutSec 3
            if ($resp.status -eq "healthy") {
                return $true
            }
        } catch {
            # retry until timeout
        }
        Start-Sleep -Seconds 2
    }
    return $false
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$liveTalkingRunner = Join-Path $repoRoot "livetalking\scripts\run_livetalking.ps1"

if (-not (Test-Path $liveTalkingRunner)) {
    throw "Missing script: $liveTalkingRunner"
}

$ragChatUrl = "http://127.0.0.1:$RagPort/sales/chat"
$ragChatStreamUrl = "http://127.0.0.1:$RagPort/sales/chat/stream"
$ragHealthUrl = "http://127.0.0.1:$RagPort/health"

if (-not $SkipDocker) {
    Push-Location $repoRoot
    try {
        Write-Host "[1/3] Starting base services..."
        $env:WHISPER_SERVICE_PORT = "$SttPort"
        docker compose up -d postgres qdrant redis ollama whisper-service

        Write-Host "[2/3] Starting rag-service on port $RagPort..."
        $env:RAG_SERVICE_PORT = "$RagPort"
        if ($NoBuildRag) {
            docker compose up -d --no-deps rag-service
        } else {
            docker compose up -d --build --no-deps rag-service
        }
    } finally {
        Pop-Location
    }
}

Write-Host "[3/3] Waiting for RAG health: $ragHealthUrl"
if (-not (Wait-RagHealth -Url $ragHealthUrl -TimeoutSec $HealthTimeoutSec)) {
    throw "RAG health check timed out: $ragHealthUrl"
}

# Bridge LiveTalking -> RAG stream endpoint.
$env:NOBLE_RAG_CHAT_URL = $ragChatUrl
$env:NOBLE_RAG_CHAT_STREAM_URL = $ragChatStreamUrl
$env:NOBLE_RAG_STREAM_ONLY = "true"
$env:NOBLE_RAG_SPEAK_THINKING_ACK = "false"
$env:NOBLE_RAG_TIMEOUT_SEC = "60"

# Force STT to Deepgram backend through local whisper-service proxy.
$env:STT_PROVIDER = "deepgram"
$env:STT_REQUIRE_DEEPGRAM = "true"
$env:STT_API_URL = "http://127.0.0.1:$SttPort/transcribe"

Write-Host "RAG stream endpoint: $env:NOBLE_RAG_CHAT_STREAM_URL"
Write-Host "Launching LiveTalking: model=$Model avatar=$AvatarId tts=$Tts transport=$Transport port=$ListenPort"
Write-Host "WebRTC UI: http://127.0.0.1:$ListenPort/dashboard.html"

$runnerArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $liveTalkingRunner,
    "-Model", $Model,
    "-AvatarId", $AvatarId,
    "-Tts", $Tts,
    "-Transport", $Transport,
    "-ListenPort", "$ListenPort",
    "-BatchSize", "$BatchSize"
)
if ($ExtraArgs) {
    $runnerArgs += $ExtraArgs
}

powershell @runnerArgs
