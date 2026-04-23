Param(
    [string]$Model = "musetalk",
    [string]$AvatarId = "half-avatar-bsn6",
    [string]$Tts = "elevenlabs",
    [string]$Transport = "webrtc",
    [int]$ListenPort = 18011,
    [int]$BatchSize = 8,
    [int]$LeftContext = 6,
    [int]$MiddleContext = 8,
    [int]$RightContext = 6,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

function Test-PortFree {
    Param(
        [int]$Port
    )
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $Port)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

function Resolve-ListenPort {
    Param(
        [int]$PreferredPort,
        [int]$SearchRange = 50
    )
    if (Test-PortFree -Port $PreferredPort) {
        return $PreferredPort
    }
    for ($candidate = $PreferredPort + 1; $candidate -le ($PreferredPort + $SearchRange); $candidate++) {
        if (Test-PortFree -Port $candidate) {
            Write-Warning "Listen port $PreferredPort dang bi chiem, tu dong chuyen sang $candidate"
            return $candidate
        }
    }
    throw "Khong tim thay cong trong tu $PreferredPort den $($PreferredPort + $SearchRange)"
}

$liveTalkingDir = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $liveTalkingDir
$pythonExe = Join-Path $liveTalkingDir "venv310\Scripts\python.exe"
$envFile = Join-Path $repoRoot ".env"

if (-not (Test-Path $pythonExe)) {
    throw "Khong tim thay Python env: $pythonExe"
}

# Load .env so ELEVEN_* and related backend vars are available.
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $idx = $line.IndexOf("=")
        if ($idx -lt 1) { return }
        $name = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        if ($name) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Bridge LiveTalking -> Noble RAG (sales chat), prefer explicit env overrides.
$candidateBases = @()
$configuredBase = ([string]$env:NOBLE_RAG_BASE_URL).Trim().TrimEnd("/")
if ($configuredBase) { $candidateBases += $configuredBase }
$portFromEnv = ([string]$env:RAG_SERVICE_PORT).Trim()
if ($portFromEnv) { $candidateBases += "http://127.0.0.1:$portFromEnv" }
$candidateBases += @(
    "http://127.0.0.1:18081",
    "http://127.0.0.1:8010"
)
$candidateBases = $candidateBases | Select-Object -Unique

$ragBase = $candidateBases[0]
foreach ($base in $candidateBases) {
    try {
        $health = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 2
        if ($health) {
            $ragBase = $base
            break
        }
    } catch {}
}
if (-not $env:NOBLE_RAG_CHAT_URL) {
    $env:NOBLE_RAG_CHAT_URL = "$ragBase/sales/chat"
}
if (-not $env:NOBLE_RAG_CHAT_STREAM_URL) {
    $env:NOBLE_RAG_CHAT_STREAM_URL = "$ragBase/sales/chat/stream"
}
if (-not $env:NOBLE_RAG_SPEAK_THINKING_ACK) {
    $env:NOBLE_RAG_SPEAK_THINKING_ACK = "true"
}
if (-not $env:NOBLE_RAG_STREAM_ONLY) {
    $env:NOBLE_RAG_STREAM_ONLY = "true"
}
if (-not $env:NOBLE_RAG_STREAM_PARTIAL_CHARS) {
    $env:NOBLE_RAG_STREAM_PARTIAL_CHARS = "24"
}
$env:NOBLE_RAG_TIMEOUT_SEC = "60"
$env:PYTHONUTF8 = "1"

try {
    $healthBase = ($env:NOBLE_RAG_CHAT_URL -replace "/sales/chat.*$", "")
    $health = Invoke-RestMethod -Uri "$healthBase/health" -TimeoutSec 3
    Write-Host "RAG health:" ($health.status) "- $healthBase/health"
} catch {
    Write-Warning "Khong goi duoc RAG health. LiveTalking van chay, nhung chat co the fail."
}

Push-Location $liveTalkingDir
try {
    $resolvedListenPort = Resolve-ListenPort -PreferredPort $ListenPort
    $argsList = @(
        "app.py",
        "--model", $Model,
        "--avatar_id", $AvatarId,
        "-l", "$LeftContext",
        "-m", "$MiddleContext",
        "-r", "$RightContext",
        "--batch_size", "$BatchSize",
        "--tts", $Tts,
        "--transport", $Transport,
        "--listenport", "$resolvedListenPort"
    )
    if ($ExtraArgs) {
        $argsList += $ExtraArgs
    }

    Write-Host "LiveTalking dashboard: http://127.0.0.1:$resolvedListenPort/dashboard.html"
    Write-Host "LiveTalking WebRTC API: http://127.0.0.1:$resolvedListenPort/webrtcapi.html"
    Write-Host "Running:" $pythonExe ($argsList -join " ")
    & $pythonExe @argsList
} finally {
    Pop-Location
}
