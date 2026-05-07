param(
    [switch]$Restart,
    [int]$HealthTimeoutSec = 90,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Load-EnvFile {
    param([string]$Path)
    $map = @{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $map
    }
    foreach ($raw in Get-Content -LiteralPath $Path) {
        $line = ($raw -as [string]).Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }
        $idx = $line.IndexOf("=")
        if ($idx -lt 1) {
            continue
        }
        $key = $line.Substring(0, $idx).Trim()
        $val = $line.Substring($idx + 1).Trim()
        if ($val.Length -ge 2 -and (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'")))) {
            $val = $val.Substring(1, $val.Length - 2)
        }
        $map[$key] = $val
    }
    return $map
}

function Convert-ToLocalUrl {
    param([string]$Url)
    if (-not $Url) {
        return $Url
    }
    return ($Url -replace "retrieval-service", "127.0.0.1" `
                  -replace "vision-service", "127.0.0.1" `
                  -replace "qdrant", "127.0.0.1" `
                  -replace "ollama", "127.0.0.1")
}

function Get-MapValueOrDefault {
    param(
        [hashtable]$Map,
        [string]$Key,
        [string]$DefaultValue
    )
    if ($Map.ContainsKey($Key)) {
        $val = [string]$Map[$Key]
        if ($val.Trim().Length -gt 0) {
            return $val
        }
    }
    return $DefaultValue
}

function Test-HttpOk {
    param([string]$Url)
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
        return ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300)
    } catch {
        return $false
    }
}

function Wait-Health {
    param(
        [string]$Name,
        [string]$Url,
        [int]$TimeoutSec
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpOk -Url $Url) {
            Write-Host ("[OK]   {0} {1}" -f $Name, $Url)
            return $true
        }
        Start-Sleep -Milliseconds 800
    }
    Write-Host ("[FAIL] {0} {1}" -f $Name, $Url) -ForegroundColor Yellow
    return $false
}

function Get-ListenerPids {
    param([int]$Port)
    $rows = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $rows) {
        return @()
    }
    return @($rows | Select-Object -ExpandProperty OwningProcess -Unique)
}

function Stop-PortListeners {
    param([int]$Port)
    $pids = Get-ListenerPids -Port $Port
    foreach ($procId in $pids) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host ("[STOP] port {0} pid {1}" -f $Port, $procId)
        } catch {
            Write-Host ("[WARN] cannot stop pid {0} on port {1}: {2}" -f $procId, $Port, $_.Exception.Message) -ForegroundColor Yellow
        }
    }
}

function Start-UvicornService {
    param(
        [string]$Name,
        [string]$Module,
        [int]$Port,
        [string]$PythonExe,
        [string]$WorkingDir,
        [string]$StdOutLog,
        [string]$StdErrLog,
        [hashtable]$EnvVars,
        [switch]$RestartPort
    )
    if ($RestartPort) {
        Stop-PortListeners -Port $Port
        Start-Sleep -Milliseconds 500
    }

    $existing = @(Get-ListenerPids -Port $Port)
    if ($existing.Count -gt 0) {
        Write-Host ("[SKIP] {0} already listening on {1} (pid: {2})" -f $Name, $Port, ($existing -join ","))
        return $null
    }

    $envFile = Join-Path $logDir (("{0}.env.local" -f $Name))
    $lines = @()
    foreach ($k in ($EnvVars.Keys | Sort-Object)) {
        $value = [string]$EnvVars[$k]
        $lines += ("{0}={1}" -f $k, $value)
    }
    Set-Content -LiteralPath $envFile -Value $lines -Encoding UTF8

    $proc = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList @("-m", "uvicorn", $Module, "--host", "0.0.0.0", "--port", [string]$Port, "--env-file", $envFile) `
        -WorkingDirectory $WorkingDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdOutLog `
        -RedirectStandardError $StdErrLog `
        -PassThru
    Write-Host ("[UP]   {0} pid={1} port={2}" -f $Name, $proc.Id, $Port)
    return $proc
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

if (-not $PythonPath) {
    $candidates = @(
        (Join-Path $root ".venv310\Scripts\python.exe"),
        (Join-Path $root ".venv\Scripts\python.exe")
    )
    $PythonPath = ($candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1)
}
if (-not $PythonPath -or -not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python executable not found. Pass -PythonPath explicitly."
}

$envRetrieval = Load-EnvFile -Path (Join-Path $root "deploy\.env.retrieval")
$envVision = Load-EnvFile -Path (Join-Path $root "deploy\.env.vision")
$envOrch = Load-EnvFile -Path (Join-Path $root "deploy\.env.orchestrator")

# Normalize container hostnames to local endpoints when running without Docker.
$retrievalUrl = Convert-ToLocalUrl ($envOrch["RETRIEVAL_SERVICE_URL"])
$visionUrl = Convert-ToLocalUrl ($envOrch["VISION_SERVICE_URL"])
$qdrantUrl = Convert-ToLocalUrl ($envRetrieval["QDRANT_URL"])
$embeddingApiUrl = Convert-ToLocalUrl ($envRetrieval["EMBEDDING_API_URL"])
$localQdrantPath = Join-Path (Join-Path $root "data") "qdrant_local_runtime"

$visionPort = if ($envVision["VISION_PORT"]) { [int]$envVision["VISION_PORT"] } else { 8031 }
$retrievalPort = 8211
$orchestratorPort = if ($envOrch["ORCHESTRATOR_PORT"]) { [int]$envOrch["ORCHESTRATOR_PORT"] } else { 8021 }
$localRetrievalUrl = ("http://127.0.0.1:{0}" -f $retrievalPort)
$localVisionUrl = ("http://127.0.0.1:{0}" -f $visionPort)
$deciderApiUrlRaw = Get-MapValueOrDefault -Map $envOrch -Key "ORCHESTRATOR_DECIDER_API_URL" -DefaultValue "http://127.0.0.1:11434/api/generate"
$deciderApiUrl = Convert-ToLocalUrl $deciderApiUrlRaw
if (($deciderApiUrl -match "ollama\.com") -or (-not $deciderApiUrl)) {
    $deciderApiUrl = "http://127.0.0.1:11434/api/generate"
}
$synthesisApiUrlRaw = Get-MapValueOrDefault -Map $envOrch -Key "ORCHESTRATOR_SYNTHESIS_API_URL" -DefaultValue "http://127.0.0.1:11434/api/generate"
$synthesisApiUrl = Convert-ToLocalUrl $synthesisApiUrlRaw
if (($synthesisApiUrl -match "ollama\.com") -or (-not $synthesisApiUrl)) {
    $synthesisApiUrl = "http://127.0.0.1:11434/api/generate"
}

$visionEnv = @{
    "VISION_HOST" = "0.0.0.0"
    "VISION_PORT" = [string]$visionPort
    "VISION_FACE_DB" = (Join-Path $root "face_db")
    "VISION_MODEL_NAME" = (Get-MapValueOrDefault -Map $envVision -Key "VISION_MODEL_NAME" -DefaultValue "buffalo_l")
    "VISION_MATCH_THRESHOLD" = (Get-MapValueOrDefault -Map $envVision -Key "VISION_MATCH_THRESHOLD" -DefaultValue "0.45")
    "VISION_DET_SIZE" = (Get-MapValueOrDefault -Map $envVision -Key "VISION_DET_SIZE" -DefaultValue "640")
    "VISION_USE_GPU" = (Get-MapValueOrDefault -Map $envVision -Key "VISION_USE_GPU" -DefaultValue "false")
}

$retrievalEnv = @{
    "HOST" = "0.0.0.0"
    "PORT" = [string]$retrievalPort
    "QDRANT_URL" = $(if ($qdrantUrl -and $qdrantUrl -notmatch '127\.0\.0\.1:6333') { $qdrantUrl } else { ("local://{0}" -f $localQdrantPath) })
    "QDRANT_COLLECTION" = (Get-MapValueOrDefault -Map $envRetrieval -Key "QDRANT_COLLECTION" -DefaultValue "retrieval_bench_qwen3_8b_local")
    "QDRANT_TIMEOUT_SEC" = (Get-MapValueOrDefault -Map $envRetrieval -Key "QDRANT_TIMEOUT_SEC" -DefaultValue "10")
    "EMBEDDING_BACKEND" = (Get-MapValueOrDefault -Map $envRetrieval -Key "EMBEDDING_BACKEND" -DefaultValue "remote")
    "EMBEDDING_API_URL" = $(if ($embeddingApiUrl) { $embeddingApiUrl } else { "http://127.0.0.1:11434/api/embeddings" })
    "EMBEDDING_API_FORMAT" = (Get-MapValueOrDefault -Map $envRetrieval -Key "EMBEDDING_API_FORMAT" -DefaultValue "ollama")
    "EMBEDDING_API_TIMEOUT_SEC" = (Get-MapValueOrDefault -Map $envRetrieval -Key "EMBEDDING_API_TIMEOUT_SEC" -DefaultValue "240")
    "EMBEDDING_MODEL" = (Get-MapValueOrDefault -Map $envRetrieval -Key "EMBEDDING_MODEL" -DefaultValue "qwen3-embedding:8b")
    "EMBEDDING_DIM" = (Get-MapValueOrDefault -Map $envRetrieval -Key "EMBEDDING_DIM" -DefaultValue "4096")
    "HAYSTACK_TELEMETRY_ENABLED" = "false"
    "HOME" = $root
    "USERPROFILE" = $root
    "XDG_CONFIG_HOME" = (Join-Path $root ".config")
}

$orchEnv = @{
    "ORCHESTRATOR_HOST" = "0.0.0.0"
    "ORCHESTRATOR_PORT" = [string]$orchestratorPort
    "RETRIEVAL_SERVICE_URL" = $localRetrievalUrl
    "VISION_SERVICE_URL" = $(if ($visionUrl) { $visionUrl } else { $localVisionUrl })
    "RETRIEVAL_TIMEOUT_SEC" = (Get-MapValueOrDefault -Map $envOrch -Key "RETRIEVAL_TIMEOUT_SEC" -DefaultValue "15")
    "VISION_TIMEOUT_SEC" = (Get-MapValueOrDefault -Map $envOrch -Key "VISION_TIMEOUT_SEC" -DefaultValue "12")
    "ORCHESTRATOR_DECIDER_ENABLED" = (Get-MapValueOrDefault -Map $envOrch -Key "ORCHESTRATOR_DECIDER_ENABLED" -DefaultValue "true")
    "ORCHESTRATOR_DECIDER_API_FORMAT" = (Get-MapValueOrDefault -Map $envOrch -Key "ORCHESTRATOR_DECIDER_API_FORMAT" -DefaultValue "ollama")
    "ORCHESTRATOR_DECIDER_API_URL" = $deciderApiUrl
    "ORCHESTRATOR_DECIDER_MODEL" = (Get-MapValueOrDefault -Map $envOrch -Key "ORCHESTRATOR_DECIDER_MODEL" -DefaultValue "gemma4:latest")
    "ORCHESTRATOR_SYNTHESIS_API_FORMAT" = (Get-MapValueOrDefault -Map $envOrch -Key "ORCHESTRATOR_SYNTHESIS_API_FORMAT" -DefaultValue "ollama")
    "ORCHESTRATOR_SYNTHESIS_API_URL" = $synthesisApiUrl
    "ORCHESTRATOR_SYNTHESIS_MODEL" = (Get-MapValueOrDefault -Map $envOrch -Key "ORCHESTRATOR_SYNTHESIS_MODEL" -DefaultValue "gemma4:latest")
}

if ($envOrch.ContainsKey("ORCHESTRATOR_DECIDER_API_KEY")) {
    $orchEnv["ORCHESTRATOR_DECIDER_API_KEY"] = $envOrch["ORCHESTRATOR_DECIDER_API_KEY"]
}
if ($envOrch.ContainsKey("ORCHESTRATOR_DECIDER_API_KEY_HEADER")) {
    $orchEnv["ORCHESTRATOR_DECIDER_API_KEY_HEADER"] = $envOrch["ORCHESTRATOR_DECIDER_API_KEY_HEADER"]
}
if ($envOrch.ContainsKey("ORCHESTRATOR_SYNTHESIS_API_KEY")) {
    $orchEnv["ORCHESTRATOR_SYNTHESIS_API_KEY"] = $envOrch["ORCHESTRATOR_SYNTHESIS_API_KEY"]
}
if ($envOrch.ContainsKey("ORCHESTRATOR_SYNTHESIS_API_KEY_HEADER")) {
    $orchEnv["ORCHESTRATOR_SYNTHESIS_API_KEY_HEADER"] = $envOrch["ORCHESTRATOR_SYNTHESIS_API_KEY_HEADER"]
}

Write-Host ("Root: {0}" -f $root)
Write-Host ("Python: {0}" -f $PythonPath)
Write-Host ("Restart mode: {0}" -f [bool]$Restart)

# Start services.
Start-UvicornService `
    -Name "vision-service" `
    -Module "vision_service.app:app" `
    -Port $visionPort `
    -PythonExe $PythonPath `
    -WorkingDir $root `
    -StdOutLog (Join-Path $logDir "vision_service.out.log") `
    -StdErrLog (Join-Path $logDir "vision_service.err.log") `
    -EnvVars $visionEnv `
    -RestartPort:$Restart | Out-Null

Start-UvicornService `
    -Name "retrieval-service" `
    -Module "retrieval_service.app:app" `
    -Port $retrievalPort `
    -PythonExe $PythonPath `
    -WorkingDir $root `
    -StdOutLog (Join-Path $logDir "retrieval.out.log") `
    -StdErrLog (Join-Path $logDir "retrieval.err.log") `
    -EnvVars $retrievalEnv `
    -RestartPort:$Restart | Out-Null

Start-UvicornService `
    -Name "orchestrator-service" `
    -Module "orchestrator_service.app:app" `
    -Port $orchestratorPort `
    -PythonExe $PythonPath `
    -WorkingDir $root `
    -StdOutLog (Join-Path $logDir "orchestrator.out.log") `
    -StdErrLog (Join-Path $logDir "orchestrator.err.log") `
    -EnvVars $orchEnv `
    -RestartPort:$Restart | Out-Null

Write-Host ""
Write-Host "Health checks..."

$visionOk = Wait-Health -Name "vision-service" -Url ("http://127.0.0.1:{0}/health" -f $visionPort) -TimeoutSec $HealthTimeoutSec
$retrievalOk = Wait-Health -Name "retrieval-service" -Url ("http://127.0.0.1:{0}/health" -f $retrievalPort) -TimeoutSec $HealthTimeoutSec
$orchOk = Wait-Health -Name "orchestrator-service" -Url ("http://127.0.0.1:{0}/health" -f $orchestratorPort) -TimeoutSec $HealthTimeoutSec

Write-Host ""
if (-not (Test-HttpOk -Url "http://127.0.0.1:6333/collections")) {
    if ((Get-MapValueOrDefault -Map $retrievalEnv -Key "QDRANT_URL" -DefaultValue "") -match "^local://") {
        Write-Host ("[INFO] Qdrant local-mode enabled at {0}" -f $retrievalEnv["QDRANT_URL"])
    } else {
        Write-Host "[WARN] Qdrant not healthy at http://127.0.0.1:6333/collections (retrieval may fail)." -ForegroundColor Yellow
    }
}
if (-not (Test-HttpOk -Url "http://127.0.0.1:11434/api/tags")) {
    Write-Host "[WARN] Ollama not healthy at http://127.0.0.1:11434/api/tags (embedding/model calls may fail)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Summary:"
Write-Host ("  vision-service:      {0}" -f ($(if ($visionOk) { "OK" } else { "FAIL" })))
Write-Host ("  retrieval-service:   {0}" -f ($(if ($retrievalOk) { "OK" } else { "FAIL" })))
Write-Host ("  orchestrator-service:{0}" -f ($(if ($orchOk) { "OK" } else { "FAIL" })))

if (-not $retrievalOk) {
    $retrievalErr = Join-Path $logDir "retrieval.err.log"
    if (Test-Path -LiteralPath $retrievalErr) {
        Write-Host ""
        Write-Host ("retrieval-service error tail: {0}" -f $retrievalErr) -ForegroundColor Yellow
        Get-Content -LiteralPath $retrievalErr -Tail 40
    }
}

if (-not ($visionOk -and $retrievalOk -and $orchOk)) {
    exit 1
}
