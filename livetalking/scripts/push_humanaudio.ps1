Param(
    [Parameter(Mandatory = $true)]
    [int]$SessionId,
    [int]$Port = 8011,
    [string]$Text = "Xin chao, day la bai test lipsync local. Neu thay mieng dong bo theo giọng noi, pipeline dang hoat dong.",
    [string]$OutputWav = ""
)

$ErrorActionPreference = "Stop"

$liveTalkingDir = Split-Path -Parent $PSScriptRoot
if (-not $OutputWav) {
    $OutputWav = Join-Path $liveTalkingDir "tmp_lipsync_local.wav"
}

Add-Type -AssemblyName System.Speech

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.Rate = 0
    $synth.Volume = 100
    $synth.SetOutputToWaveFile($OutputWav)
    $synth.Speak($Text)
} finally {
    $synth.Dispose()
}

if (-not (Test-Path $OutputWav)) {
    throw "Khong tao duoc wav test: $OutputWav"
}

$url = "http://127.0.0.1:$Port/humanaudio"
Write-Host "Uploading audio -> $url (sessionid=$SessionId)"

$resp = & curl.exe -sS -X POST $url `
    -F "sessionid=$SessionId" `
    -F "file=@$OutputWav"

Write-Host "Server response:" $resp
