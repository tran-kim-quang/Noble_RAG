$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$targetScript = Join-Path $repoRoot "livetalking\run_with_idle.ps1"

if (-not (Test-Path $targetScript)) {
  Write-Host "Cannot find target script: $targetScript"
  exit 1
}

# Default avatar when LIVETALKING_AVATAR_ID is not provided.
if (-not $env:LIVETALKING_AVATAR_ID -or [string]::IsNullOrWhiteSpace($env:LIVETALKING_AVATAR_ID)) {
  $env:LIVETALKING_AVATAR_ID = "half"
}
if ($env:LIVETALKING_AVATAR_ID -in @("half-avatar", "half_avatar")) {
  $env:LIVETALKING_AVATAR_ID = "half"
}

& $targetScript @args
