param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$falconDir = if ($env:FALCON_DIR) { $env:FALCON_DIR } else { Join-Path $root "data\\FALCON" }

if (-not (Test-Path $falconDir)) {
    Write-Error "FALCON directory not found: $falconDir"
    exit 1
}

Push-Location $falconDir
try {
    python scripts\example_mlp_minimal.py @Args
} finally {
    Pop-Location
}
