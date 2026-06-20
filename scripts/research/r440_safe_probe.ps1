param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ProbeArgs
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

& (Join-Path $PSScriptRoot "r440_resource_preflight.ps1")

$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"
$env:PYTHONHASHSEED = "0"
$env:PYTHONUNBUFFERED = "1"

python scripts/research/r440_growth_probe.py @ProbeArgs
exit $LASTEXITCODE
