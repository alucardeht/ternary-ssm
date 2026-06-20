param(
    [int]$MaxGpuMemoryMiB = 3000,
    [int]$MinFreeRamMiB = 8192,
    [int]$MaxCpuLoadPercent = 85,
    [int]$MaxUntrackedFiles = 200,
    [int]$MaxUntrackedFileMiB = 100
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

function Fail($Message) {
    Write-Error "R440_PREFLIGHT_FAIL: $Message"
    exit 1
}

if (Test-Path ".git\index.lock") {
    Fail "git index lock exists; do not run R440 while Git or a watcher is writing the index"
}

$rootText = [regex]::Escape($Root.Path)
$activeProjectProcesses = Get-CimInstance Win32_Process |
    Where-Object {
        ($_.Name -in @("python.exe", "python3.exe", "train.exe", "train")) -and
        ($_.CommandLine -match $rootText)
    } |
    Select-Object ProcessId, Name, CommandLine

if ($activeProjectProcesses) {
    $activeProjectProcesses | Format-List | Out-String | Write-Host
    Fail "project Python/train process is already active in this workspace"
}

$os = Get-CimInstance Win32_OperatingSystem
$freeRamMiB = [math]::Round($os.FreePhysicalMemory / 1024, 1)
if ($freeRamMiB -lt $MinFreeRamMiB) {
    Fail "free RAM ${freeRamMiB} MiB is below required ${MinFreeRamMiB} MiB"
}

$cpuLoad = [math]::Round(((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average), 1)
if ($cpuLoad -gt $MaxCpuLoadPercent) {
    Fail "CPU load ${cpuLoad}% is above allowed ${MaxCpuLoadPercent}%"
}

$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    $gpuLine = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null |
        Select-Object -First 1
    if ($gpuLine) {
        $parts = $gpuLine -split "," | ForEach-Object { $_.Trim() }
        $gpuUsedMiB = [int]$parts[0]
        $gpuTotalMiB = [int]$parts[1]
        $gpuUtil = [int]$parts[2]
        if ($gpuUsedMiB -gt $MaxGpuMemoryMiB) {
            Fail "GPU memory ${gpuUsedMiB}/${gpuTotalMiB} MiB is above allowed ${MaxGpuMemoryMiB} MiB for R440 CPU probes"
        }
    }
}

$untracked = @(git ls-files --others --exclude-standard)
if ($untracked.Count -gt $MaxUntrackedFiles) {
    Fail "untracked file count $($untracked.Count) is above allowed $MaxUntrackedFiles"
}

$maxUntrackedBytes = [int64]$MaxUntrackedFileMiB * 1MB
$largeUntracked = foreach ($path in $untracked) {
    $item = Get-Item -LiteralPath $path -ErrorAction SilentlyContinue
    if ($item -and $item.Length -gt $maxUntrackedBytes) {
        $item
    }
}
if ($largeUntracked) {
    $largeUntracked | Select-Object Length, FullName | Format-Table -AutoSize | Out-String | Write-Host
    Fail "one or more untracked files exceed ${MaxUntrackedFileMiB} MiB"
}

Write-Host "R440_RESOURCE_PREFLIGHT_OK free_ram_mib=$freeRamMiB cpu_load_percent=$cpuLoad untracked_count=$($untracked.Count)"
