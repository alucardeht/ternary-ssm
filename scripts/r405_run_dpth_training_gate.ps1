param(
    [int]$Steps = 520,
    [int]$Batch = 1,
    [int]$Seed = 40521,
    [int]$LogEvery = 40,
    [int]$EvalEvery = 130,
    [int]$EvalBatches = 4,
    [int]$DpthPool = 4,
    [int]$PollSeconds = 15,
    [int]$MaxGpuTempC = 86,
    [int]$MaxGpuMemoryMb = 11000,
    [int]$MaxNoGrowthMinutes = 20,
    [string]$JobName = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$modelDir = Join-Path $repoRoot "src\model_v12"
$trainExe = Join-Path $modelDir "train.exe"
$datasetPath = Join-Path $modelDir "wiki_1mb.bin"
$analyzer = Join-Path $modelDir "tools\r405_dpth_training_analyze.py"

if ([string]::IsNullOrWhiteSpace($JobName)) {
    $JobName = "nightly-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

$jobRoot = Join-Path $repoRoot ".work\jobs\r405-s1-dpth-training-gate\$JobName"
$logRoot = Join-Path $jobRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$statusPath = Join-Path $jobRoot "status.jsonl"
$telemetryPath = Join-Path $jobRoot "gpu-telemetry.csv"
$manifestPath = Join-Path $jobRoot "manifest.json"
$analysisPath = Join-Path $jobRoot "analysis.json"

function Write-JsonLine {
    param([hashtable]$Data)
    $Data["timestamp"] = (Get-Date).ToString("o")
    ($Data | ConvertTo-Json -Compress) | Add-Content -Path $statusPath -Encoding UTF8
}

function Get-ProjectTrainProcesses {
    Get-CimInstance Win32_Process |
        Where-Object {
            ($_.Name -ieq "train.exe" -and $_.CommandLine -match [regex]::Escape($modelDir)) -or
            ($_.CommandLine -match "src\\model_v12" -and $_.CommandLine -match "train.exe")
        } |
        Select-Object ProcessId, ParentProcessId, Name, CommandLine
}

function Read-GpuSnapshot {
    $gpuLine = (& nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,pstate --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($gpuLine)) {
        throw "nvidia-smi returned no GPU telemetry"
    }
    $parts = $gpuLine.Split(",") | ForEach-Object { $_.Trim() }
    [pscustomobject]@{
        UtilGpu = [double]$parts[0]
        UtilMemory = [double]$parts[1]
        MemoryUsedMb = [double]$parts[2]
        MemoryTotalMb = [double]$parts[3]
        TempC = [double]$parts[4]
        PowerW = [double]$parts[5]
        PState = $parts[6]
    }
}

function Write-GpuTelemetry {
    param(
        [string]$Arm,
        [int]$ProcId,
        [int]$ElapsedSeconds
    )
    $snap = Read-GpuSnapshot
    $row = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}" -f `
        (Get-Date).ToString("o"), $Arm, $ProcId, $ElapsedSeconds, `
        $snap.UtilGpu, $snap.UtilMemory, $snap.MemoryUsedMb, $snap.MemoryTotalMb, `
        $snap.TempC, $snap.PowerW, $snap.PState
    Add-Content -Path $telemetryPath -Value $row -Encoding ASCII
    return $snap
}

function Assert-LogContains {
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$Description
    )
    if (-not (Select-String -Path $Path -Pattern $Pattern -Quiet)) {
        throw "Missing $Description in $Path"
    }
}

function Invoke-Arm {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [bool]$RequiresDpth
    )

    $stdoutPath = Join-Path $logRoot "$Name.stdout.log"
    $stderrPath = Join-Path $logRoot "$Name.stderr.log"
    Remove-Item -LiteralPath $stdoutPath, $stderrPath -ErrorAction SilentlyContinue

    Write-JsonLine @{ event = "arm_start"; arm = $Name; args = ($Arguments -join " ") }
    $proc = Start-Process -FilePath $trainExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $modelDir `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -PassThru

    $start = Get-Date
    $lastGrowth = Get-Date
    $lastStdoutLength = 0L
    $lastStderrLength = 0L
    $sampleIndex = 0

    try {
        while (-not $proc.HasExited) {
            Start-Sleep -Seconds $PollSeconds
            $proc.Refresh()
            $elapsed = [int]((Get-Date) - $start).TotalSeconds
            $sampleIndex++

            $snap = Write-GpuTelemetry -Arm $Name -ProcId $proc.Id -ElapsedSeconds $elapsed
            if ($snap.TempC -ge $MaxGpuTempC) {
                throw "GPU temperature $($snap.TempC)C exceeded limit $MaxGpuTempC C"
            }
            if ($snap.MemoryUsedMb -ge $MaxGpuMemoryMb) {
                throw "GPU memory $($snap.MemoryUsedMb)MB exceeded limit $MaxGpuMemoryMb MB"
            }

            $stdoutLength = if (Test-Path $stdoutPath) { (Get-Item $stdoutPath).Length } else { 0L }
            $stderrLength = if (Test-Path $stderrPath) { (Get-Item $stderrPath).Length } else { 0L }
            if ($stdoutLength -ne $lastStdoutLength -or $stderrLength -ne $lastStderrLength) {
                $lastGrowth = Get-Date
                $lastStdoutLength = $stdoutLength
                $lastStderrLength = $stderrLength
            }
            if (((Get-Date) - $lastGrowth).TotalMinutes -ge $MaxNoGrowthMinutes) {
                throw "No stdout/stderr growth for $MaxNoGrowthMinutes minutes"
            }

            if ($sampleIndex % 4 -eq 0) {
                Write-JsonLine @{
                    event = "arm_monitor"
                    arm = $Name
                    pid = $proc.Id
                    elapsed_seconds = $elapsed
                    gpu_util = $snap.UtilGpu
                    memory_used_mb = $snap.MemoryUsedMb
                    temp_c = $snap.TempC
                    power_w = $snap.PowerW
                    stdout_bytes = $stdoutLength
                    stderr_bytes = $stderrLength
                }
            }
        }
    } catch {
        Write-JsonLine @{ event = "arm_stop"; arm = $Name; reason = $_.Exception.Message; pid = $proc.Id }
        if (-not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force
        }
        throw
    }

    $proc.WaitForExit()
    $proc.Refresh()
    $exitCode = $proc.ExitCode
    Write-JsonLine @{ event = "arm_exit"; arm = $Name; exit_code = $exitCode }
    if (-not [string]::IsNullOrWhiteSpace([string]$exitCode) -and $exitCode -ne 0) {
        throw "Arm $Name exited with code $exitCode"
    }

    Assert-LogContains -Path $stdoutPath -Pattern "\[TRAIN\] Complete" -Description "training completion"
    Assert-LogContains -Path $stdoutPath -Pattern "\[EVAL\]" -Description "eval output"
    if ($RequiresDpth) {
        Assert-LogContains -Path $stdoutPath -Pattern "\[DPTH_GATE\].*step=" -Description "DPTH step output"
    }

    return $stdoutPath
}

Write-JsonLine @{ event = "preflight_start"; job_root = $jobRoot }

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi is not available"
}
if (-not (Test-Path $datasetPath)) {
    throw "Dataset not found: $datasetPath"
}
if (-not (Test-Path $analyzer)) {
    throw "Analyzer not found: $analyzer"
}

$existing = @(Get-ProjectTrainProcesses)
if ($existing.Count -gt 0) {
    $existing | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $jobRoot "blocking-processes.json") -Encoding UTF8
    throw "Existing project train process detected"
}

"timestamp,arm,pid,elapsed_seconds,gpu_util_pct,gpu_mem_util_pct,memory_used_mb,memory_total_mb,temp_c,power_w,pstate" |
    Set-Content -Path $telemetryPath -Encoding ASCII

Read-GpuSnapshot | Out-Null
python -m py_compile $analyzer

if (-not $SkipBuild) {
    Write-JsonLine @{ event = "build_start" }
    Push-Location $modelDir
    try {
        mingw32-make USE_VULKAN=1 train
    } finally {
        Pop-Location
    }
    Write-JsonLine @{ event = "build_done" }
}

if (-not (Test-Path $trainExe)) {
    throw "train.exe not found after build"
}

$arms = @(
    @{
        Name = "a0-baseline"
        RequiresDpth = $false
        Args = @(".\wiki_1mb.bin", "$Steps", "--batch", "$Batch", "--seed", "$Seed", "--teaching-off", "--log-every", "$LogEvery", "--eval-every", "$EvalEvery", "--eval-batches", "$EvalBatches")
    },
    @{
        Name = "a1-packet"
        RequiresDpth = $true
        Args = @(".\wiki_1mb.bin", "$Steps", "--batch", "$Batch", "--seed", "$Seed", "--teaching-off", "--log-every", "$LogEvery", "--eval-every", "$EvalEvery", "--eval-batches", "$EvalBatches", "--dpth-gate", "--dpth-mode", "packet", "--dpth-pool", "$DpthPool")
    },
    @{
        Name = "a2-loss-control"
        RequiresDpth = $true
        Args = @(".\wiki_1mb.bin", "$Steps", "--batch", "$Batch", "--seed", "$Seed", "--teaching-off", "--log-every", "$LogEvery", "--eval-every", "$EvalEvery", "--eval-batches", "$EvalBatches", "--dpth-gate", "--dpth-mode", "loss-control", "--dpth-pool", "$DpthPool")
    }
)

@{
    job_name = $JobName
    job_root = $jobRoot
    model_dir = $modelDir
    train_exe = $trainExe
    dataset = $datasetPath
    analyzer = $analyzer
    steps = $Steps
    batch = $Batch
    seed = $Seed
    log_every = $LogEvery
    eval_every = $EvalEvery
    eval_batches = $EvalBatches
    dpth_pool = $DpthPool
    poll_seconds = $PollSeconds
    max_gpu_temp_c = $MaxGpuTempC
    max_gpu_memory_mb = $MaxGpuMemoryMb
    arms = $arms
} | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8

$stdoutLogs = @()
foreach ($arm in $arms) {
    $stdoutLogs += Invoke-Arm -Name $arm.Name -Arguments $arm.Args -RequiresDpth $arm.RequiresDpth
}

Write-JsonLine @{ event = "analysis_start" }
$analysis = & python $analyzer $stdoutLogs --min-ratio 1.10
$analysis | Set-Content -Path $analysisPath -Encoding UTF8
Write-JsonLine @{ event = "analysis_done"; analysis_path = $analysisPath }

$leftover = @(Get-ProjectTrainProcesses)
if ($leftover.Count -gt 0) {
    $leftover | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $jobRoot "leftover-processes.json") -Encoding UTF8
    throw "Leftover train process detected after run"
}

Write-JsonLine @{ event = "complete"; analysis_path = $analysisPath }
Write-Host $analysis
Write-Host "R405 DPTH training gate complete. Artifacts: $jobRoot"
