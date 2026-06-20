param(
    [int]$Steps = 160,
    [int]$TextSteps = 80,
    [int]$Batch = 4,
    [int]$Seed = 42140,
    [int]$LogEvery = 20,
    [int]$EvalEvery = 40,
    [int]$EvalBatches = 4,
    [int]$SmokeSteps = 4,
    [int]$PollSeconds = 15,
    [int]$MaxNoGrowthMinutes = 20,
    [int]$MaxGpuTempC = 86,
    [int]$MaxGpuMemoryMb = 11000,
    [switch]$EnableCheckpoints,
    [switch]$SkipTextProbe,
    [switch]$SmokeOnly,
    [switch]$AcknowledgeLongRunRisk,
    [string]$JobName = ("run-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
)

$ErrorActionPreference = "Stop"

if (-not $SmokeOnly -and -not $AcknowledgeLongRunRisk) {
    throw "This runner can hold the GPU at full utilization for a long time. Use -SmokeOnly for bounded validation, or pass -AcknowledgeLongRunRisk only after the user explicitly grants exclusive long-run mode."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$modelDir = Join-Path $repoRoot "src\model_v12"
$trainExe = Join-Path $modelDir "train.exe"
$jobRoot = Join-Path $repoRoot ".work\jobs\r421-macro-ce-quotient-overnight\$JobName"
$logDir = Join-Path $jobRoot "logs"
$ckptDir = Join-Path $jobRoot "checkpoints"
$telemetryPath = Join-Path $jobRoot "gpu-telemetry.csv"
$manifestPath = Join-Path $jobRoot "manifest.json"
$analysisPath = Join-Path $jobRoot "analysis.json"

New-Item -ItemType Directory -Force -Path $logDir, $ckptDir | Out-Null

function Get-ProjectTrainProcesses {
    Get-CimInstance Win32_Process -Filter "name = 'train.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            ($_.ExecutablePath -like "*code-ai-workspace*") -or
            ($_.CommandLine -like "*code-ai-workspace*") -or
            ($_.ExecutablePath -like "*model_v12*") -or
            ($_.CommandLine -like "*model_v12*")
        }
}

function Get-GpuSnapshot {
    $line = & nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0 -or -not $line) {
        throw "nvidia-smi telemetry failed"
    }
    $parts = $line -split ",\s*"
    [pscustomobject]@{
        timestamp = $parts[0]
        name = $parts[1]
        util_gpu = [double]$parts[2]
        mem_used_mb = [double]$parts[3]
        mem_total_mb = [double]$parts[4]
        temp_c = [double]$parts[5]
        power_w = [double]$parts[6]
        power_limit_w = [double]$parts[7]
    }
}

function Write-GpuTelemetry {
    param([string]$Phase)
    $snap = Get-GpuSnapshot
    if (-not (Test-Path $telemetryPath)) {
        "wall_time,phase,gpu_timestamp,name,util_gpu,mem_used_mb,mem_total_mb,temp_c,power_w,power_limit_w" |
            Set-Content -Encoding UTF8 $telemetryPath
    }
    $row = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f `
        (Get-Date -Format o), $Phase, $snap.timestamp, $snap.name, $snap.util_gpu, `
        $snap.mem_used_mb, $snap.mem_total_mb, $snap.temp_c, $snap.power_w, $snap.power_limit_w
    Add-Content -Encoding UTF8 $telemetryPath $row
    return $snap
}

function Assert-NoTrainProcess {
    $running = @(Get-ProjectTrainProcesses)
    if ($running.Count -gt 0) {
        $desc = ($running | ForEach-Object { "pid=$($_.ProcessId) cmd=$($_.CommandLine)" }) -join "`n"
        throw "Refusing to start: project train.exe already active.`n$desc"
    }
}

function Convert-ArgListToCommandLine {
    param([string[]]$ArgsList)
    ($ArgsList | ForEach-Object {
        if ($_ -match "\s") { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join " "
}

function Invoke-TrainArm {
    param(
        [string]$Name,
        [string[]]$ArgsList,
        [string[]]$RequiredPatterns
    )

    Assert-NoTrainProcess
    $stdout = Join-Path $logDir "$Name.stdout.log"
    $stderr = Join-Path $logDir "$Name.stderr.log"
    $cmdline = ".\train.exe " + (Convert-ArgListToCommandLine $ArgsList)
    $redirected = "$cmdline > `"$stdout`" 2> `"$stderr`""

    Write-Host "[ARM_START] $Name"
    Write-Host "[COMMAND] $cmdline"
    Write-GpuTelemetry "$Name-before" | Out-Null

    $process = Start-Process -FilePath "cmd.exe" `
        -ArgumentList @("/d", "/s", "/c", $redirected) `
        -WorkingDirectory $modelDir `
        -NoNewWindow `
        -PassThru

    $lastGrowthAt = Get-Date
    $lastSize = -1L
    while (-not $process.HasExited) {
        Start-Sleep -Seconds $PollSeconds
        $snap = Write-GpuTelemetry $Name
        if ($snap.temp_c -ge $MaxGpuTempC) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            Get-ProjectTrainProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
            throw "Stopped ${Name}: GPU temperature $($snap.temp_c)C reached threshold $MaxGpuTempC C"
        }
        if ($snap.mem_used_mb -ge $MaxGpuMemoryMb) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            Get-ProjectTrainProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
            throw "Stopped ${Name}: GPU memory $($snap.mem_used_mb) MB reached threshold $MaxGpuMemoryMb MB"
        }

        $size = 0L
        if (Test-Path $stdout) { $size += (Get-Item $stdout).Length }
        if (Test-Path $stderr) { $size += (Get-Item $stderr).Length }
        if ($size -gt $lastSize) {
            $lastSize = $size
            $lastGrowthAt = Get-Date
        } elseif (((Get-Date) - $lastGrowthAt).TotalMinutes -ge $MaxNoGrowthMinutes) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            Get-ProjectTrainProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
            throw "Stopped ${Name}: stdout/stderr did not grow for $MaxNoGrowthMinutes minutes"
        }
    }

    $process.WaitForExit()
    $process.Refresh()
    Write-GpuTelemetry "$Name-after" | Out-Null
    if ($null -ne $process.ExitCode -and $process.ExitCode -ne 0) {
        throw "$Name failed with exit code $($process.ExitCode). See $stdout and $stderr"
    }

    $combined = ""
    if (Test-Path $stdout) { $combined += Get-Content -Raw $stdout }
    if (Test-Path $stderr) { $combined += "`n" + (Get-Content -Raw $stderr) }
    foreach ($pattern in $RequiredPatterns) {
        if ($combined -notmatch $pattern) {
            throw "$Name completed but missing required pattern: $pattern"
        }
    }

    Assert-NoTrainProcess
    Write-Host "[ARM_DONE] $Name"
}

function Parse-KeyValueLine {
    param([string]$Line)
    $obj = [ordered]@{}
    foreach ($match in [regex]::Matches($Line, "([A-Za-z0-9_\/.-]+)=([A-Za-z0-9_+.-]+)")) {
        $key = $match.Groups[1].Value
        $value = $match.Groups[2].Value
        $number = 0.0
        if ([double]::TryParse($value, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$number)) {
            $obj[$key] = $number
        } else {
            $obj[$key] = $value
        }
    }
    return $obj
}

function Analyze-Arm {
    param([string]$Name)
    $stdout = Join-Path $logDir "$Name.stdout.log"
    $stderr = Join-Path $logDir "$Name.stderr.log"
    $lines = @()
    if (Test-Path $stdout) { $lines += Get-Content $stdout }
    if (Test-Path $stderr) { $lines += Get-Content $stderr }

    $trainLines = @($lines | Where-Object { $_ -match "^\[TRAIN\] step=" })
    $evalLines = @($lines | Where-Object { $_ -match "^\[EVAL\]" })
    $observerLines = @($lines | Where-Object { $_ -match "^\[R421_MACRO_CE_QUOTIENT_RESULT\]" })
    $requested = @($lines | Where-Object { $_ -match "^\[R421_MACRO_CE_QUOTIENT\]" })
    $complete = [bool]($lines | Where-Object { $_ -match "^\[TRAIN\] Complete" } | Select-Object -First 1)

    $lastTrain = if ($trainLines.Count -gt 0) { Parse-KeyValueLine $trainLines[-1] } else { [ordered]@{} }
    $lastEval = if ($evalLines.Count -gt 0) { Parse-KeyValueLine $evalLines[-1] } else { [ordered]@{} }
    $lastObserver = if ($observerLines.Count -gt 0) { Parse-KeyValueLine $observerLines[-1] } else { [ordered]@{} }

    [ordered]@{
        name = $Name
        complete = $complete
        train_line_count = $trainLines.Count
        eval_line_count = $evalLines.Count
        observer_requested = ($requested.Count -gt 0)
        observer_result_count = $observerLines.Count
        last_train = $lastTrain
        last_eval = $lastEval
        last_observer = $lastObserver
    }
}

function Build-ArmArgs {
    param(
        [string]$Dataset,
        [int]$ArmSteps,
        [string]$Checkpoint,
        [switch]$Observer,
        [int]$ArmSeed
    )
    $args = @(
        $Dataset,
        "$ArmSteps"
    )
    if ($Checkpoint) {
        $args += $Checkpoint
    }
    $args += @(
        "--batch", "$Batch",
        "--seed", "$ArmSeed",
        "--teaching-off",
        "--log-every", "$LogEvery",
        "--eval-every", "$EvalEvery",
        "--eval-batches", "$EvalBatches"
    )
    if ($Observer) {
        $args += @("--macro-ce-quotient-observer", "--macro-ce-quotient-seed", "$Seed")
    }
    return $args
}

Assert-NoTrainProcess
$initialGpu = Write-GpuTelemetry "preflight"

$datasetWiki = Join-Path $modelDir "wiki_1mb.bin"
$datasetText = Join-Path $modelDir "text_data.bin"
foreach ($required in @($datasetWiki, $datasetText)) {
    if (-not (Test-Path $required)) {
        throw "Missing dataset: $required"
    }
}

$buildLog = Join-Path $logDir "build.log"
Push-Location $modelDir
try {
    & mingw32-make USE_VULKAN=1 train *> $buildLog
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed. See $buildLog"
    }
} finally {
    Pop-Location
}

$baselineCkpt = if ($EnableCheckpoints) { Join-Path $ckptDir "baseline-wiki.ckpt" } else { "" }
$observerCkpt = if ($EnableCheckpoints) { Join-Path $ckptDir "observer-wiki.ckpt" } else { "" }
$textCkpt = if ($EnableCheckpoints) { Join-Path $ckptDir "observer-text.ckpt" } else { "" }
$baselineArgs = Build-ArmArgs ".\wiki_1mb.bin" $Steps $baselineCkpt -ArmSeed $Seed
$observerArgs = Build-ArmArgs ".\wiki_1mb.bin" $Steps $observerCkpt -Observer -ArmSeed $Seed
$textArgs = Build-ArmArgs ".\text_data.bin" $TextSteps $textCkpt -Observer -ArmSeed ($Seed + 1)

$commands = [ordered]@{
    build = "mingw32-make USE_VULKAN=1 train"
    baseline_wiki = ".\train.exe " + (Convert-ArgListToCommandLine $baselineArgs)
    observer_wiki = ".\train.exe " + (Convert-ArgListToCommandLine $observerArgs)
    observer_text = ".\train.exe " + (Convert-ArgListToCommandLine $textArgs)
}

$manifest = [ordered]@{
    job_name = $JobName
    created_at = (Get-Date -Format o)
    repo_root = $repoRoot
    model_dir = $modelDir
    steps = $Steps
    text_steps = $TextSteps
    batch = $Batch
    seed = $Seed
    log_every = $LogEvery
    eval_every = $EvalEvery
    eval_batches = $EvalBatches
    smoke_steps = $SmokeSteps
    poll_seconds = $PollSeconds
    max_no_growth_minutes = $MaxNoGrowthMinutes
    max_gpu_temp_c = $MaxGpuTempC
    max_gpu_memory_mb = $MaxGpuMemoryMb
    enable_checkpoints = [bool]$EnableCheckpoints
    skip_text_probe = [bool]$SkipTextProbe
    smoke_only = [bool]$SmokeOnly
    initial_gpu = $initialGpu
    commands = $commands
    outputs = [ordered]@{
        job_root = $jobRoot
        log_dir = $logDir
        checkpoint_dir = $ckptDir
        telemetry = $telemetryPath
        analysis = $analysisPath
    }
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $manifestPath

$smokeLogEvery = $LogEvery
$LogEvery = 1
try {
    $smokeBaselineCkpt = if ($EnableCheckpoints) { Join-Path $ckptDir "smoke-baseline.ckpt" } else { "" }
    $smokeObserverCkpt = if ($EnableCheckpoints) { Join-Path $ckptDir "smoke-observer.ckpt" } else { "" }
    $smokeBaselineArgs = Build-ArmArgs ".\wiki_1mb.bin" $SmokeSteps $smokeBaselineCkpt -ArmSeed $Seed
    $smokeObserverArgs = Build-ArmArgs ".\wiki_1mb.bin" $SmokeSteps $smokeObserverCkpt -Observer -ArmSeed $Seed
} finally {
    $LogEvery = $smokeLogEvery
}

Invoke-TrainArm "smoke-baseline-wiki" $smokeBaselineArgs @("\[TRAIN\] Complete")
Invoke-TrainArm "smoke-observer-wiki" $smokeObserverArgs @(
    "\[R421_MACRO_CE_QUOTIENT\] requested=1",
    "\[R421_MACRO_CE_QUOTIENT_RESULT\]",
    "observer=1",
    "training=0",
    "\[TRAIN\] Complete"
)

if (-not $SmokeOnly) {
    Invoke-TrainArm "baseline-wiki" $baselineArgs @("\[TRAIN\] Complete", "\[EVAL\]")
    Invoke-TrainArm "observer-wiki" $observerArgs @(
        "\[R421_MACRO_CE_QUOTIENT\] requested=1",
        "\[R421_MACRO_CE_QUOTIENT_RESULT\]",
        "observer=1",
        "training=0",
        "\[TRAIN\] Complete",
        "\[EVAL\]"
    )
    if (-not $SkipTextProbe) {
        Invoke-TrainArm "observer-text" $textArgs @(
            "\[R421_MACRO_CE_QUOTIENT\] requested=1",
            "\[R421_MACRO_CE_QUOTIENT_RESULT\]",
            "observer=1",
            "training=0",
            "\[TRAIN\] Complete",
            "\[EVAL\]"
        )
    }
}

$arms = @("smoke-baseline-wiki", "smoke-observer-wiki")
if (-not $SmokeOnly) {
    $arms += @("baseline-wiki", "observer-wiki")
    if (-not $SkipTextProbe) { $arms += "observer-text" }
}

$analysis = [ordered]@{
    job_name = $JobName
    finished_at = (Get-Date -Format o)
    manifest = $manifestPath
    telemetry = $telemetryPath
    arms = @($arms | ForEach-Object { Analyze-Arm $_ })
    leftover_train_processes = @(Get-ProjectTrainProcesses | ForEach-Object {
        [ordered]@{ pid = $_.ProcessId; command_line = $_.CommandLine }
    })
}
$analysis | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $analysisPath

if ($analysis.leftover_train_processes.Count -gt 0) {
    throw "Analysis complete but leftover train.exe process exists"
}

Write-Host "[R421_OVERNIGHT_DONE] $analysisPath"
