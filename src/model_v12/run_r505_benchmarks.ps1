$ErrorActionPreference = "Stop"

Write-Host "=== Building R505 Architectures ==="

Write-Host "Cleaning..."
mingw32-make clean

Write-Host "Building v12 Base..."
mingw32-make train USE_VULKAN=1
Copy-Item train.exe train_v12_base.exe -ErrorAction SilentlyContinue

Write-Host "Cleaning..."
mingw32-make clean

Write-Host "Building v13 Multiscale Permutation..."
mingw32-make v13_multiscale USE_VULKAN=1

Write-Host "Cleaning..."
mingw32-make clean

Write-Host "Building v14 Hadamard Spectral..."
mingw32-make v14_hadamard USE_VULKAN=1

Write-Host "Cleaning..."
mingw32-make clean

Write-Host "Building v15 GF(2) Automata..."
mingw32-make v15_gf2 USE_VULKAN=1

Write-Host "=== Compilation Complete ==="

function Run-Benchmark {
    param([string]$ExeName, [string]$ArchName)
    Write-Host ""
    Write-Host "--- Benchmarking $ArchName ---"
    if (Test-Path $ExeName) {
        # We run it for just a short duration or let it do its default 100 steps
        # To avoid blocking forever, we'll run it and capture the last few lines
        # But actually train.exe runs 100 steps by default. We'll run it and measure time.
        
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $process = Start-Process -FilePath ".\$ExeName" -ArgumentList "--steps 3" -NoNewWindow -PassThru -RedirectStandardOutput "bench_$ArchName.log" -RedirectStandardError "bench_${ArchName}_err.log"
        $process.WaitForExit()
        $sw.Stop()
        
        Write-Host "Completed in $($sw.Elapsed.TotalSeconds) seconds."
        
        # Check for [R505] prints to confirm activation
        $activated = Select-String -Path "bench_$ArchName.log" -Pattern "\[R505\]" | Select-Object -First 1
        if ($activated) {
            Write-Host "Activation Confirmed: $($activated.Line)" -ForegroundColor Green
        } else {
            Write-Host "Activation Not Found (Might be base v12)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Executable $ExeName not found!" -ForegroundColor Red
    }
}

Run-Benchmark "train_v12_base.exe" "Base"
Run-Benchmark "train_v13_multiscale.exe" "Multiscale"
Run-Benchmark "train_v14_hadamard.exe" "Hadamard"
Run-Benchmark "train_v15_gf2.exe" "GF2"

Write-Host "=== Benchmarks Finished ==="
