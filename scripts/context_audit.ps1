param(
    [string]$Root = (Get-Location).Path,
    [int]$LargeKB = 100,
    [int]$HugeKB = 500
)

$ErrorActionPreference = "Stop"

$excludedPathPattern = '(\\.workspace\\(analysis|codex-schema)(\\|$))|\\(\.git|\.venv|venv|__pycache__|\.ruff_cache|\.mypy_cache|\.pytest_cache|build|data|experiments|results|runs|checkpoints|\.checkpoints|modal-results|models|\.work)(\\|$)'
$excludedFilePattern = '\\.workspace\\r\d+_output\.txt$'
$textExtensions = @(".md", ".txt", ".log", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".py", ".ps1")

Write-Host "Context audit root: $Root"
Write-Host "Excluded cold directories: data, experiments, results, runs, checkpoints, models, modal-results, .work, .workspace/analysis, .workspace/codex-schema, build outputs, caches, and virtual environments"
Write-Host ""

$allFiles = Get-ChildItem -LiteralPath $Root -Recurse -Force -File -ErrorAction SilentlyContinue |
    Where-Object { ($_.FullName -notmatch $excludedPathPattern) -and ($_.FullName -notmatch $excludedFilePattern) }

Write-Host "Largest files:"
$allFiles |
    Sort-Object Length -Descending |
    Select-Object -First 25 @{Name="MB";Expression={[math]::Round($_.Length / 1MB, 2)}}, Extension, FullName |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Largest text-like files:"
$allFiles |
    Where-Object { $textExtensions -contains $_.Extension.ToLowerInvariant() } |
    Sort-Object Length -Descending |
    Select-Object -First 25 @{Name="KB";Expression={[math]::Round($_.Length / 1KB, 1)}}, Extension, FullName |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Hot-path text risk summary:"
$hotScopes = @("memory", ".workspace\research", ".workspace\codex", ".work\knowledge", "docs", "src\model_v12\docs")
$rows = @()
foreach ($scope in $hotScopes) {
    $path = Join-Path $Root $scope
    if (Test-Path -LiteralPath $path) {
        $files = @(Get-ChildItem -LiteralPath $path -Recurse -Force -File -ErrorAction SilentlyContinue |
            Where-Object { $textExtensions -contains $_.Extension.ToLowerInvariant() })
        $large = @($files | Where-Object { $_.Length -gt ($LargeKB * 1KB) })
        $huge = @($files | Where-Object { $_.Length -gt ($HugeKB * 1KB) })
        $largest = $files | Sort-Object Length -Descending | Select-Object -First 1
        $rows += [PSCustomObject]@{
            Scope = $scope
            Files = $files.Count
            Large = $large.Count
            Huge = $huge.Count
            LargestKB = if ($largest) { [math]::Round($largest.Length / 1KB, 1) } else { 0 }
        }
    }
}
$rows | Format-Table -AutoSize

Write-Host ""
Write-Host "Files over $($LargeKB)KB in hot/default paths:"
foreach ($scope in $hotScopes) {
    $path = Join-Path $Root $scope
    if (Test-Path -LiteralPath $path) {
        Get-ChildItem -LiteralPath $path -Recurse -Force -File -ErrorAction SilentlyContinue |
            Where-Object {
                ($textExtensions -contains $_.Extension.ToLowerInvariant()) -and
                ($_.Length -gt ($LargeKB * 1KB))
            } |
            Sort-Object Length -Descending |
            Select-Object @{Name="KB";Expression={[math]::Round($_.Length / 1KB, 1)}}, FullName |
            Format-Table -AutoSize
    }
}

Write-Host ""
Write-Host "Files over $($HugeKB)KB in hot paths:"
foreach ($scope in $hotScopes) {
    $path = Join-Path $Root $scope
    if (Test-Path -LiteralPath $path) {
        Get-ChildItem -LiteralPath $path -Recurse -Force -File -ErrorAction SilentlyContinue |
            Where-Object {
                ($textExtensions -contains $_.Extension.ToLowerInvariant()) -and
                ($_.Length -gt ($HugeKB * 1KB))
            } |
            Sort-Object Length -Descending |
            Select-Object @{Name="KB";Expression={[math]::Round($_.Length / 1KB, 1)}}, FullName |
            Format-Table -AutoSize
    }
}
