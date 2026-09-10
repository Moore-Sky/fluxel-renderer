#Requires -Version 7.0
<#
.SYNOPSIS
Runs the release-only Windows GPU conformance gate and preserves its evidence.

.DESCRIPTION
This gate deliberately accepts only a clean x86_64-pc-windows-msvc checkout.
It injects HEAD into every fixture as FLUXEL_TEST_COMMIT and records the full
workspace ignored-test invocation under target/conformance/<sha>.  A failing
test command still leaves its manifest and combined log behind for review.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'x86_64-pc-windows-msvc'

function Get-CommandOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$File,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    $output = & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$File $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
    }
    return @($output)
}

if (-not $IsWindows) {
    throw 'GPU conformance is a Windows/MSVC release gate; run this script on Windows.'
}

if ($env:CARGO_BUILD_TARGET -and $env:CARGO_BUILD_TARGET -ne $target) {
    throw "CARGO_BUILD_TARGET must be empty or $target; found '$env:CARGO_BUILD_TARGET'."
}

$rustcVerbose = Get-CommandOutput rustc '-vV'
$rustcHost = ($rustcVerbose | Where-Object { $_ -match '^host:' } | Select-Object -First 1)
if ($rustcHost -notmatch "^host:\s+$([regex]::Escape($target))$") {
    throw "rustc host must be $target; found '$rustcHost'."
}

$cargoVersion = (Get-CommandOutput cargo '--version') -join [Environment]::NewLine
$gitStatus = @(Get-CommandOutput git 'status' '--porcelain' '--untracked-files=all')
if ($gitStatus.Count -ne 0) {
    throw "conformance requires a clean worktree, including untracked files:`n$($gitStatus -join [Environment]::NewLine)"
}

$commit = ((Get-CommandOutput git 'rev-parse' 'HEAD') -join '').Trim()
if ($commit -notmatch '^[0-9a-f]{40}$') {
    throw "HEAD must resolve to a 40-character lowercase SHA; found '$commit'."
}

# Nothing above writes an artifact: a rejected checkout must not make itself dirty.
$artifactDirectory = Join-Path $PSScriptRoot "..\target\conformance\$commit"
$artifactDirectory = [IO.Path]::GetFullPath($artifactDirectory)
New-Item -ItemType Directory -Force -Path $artifactDirectory | Out-Null
$logPath = Join-Path $artifactDirectory 'cargo.log'
$manifestPath = Join-Path $artifactDirectory 'manifest.json'
$start = [DateTime]::UtcNow.ToString('o')
$env:FLUXEL_TEST_COMMIT = $commit
$env:RUST_TEST_THREADS = '1'

$gpu = @()
try {
    $gpu = @(Get-CimInstance -ClassName Win32_VideoController | ForEach-Object {
        [ordered]@{ name = $_.Name; driver_version = $_.DriverVersion; pnp_device_id = $_.PNPDeviceID }
    })
}
catch {
    $gpu = @([ordered]@{ collection_error = $_.Exception.Message })
}

# Test-support submit/completion faults are process-global by design.  The
# RUST_TEST_THREADS environment above serializes libtest without passing an
# unsupported --test-threads argument to Criterion benchmark targets.
$cargoArgs = @(
    'test', '--workspace', '--all-targets', '--all-features', '--locked',
    '--target', $target, '--', '--ignored', '--nocapture'
)
$exitCode = 1
try {
    & cargo @cargoArgs 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
}
catch {
    $_ | Out-String | Tee-Object -FilePath $logPath -Append
    $exitCode = 1
}

$caseSummary = @()
$binarySummary = @()
$pendingCase = $null
if (Test-Path -LiteralPath $logPath) {
    foreach ($line in Get-Content -LiteralPath $logPath) {
        if ($line -match '^\s*Running (.+)$') {
            $binarySummary += $Matches[1]
        }
        elseif ($line -match '^test (.+) \.\.\. (ok|FAILED|ignored)$') {
            $caseSummary += [ordered]@{ case = $Matches[1]; result = $Matches[2] }
            $pendingCase = $null
        }
        elseif ($line -match '^test (\S+) \.\.\.') {
            # With --nocapture, fixture evidence may follow the test name and
            # libtest emits the result on a later line once that output ends.
            $pendingCase = $Matches[1]
        }
        elseif ($null -ne $pendingCase -and $line -match '^(ok|FAILED|ignored)$') {
            $caseSummary += [ordered]@{ case = $pendingCase; result = $Matches[1] }
            $pendingCase = $null
        }
    }
}

$manifest = [ordered]@{
    schema = 1
    commit = $commit
    started_at_utc = $start
    finished_at_utc = [DateTime]::UtcNow.ToString('o')
    platform = [ordered]@{ os = 'Windows'; rustc_host = $target; cargo_target = $target }
    tools = [ordered]@{
        rustc_verbose = $rustcVerbose
        cargo_version = $cargoVersion
    }
    environment = [ordered]@{
        FLUXEL_TEST_COMMIT = $env:FLUXEL_TEST_COMMIT
        RUST_TEST_THREADS = $env:RUST_TEST_THREADS
    }
    hardware = $gpu
    command = [ordered]@{
        executable = 'cargo'
        arguments = $cargoArgs
        display = "cargo $($cargoArgs -join ' ')"
    }
    result = [ordered]@{
        exit_code = $exitCode
        log = 'cargo.log'
        test_binaries = $binarySummary
        test_cases = $caseSummary
    }
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8

if ($exitCode -ne 0) {
    [Console]::Error.WriteLine(
        "GPU conformance failed (exit $exitCode). Evidence: $artifactDirectory"
    )
    exit $exitCode
}

Write-Host "GPU conformance passed. Evidence: $artifactDirectory"
