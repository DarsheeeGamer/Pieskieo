$ErrorActionPreference = "Stop"

# Pieskieo installer (Windows, PowerShell 5+)
# Builds from source and installs binaries into $env:ProgramData\Pieskieo\bin (fallback: $HOME\.local\bin).

param(
    [string]$Prefix
)

function Usage {
@"
Pieskieo installer

Parameters:
  -Prefix <dir>   Installation prefix (default: ProgramData\Pieskieo or $HOME\.local)
"@
}

if ($PSBoundParameters.ContainsKey("Help")) { Usage; exit 0 }

if (-not $Prefix) {
    if (Test-Path "$env:ProgramData") {
        $Prefix = Join-Path $env:ProgramData "Pieskieo"
    } else {
        $Prefix = Join-Path $HOME ".local"
    }
}

$binDst = Join-Path $Prefix "bin"
New-Item -Force -ItemType Directory -Path $binDst | Out-Null

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "Rust toolchain (cargo) is required. Install via https://rustup.rs/ then rerun."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

Write-Host "Building Pieskieo in release mode..."
# Prefer self-contained GNU build to avoid MSVC requirement.
$env:RUSTFLAGS = "-Clink-self-contained=yes"
cargo +stable-x86_64-pc-windows-gnu build --release --locked

$binSrc = Join-Path $repoRoot "target/release"
$bins = @("pieskieo-server.exe", "pieskieo.exe", "load.exe", "bench.exe")
foreach ($b in $bins) {
    $src = Join-Path $binSrc $b
    if (Test-Path $src) {
        Copy-Item $src $binDst -Force
        Write-Host "Installed $b -> $binDst"
    }
}

Write-Host "Done. Ensure $binDst is on your PATH."
