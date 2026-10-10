param(
  [switch]$EnableSyzygy,
  [ValidateSet('search', 'tablebase')][string]$Benchmark = 'search',
  [string]$SourceDirectory = "$PSScriptRoot\..\KittyEngineV5",
  [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$Name = 'plain',
  [switch]$Assembly
)
$ErrorActionPreference = 'Stop'
if ($Benchmark -eq 'tablebase' -and !$EnableSyzygy) { throw 'The tablebase benchmark requires -EnableSyzygy' }
$repo = (Resolve-Path "$PSScriptRoot\..").Path
$source = (Resolve-Path $SourceDirectory).Path
$out = Join-Path $repo "DiagnosticsScratch\bench\$Name"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'Visual Studio C++ tools are required' }
if (!(Get-Command cl -ErrorAction SilentlyContinue)) {
  & "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
}
$config = if ($EnableSyzygy) { "$source\config\syzygy" } else { $source }
$driver = Join-Path $PSScriptRoot ($Benchmark + '_benchmark.cpp')
$files = @($driver, "$source\boardstate.cpp", "$source\notation.cpp")
if ($EnableSyzygy) { $files += @("$source\tablebase.cpp", "$repo\third_party\fathom\tbprobe.c") }
$flags = @('/nologo','/std:c++20','/Zc:__cplusplus','/EHsc','/O2','/Ob2','/MD','/arch:AVX2','/fp:fast','/DNDEBUG',
  '/D_CRT_SECURE_NO_WARNINGS','/D_SILENCE_CXX20_ATOMIC_INIT_DEPRECATION_WARNING',
  '/DTB_NO_HELPER_API',"/I$config","/I$source","/I$repo\third_party\fathom",'/TP',"/Fo$out\")
if ($Assembly) {
  & cl @flags /c /FAs "/Fa$out\$Benchmark.asm" $driver
} else {
  & cl @flags /GL @files "/Fe$out\bench.exe" /link /LTCG /STACK:16777216 "/MAP:$out\bench.map"
}
if ($LASTEXITCODE -ne 0) { throw 'Benchmark compilation failed' }
Write-Output $out
