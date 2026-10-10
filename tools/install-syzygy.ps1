#requires -Version 5.1
param([string]$Destination = "$PSScriptRoot\..\Tablebases\syzygy")
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$manifest = Get-Content -Raw "$PSScriptRoot\syzygy-3-4-5.json" | ConvertFrom-Json

# Validate all names before touching the destination. The manifest pins the
# complete standard 3-5 piece WDL+DTZ set; it is not fetched during installation.
if ($manifest.files.Count -ne 290) { throw 'Expected 290 files in the 3-5 piece manifest' }
if ($manifest.base_url -ne 'https://tablebase.lichess.ovh/tables/standard') { throw 'Unexpected tablebase source' }
$names = @{}
foreach ($file in $manifest.files) {
  if ($file.name -cnotmatch '^K[QRBNP]*vK[QRBNP]*\.rtb[wz]$' -or
      $file.name.Split('.')[0].Length -notin 4, 5, 6 -or
      $file.sha256 -notmatch '^[0-9a-f]{64}$' -or $file.bytes -le 0 -or $names.ContainsKey($file.name)) {
    throw 'Invalid entry in the 3-5 piece manifest'
  }
  $names[$file.name] = $true
}
foreach ($name in $names.Keys) {
  $stem = $name.Split('.')[0]
  if (!$names.ContainsKey("$stem.rtbw") -or !$names.ContainsKey("$stem.rtbz")) {
    throw "Missing WDL/DTZ pair for $stem"
  }
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$directory = (Resolve-Path -LiteralPath $Destination).Path
if ($directory.Contains(';')) { throw 'SyzygyPath uses semicolons to separate directories; choose a path without them' }

function Test-TableFile($Path, $File) {
  return (Test-Path -LiteralPath $Path -PathType Leaf) -and
    (Get-Item -LiteralPath $Path).Length -eq $File.bytes -and
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $File.sha256
}

$totalBytes = ($manifest.files | Measure-Object -Property bytes -Sum).Sum
$fileCount = $manifest.files.Count
Write-Host ("Installing $fileCount Syzygy files ({0:N1} MiB) in {1}" -f ($totalBytes / 1MB), $directory)
$verified = 0
$downloaded = 0
foreach ($file in $manifest.files) {
  ++$verified
  $target = Join-Path $directory $file.name
  if (Test-TableFile $target $file) { continue }
  $kind = if ($file.name.EndsWith('.rtbw')) { 'wdl' } else { 'dtz' }
  $uri = "$($manifest.base_url)/3-4-5-$kind/$($file.name)"
  $partial = "$target.part"
  Write-Host "[$verified/$fileCount] Downloading $($file.name)"
  for ($attempt = 1; $attempt -le 3; ++$attempt) {
    try {
      # A rerun restarts an unfinished file and skips every verified file.
      # Only a complete, verified download may replace the final filename.
      Invoke-WebRequest -UseBasicParsing -Uri $uri -OutFile $partial -TimeoutSec 180
      if (!(Test-TableFile $partial $file)) { throw "Checksum or size mismatch for $($file.name)" }
      break
    } catch {
      if ($attempt -eq 3) {
        throw "Could not install $($file.name): $($_.Exception.Message). Rerun the same command to continue."
      }
      Write-Warning "Attempt $attempt failed for $($file.name): $($_.Exception.Message). Retrying."
      Start-Sleep -Seconds 2
    }
  }
  Move-Item -LiteralPath $partial -Destination $target -Force
  ++$downloaded
}

Write-Host "Verified $verified files; downloaded $downloaded, reused $($verified - $downloaded)."
Write-Host 'Set SyzygyPath in your chess GUI, or send these UCI commands to an enabled engine:'
Write-Output "setoption name SyzygyPath value $directory"
Write-Output 'setoption name SyzygyProbeLimit value 5'
Write-Output 'isready'
