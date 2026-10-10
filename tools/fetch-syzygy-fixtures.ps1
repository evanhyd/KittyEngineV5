param([string]$Destination = "$PSScriptRoot\..\DiagnosticsScratch\syzygy")
$ErrorActionPreference = 'Stop'
$manifest = Get-Content -Raw "$PSScriptRoot\syzygy-fixtures.json" | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$directory = (Resolve-Path -LiteralPath $Destination).Path
foreach ($file in $manifest.files) {
  if ($file.name -notmatch '^K[QRBNP]*vK[QRBNP]*\.rtb[wz]$') { throw 'Invalid fixture filename' }
  $target = Join-Path $directory $file.name
  if ((Test-Path -LiteralPath $target) -and
      (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $file.sha256) { continue }
  $uri = "https://raw.githubusercontent.com/niklasf/python-chess/$($manifest.revision)/$($manifest.directory)/$($file.name)"
  Invoke-WebRequest -Uri $uri -OutFile $target
  if ((Get-Item -LiteralPath $target).Length -ne $file.bytes -or
      (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $file.sha256) {
    throw "Fixture checksum mismatch: $($file.name)"
  }
}
Write-Output "Verified $($manifest.files.Count) fixture files in $directory"
