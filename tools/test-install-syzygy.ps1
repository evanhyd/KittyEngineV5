#requires -Version 5.1
$ErrorActionPreference = 'Stop'
$scratchRoot = [System.IO.Path]::GetFullPath("$PSScriptRoot\..\DiagnosticsScratch")
$scratch = Join-Path $scratchRoot ("installer-test-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $scratch | Out-Null

function Assert-True($Condition, $Message) {
  if (!$Condition) { throw $Message }
}

# Test the actual installer with isolated manifests and a local fake transport.
# No network requests or tablebase downloads are made by this test script.
$testState = @{
  requests = 0
  failuresLeft = 2
  transport = 'copy'
  payload = [System.Text.Encoding]::ASCII.GetBytes('installer test data')
}
function Invoke-WebRequest {
  param($Uri, $OutFile, [switch]$UseBasicParsing, [int]$TimeoutSec)
  ++$testState.requests
  if ($testState.transport -eq 'offline') { throw 'Unexpected network request' }
  if ($testState.failuresLeft -gt 0) {
    --$testState.failuresLeft
    throw 'Simulated network interruption'
  }
  Assert-True ($Uri -match '/3-4-5-(wdl|dtz)/K[QRBNP]*vK[QRBNP]*\.rtb[wz]$') 'Invalid download URI'
  $bytes = if ($testState.transport -eq 'corrupt') { [byte[]]::new($testState.payload.Length) } else { $testState.payload }
  [System.IO.File]::WriteAllBytes($OutFile, $bytes)
}
function Start-Sleep { param([int]$Seconds) }

try {
  $installer = Join-Path $scratch 'install-syzygy.ps1'
  $manifestPath = Join-Path $scratch 'syzygy-3-4-5.json'
  Copy-Item -LiteralPath "$PSScriptRoot\install-syzygy.ps1" -Destination $installer
  $manifest = Get-Content -Raw "$PSScriptRoot\syzygy-3-4-5.json" | ConvertFrom-Json
  Assert-True ($manifest.files.Count -eq 290) 'Production manifest must contain 290 files'
  Assert-True (($manifest.files | Measure-Object bytes -Sum).Sum -eq 983957920) 'Unexpected production data size'
  $payloadPath = Join-Path $scratch 'payload'
  [System.IO.File]::WriteAllBytes($payloadPath, $testState.payload)
  $hash = (Get-FileHash -LiteralPath $payloadPath -Algorithm SHA256).Hash
  foreach ($file in $manifest.files) {
    $file.bytes = $testState.payload.Length
    $file.sha256 = $hash
  }
  $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
  $destination = Join-Path $scratch 'Tables with spaces [test]'
  New-Item -ItemType Directory -Path $destination | Out-Null
  $target = Join-Path $destination $manifest.files[0].name
  [System.IO.File]::WriteAllText("$target.part", 'interrupted download')
  [System.IO.File]::WriteAllText((Join-Path $destination 'notes.txt'), 'keep this file')

  $output = & $installer -Destination $destination 3>&1 6>&1
  Assert-True ($testState.requests -eq 292) 'Fresh install must retry twice and fetch every file'
  Assert-True (@(Get-ChildItem -LiteralPath $destination -Filter '*.rtb?').Count -eq 290) 'Missing installed files'
  Assert-True (!(Test-Path -LiteralPath "$target.part")) 'Verified partial file was not promoted'
  Assert-True ($output -contains "setoption name SyzygyPath value $destination") 'Missing usable UCI path'
  foreach ($file in $manifest.files) {
    Assert-True ((Get-FileHash -LiteralPath (Join-Path $destination $file.name)).Hash -eq $hash) 'Installed corrupt data'
  }
  Write-Output 'PASS: complete installation, bounded retries, interrupted file, path with spaces and brackets'

  $timestamp = (Get-Item -LiteralPath $target).LastWriteTimeUtc
  $testState.transport = 'offline'
  $output = & $installer -Destination $destination 3>&1 6>&1
  Assert-True ($testState.requests -eq 292) 'Verified rerun accessed the network'
  Assert-True ((Get-Item -LiteralPath $target).LastWriteTimeUtc -eq $timestamp) 'Verified file was rewritten'
  Write-Output 'PASS: verified rerun works offline without rewriting files'

  [System.IO.File]::WriteAllText($target, 'existing damaged file')
  $testState.transport = 'corrupt'
  $failed = $false
  try { $output = & $installer -Destination $destination 3>&1 6>&1 }
  catch { $failed = $_.Exception.Message -like '*Checksum or size mismatch*' }
  Assert-True $failed 'Corrupt download was accepted'
  Assert-True ($testState.requests -eq 295) 'Failure did not stop after three attempts'
  Assert-True ([System.IO.File]::ReadAllText($target) -eq 'existing damaged file') 'Failed download replaced the old file'
  Write-Output 'PASS: checksum failure preserves the existing target and stops after three attempts'

  $testState.transport = 'copy'
  $output = & $installer -Destination $destination 3>&1 6>&1
  Assert-True ($testState.requests -eq 296) 'Repair should download only the damaged file'
  Assert-True ((Get-FileHash -LiteralPath $target).Hash -eq $hash) 'Repair did not restore the file'
  Assert-True ([System.IO.File]::ReadAllText((Join-Path $destination 'notes.txt')) -eq 'keep this file') 'Unrelated file changed'
  Write-Output 'PASS: rerun repairs damaged data and preserves unrelated files'

  $manifest.files[0].name = '../escape.rtbw'
  $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
  $failed = $false
  try { $output = & $installer -Destination $destination 3>&1 6>&1 }
  catch { $failed = $_.Exception.Message -like '*Invalid entry*' }
  Assert-True $failed 'Unsafe manifest filename was accepted'
  Assert-True ($testState.requests -eq 296) 'Invalid manifest reached the transport'
  Write-Output 'PASS: unsafe manifest is rejected before any download'
} finally {
  $resolved = (Resolve-Path -LiteralPath $scratch).Path
  if (!$resolved.StartsWith($scratchRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Test cleanup path escaped DiagnosticsScratch'
  }
  Remove-Item -LiteralPath $resolved -Recurse -Force
}
