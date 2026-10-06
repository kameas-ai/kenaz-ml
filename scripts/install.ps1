# install.ps1 — install the kenaz-ml engine on Windows as a self-contained binary.
#
#   irm https://raw.githubusercontent.com/kameas-ai/kenaz-ml/main/scripts/install.ps1 | iex
#
# No Python, pip or uv on the machine: downloads the frozen bundle for
# Windows x86_64 from the GitHub Release, verifies it against the release's
# SHA256SUMS, unpacks it under %LOCALAPPDATA%\kenaz-ml and adds a `kenaz-ml`
# shim directory to the user's PATH. Re-running upgrades in place.
#
# Options (environment):
#   KENAZ_ML_VERSION   install this version instead of the latest release
#   KENAZ_ML_HOME      install root (default: %LOCALAPPDATA%\kenaz-ml)
#   KENAZ_ML_REPO      GitHub repository (default: kameas-ai/kenaz-ml)
#   KENAZ_ML_BASE_URL  fetch <url>/<asset> instead of the GitHub Release (mirrors, tests)
$ErrorActionPreference = 'Stop'

$Repo = if ($env:KENAZ_ML_REPO) { $env:KENAZ_ML_REPO } else { 'kameas-ai/kenaz-ml' }
$Home_ = if ($env:KENAZ_ML_HOME) { $env:KENAZ_ML_HOME } else { Join-Path $env:LOCALAPPDATA 'kenaz-ml' }

$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -ne 'AMD64') { throw "kenaz-ml: unsupported CPU $arch (Windows builds are x86_64 only)" }
$target = 'windows-x86_64'

if ($env:KENAZ_ML_VERSION) {
  $tag = 'v' + $env:KENAZ_ML_VERSION.TrimStart('v')
} else {
  $latest = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ 'User-Agent' = 'kenaz-ml-install' }
  $tag = $latest.tag_name
  if (-not $tag) { throw "kenaz-ml: could not determine the latest release of $Repo" }
}
$version = $tag.TrimStart('v')
$base = if ($env:KENAZ_ML_BASE_URL) { $env:KENAZ_ML_BASE_URL } else { "https://github.com/$Repo/releases/download/$tag" }
$asset = "kenaz-ml-$version-$target.zip"

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("kenaz-ml-" + [System.Guid]::NewGuid().ToString('n'))
New-Item -ItemType Directory -Path $work | Out-Null
try {
  Write-Host "kenaz-ml: downloading $asset"
  Invoke-WebRequest -Uri "$base/$asset" -OutFile (Join-Path $work $asset)
  Invoke-WebRequest -Uri "$base/SHA256SUMS" -OutFile (Join-Path $work 'SHA256SUMS')
  $line = Get-Content (Join-Path $work 'SHA256SUMS') | Where-Object { $_ -match "\s$([regex]::Escape($asset))$" } | Select-Object -First 1
  if (-not $line) { throw "kenaz-ml: $asset is not listed in SHA256SUMS" }
  $expected = ($line -split '\s+')[0].ToLower()
  $actual = (Get-FileHash -Algorithm SHA256 (Join-Path $work $asset)).Hash.ToLower()
  if ($actual -ne $expected) { throw "kenaz-ml: sha256 mismatch for ${asset}: got $actual, expected $expected" }
  Write-Host "kenaz-ml: verified sha256:$actual"

  $dest = Join-Path $Home_ "versions\$version"
  $partial = "$dest.partial"
  if (Test-Path $partial) { Remove-Item -Recurse -Force $partial }
  New-Item -ItemType Directory -Path $partial | Out-Null
  Expand-Archive -Path (Join-Path $work $asset) -DestinationPath $partial
  if (-not (Test-Path "$partial\kameas-ml\kameas-ml.exe")) { throw "kenaz-ml: the archive has no $partial\kameas-ml\kameas-ml.exe" }
  if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
  Move-Item $partial $dest

  # current -> versions\<version> (a junction needs no elevation), and a shim on PATH.
  $current = Join-Path $Home_ 'current'
  if (Test-Path $current) { (Get-Item $current).Delete() }
  New-Item -ItemType Junction -Path $current -Target $dest | Out-Null
  $binDir = Join-Path $Home_ 'bin'
  New-Item -ItemType Directory -Force -Path $binDir | Out-Null
  $shim = Join-Path $binDir 'kenaz-ml.cmd'
  Set-Content -Path $shim -Value "@echo off`r`n`"$current\kameas-ml\kameas-ml.exe`" %*`r`n" -Encoding ASCII
  $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
  if (-not ($userPath -split ';' | Where-Object { $_ -eq $binDir })) {
    [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $binDir), 'User')
    Write-Host "kenaz-ml: added $binDir to your user PATH (open a new terminal to pick it up)"
  }
  & "$current\kameas-ml\kameas-ml.exe" --version | Out-Null
  Write-Host "kenaz-ml: installed kenaz-ml $version to $dest"
  Write-Host 'kenaz-ml: run:  kenaz-ml serve'
} finally {
  Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
