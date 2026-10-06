#!/usr/bin/env bash
# assemble_release_assets.sh — collect every platform bundle under its canonical
# name and write SHA256SUMS. Shared by the GitHub Release and S3 publish jobs.
#
#   scripts/assemble_release_assets.sh <version> <bundles-dir> <assets-dir>
#
# <bundles-dir> is where actions/download-artifact put the CI artifacts (one
# subdirectory per artifact); <assets-dir> receives:
#   kenaz-ml-<version>-darwin-arm64.dmg
#   kenaz-ml-<version>-darwin-x86_64.dmg
#   kenaz-ml-<version>-linux-x86_64.zip
#   kenaz-ml-<version>-linux-arm64.zip
#   kenaz-ml-<version>-windows-x86_64.zip
#   SHA256SUMS
# Every bundle is REQUIRED: a release is all platforms or none.
set -euo pipefail
VERSION="$1"; BUNDLES="$2"; ASSETS="$3"
mkdir -p "$ASSETS"
# Linux/Windows zips are canonically named by the freeze job; the zip inside
# the artifact directory already carries the version the freeze computed.
find "$BUNDLES" -name 'kenaz-ml-*.zip' -exec cp {} "$ASSETS/" \;
for arch in arm64 x86_64; do
  dmg="$BUNDLES/kenaz-ml-macos-${arch}-notarized/kenaz-ml.dmg"
  if [ -f "$dmg" ]; then
    cp "$dmg" "$ASSETS/kenaz-ml-${VERSION}-darwin-${arch}.dmg"
  else
    echo "::error::no notarized .dmg for darwin-${arch} — the notarize job is the release gate for macOS"
    exit 1
  fi
done
for t in linux-x86_64 linux-arm64 windows-x86_64; do
  [ -f "$ASSETS/kenaz-ml-${VERSION}-${t}.zip" ] || { echo "::error::missing bundle for ${t} (looked for kenaz-ml-${VERSION}-${t}.zip)"; exit 1; }
done
(cd "$ASSETS" && sha256sum -- * > SHA256SUMS)
ls -l "$ASSETS"
cat "$ASSETS/SHA256SUMS"
