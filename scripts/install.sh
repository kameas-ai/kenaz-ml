#!/bin/sh
# install.sh — install the kenaz-ml engine as a self-contained binary.
#
#   curl -fsSL https://raw.githubusercontent.com/kameas-ai/kenaz-ml/main/scripts/install.sh | sh
#
# No Python, pip or uv on the machine: downloads the frozen bundle for this
# OS/CPU from the GitHub Release, verifies it against the release's
# SHA256SUMS, unpacks it under the user's data directory and links `kenaz-ml`
# into ~/.local/bin. Re-running upgrades (or re-installs) in place; previous
# versions are kept until you delete them.
#
# Options (environment):
#   KENAZ_ML_VERSION=1.2.3   install this version instead of the latest release
#   KENAZ_ML_HOME=<dir>      install root (default: ~/.kenaz/ml/standalone)
#   KENAZ_ML_BIN_DIR=<dir>   where the `kenaz-ml` link goes (default: ~/.local/bin)
#   KENAZ_ML_REPO=owner/name GitHub repository (default: kameas-ai/kenaz-ml)
#   KENAZ_ML_BASE_URL=<url>  fetch <url>/<asset> instead of the GitHub Release (mirrors, tests)
set -eu

REPO="${KENAZ_ML_REPO:-kameas-ai/kenaz-ml}"
HOME_DIR="${KENAZ_ML_HOME:-$HOME/.kenaz/ml/standalone}"
BIN_DIR="${KENAZ_ML_BIN_DIR:-$HOME/.local/bin}"

say() { printf 'kenaz-ml: %s\n' "$*"; }
die() { printf 'kenaz-ml: error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required"; }

need curl
need tar

os=$(uname -s)
arch=$(uname -m)
case "$os" in
  Darwin) os=darwin ;;
  Linux)  os=linux ;;
  *) die "unsupported OS: $os (supported: macOS, Linux; Windows uses install.ps1)" ;;
esac
case "$arch" in
  arm64|aarch64) arch=arm64 ;;
  x86_64|amd64)  arch=x86_64 ;;
  *) die "unsupported CPU: $arch" ;;
esac
target="$os-$arch"
if [ "$target" = "darwin-x86_64" ]; then
  die "macOS is built for Apple silicon only (darwin-arm64); run under Rosetta is not supported"
fi

# ---- resolve the version ----------------------------------------------------
if [ -n "${KENAZ_ML_BASE_URL:-}" ] && [ -z "${KENAZ_ML_VERSION:-}" ]; then
  die "KENAZ_ML_BASE_URL needs KENAZ_ML_VERSION as well"
fi
if [ -n "${KENAZ_ML_VERSION:-}" ]; then
  tag="v${KENAZ_ML_VERSION#v}"
else
  tag=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
  [ -n "$tag" ] || die "could not determine the latest release of $REPO"
fi
version="${tag#v}"
base="${KENAZ_ML_BASE_URL:-https://github.com/$REPO/releases/download/$tag}"

case "$os" in
  darwin) asset="kenaz-ml-$version-$target.dmg" ;;
  linux)  asset="kenaz-ml-$version-$target.zip" ;;
esac
if [ "$os" = "linux" ]; then need unzip; fi

# ---- download + verify -------------------------------------------------------
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
say "downloading $asset"
curl -fsSL -o "$work/$asset" "$base/$asset" || die "download failed: $base/$asset"
curl -fsSL -o "$work/SHA256SUMS" "$base/SHA256SUMS" || die "download failed: $base/SHA256SUMS"
expected=$(grep " $asset\$" "$work/SHA256SUMS" | cut -d' ' -f1)
[ -n "$expected" ] || die "$asset is not listed in SHA256SUMS"
if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "$work/$asset" | cut -d' ' -f1)
else
  actual=$(shasum -a 256 "$work/$asset" | cut -d' ' -f1)
fi
[ "$actual" = "$expected" ] || die "sha256 mismatch for $asset: got $actual, expected $expected"
say "verified sha256:$actual"

# ---- unpack the onedir into versions/<version>/ ---------------------------------
dest="$HOME_DIR/versions/$version"
rm -rf "$dest.partial"
mkdir -p "$dest.partial"
case "$os" in
  darwin)
    mount=$(hdiutil attach "$work/$asset" -nobrowse -readonly -mountrandom "$work" | tail -1 | awk '{print $NF}')
    [ -d "$mount/kameas-ml" ] || { hdiutil detach "$mount" >/dev/null 2>&1 || true; die "no onedir at $mount/kameas-ml"; }
    cp -R "$mount/kameas-ml" "$dest.partial/kameas-ml"
    hdiutil detach "$mount" >/dev/null
    ;;
  linux)
    unzip -q "$work/$asset" -d "$dest.partial"
    [ -d "$dest.partial/kameas-ml" ] || die "no onedir at $dest.partial/kameas-ml"
    chmod +x "$dest.partial/kameas-ml/kameas-ml"
    ;;
esac
rm -rf "$dest"
mv "$dest.partial" "$dest"

# ---- make it the current version and put it on PATH -----------------------
ln -sfn "versions/$version" "$HOME_DIR/current"
mkdir -p "$BIN_DIR"
ln -sfn "$HOME_DIR/current/kameas-ml/kameas-ml" "$BIN_DIR/kenaz-ml"

"$BIN_DIR/kenaz-ml" --version >/dev/null 2>&1 || die "the installed engine does not run ($BIN_DIR/kenaz-ml --version failed)"
say "installed kenaz-ml $version to $dest"
say "linked $BIN_DIR/kenaz-ml"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) say "note: $BIN_DIR is not on your PATH; add it, e.g.  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
say "run:  kenaz-ml serve"
