#!/usr/bin/env bash
# publish_release_if_complete.sh — flip a draft GitHub Release to published
# once every asset is on it, and not a moment before.
#
#   GH_TOKEN=... GITHUB_REPOSITORY=owner/repo scripts/publish_release_if_complete.sh <tag>
#
# Releases are created as drafts (tag-on-merge.yml, release.yml, ci.yml's
# release-assets job). Drafts are invisible to `releases/latest`, which
# scripts/install.sh and install.ps1 resolve, so an install never lands on a
# release whose engine bundles are missing (v0.2.2 and v0.3.0 shipped that
# way). Two workflows add assets to the same release — ci.yml (engine bundles
# + SHA256SUMS) and release.yml (sdist + wheel) — and both call this script
# after uploading; whichever finishes last publishes. Idempotent: a published
# release is left alone, an incomplete one stays a draft and exits 0.
#
# The full set (scripts/assemble_release_assets.sh names the bundles):
#   kenaz-ml-<v>-darwin-arm64.dmg   kenaz-ml-<v>-darwin-x86_64.dmg
#   kenaz-ml-<v>-linux-x86_64.zip   kenaz-ml-<v>-linux-arm64.zip
#   kenaz-ml-<v>-windows-x86_64.zip SHA256SUMS
#   one *.whl and one *.tar.gz (PEP 440 may normalise <v>, so matched by suffix)
set -euo pipefail
TAG="${1:?usage: publish_release_if_complete.sh <tag>}"
REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"
V="${TAG#v}"

# `gh release view <tag>` finds drafts (it falls back to listing releases);
# the REST `releases/tags/{tag}` endpoint alone would not.
INFO=$(gh release view "$TAG" -R "$REPO" --json isDraft,assets)
if [ "$(jq -r .isDraft <<<"$INFO")" != "true" ]; then
  echo "Release $TAG is already published; nothing to do."
  exit 0
fi
NAMES=$(jq -r '.assets[].name' <<<"$INFO")

missing=()
for f in "kenaz-ml-$V-darwin-arm64.dmg" "kenaz-ml-$V-darwin-x86_64.dmg" \
         "kenaz-ml-$V-linux-x86_64.zip" "kenaz-ml-$V-linux-arm64.zip" \
         "kenaz-ml-$V-windows-x86_64.zip" SHA256SUMS; do
  grep -qxF "$f" <<<"$NAMES" || missing+=("$f")
done
grep -qE '\.whl$' <<<"$NAMES" || missing+=("<wheel>.whl")
grep -qE '\.tar\.gz$' <<<"$NAMES" || missing+=("<sdist>.tar.gz")

if [ "${#missing[@]}" -gt 0 ]; then
  # Expected when the other workflow is still uploading; if it already
  # finished (or failed), this draft is stuck and needs a re-run.
  echo "::warning::Release $TAG stays a DRAFT until the rest of its assets arrive: ${missing[*]}"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    echo "### Release $TAG is still a DRAFT — missing: ${missing[*]}" >> "$GITHUB_STEP_SUMMARY"
  fi
  exit 0
fi

# Only a plain vX.Y.Z at least as high as every PUBLISHED plain release
# becomes "Latest"; a re-run of an older tag or an -rc publishes without
# stealing it. Compared against published releases, not tags, so a higher
# tag stuck as a draft can't stop a completed hotfix from becoming latest.
LATEST=false
if [[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  HIGHEST=$( { gh release list -R "$REPO" --exclude-drafts --exclude-pre-releases --limit 200 \
      --json tagName --jq '.[].tagName' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true; echo "$TAG"; } \
    | sort -V | tail -1)
  [ "$HIGHEST" = "$TAG" ] && LATEST=true
fi
PRERELEASE=false
[[ "$TAG" == *-* ]] && PRERELEASE=true

gh release edit "$TAG" -R "$REPO" --draft=false --latest="$LATEST" --prerelease="$PRERELEASE"
echo "Published $TAG (latest=$LATEST, prerelease=$PRERELEASE) with every asset present."
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  echo "### Release $TAG published (latest=$LATEST)" >> "$GITHUB_STEP_SUMMARY"
fi
