# shellcheck shell=bash
# dmg_helpers.sh — retrying wrappers around hdiutil for CI (source, don't run).
#
# GitHub's macOS runners intermittently answer `hdiutil create|attach|detach`
# with "Resource busy" (Spotlight/XProtect scanning a fresh image, or an
# engine process still holding a file on the mounted volume). v0.2.2 and
# v0.3.0 lost their engine bundles to exactly that: `hdiutil create` and the
# post-pytest `hdiutil detach` (exit 16) failed once and took the release
# with them. Every hdiutil call in ci.yml goes through these functions.
#
#   source scripts/dmg_helpers.sh
#   dmg_quiet_spotlight
#   dmg_create <srcfolder> <out.dmg> <volname>
#   MOUNT=$(dmg_attach <image.dmg>)
#   dmg_detach "$MOUNT"            # non-zero only if even `detach -force` fails
#
# Tunables: DMG_ATTEMPTS (default 5), DMG_BACKOFF_SEC (first delay, default 3;
# doubles each retry).

# Spotlight indexing a just-created or just-mounted image is the usual
# "Resource busy" culprit. Best effort: the runner has passwordless sudo.
dmg_quiet_spotlight() {
  sudo -n mdutil -a -i off >/dev/null 2>&1 || true
}

# _dmg_retry <cmd...> — run with exponential backoff.
_dmg_retry() {
  local n=1 max="${DMG_ATTEMPTS:-5}" delay="${DMG_BACKOFF_SEC:-3}" rc
  while :; do
    rc=0
    "$@" || rc=$?
    [ "$rc" -eq 0 ] && return 0
    if [ "$n" -ge "$max" ]; then
      echo "::error::'$*' failed after $n attempts (exit $rc)" >&2
      return "$rc"
    fi
    echo "::warning::'$*' failed (exit $rc), attempt $n/$max; retrying in ${delay}s" >&2
    sleep "$delay"
    n=$((n + 1))
    delay=$((delay * 2))
  done
}

dmg_create() {
  local src="$1" out="$2" vol="$3"
  # -ov replaces a partial image a failed attempt may have left behind.
  _dmg_retry hdiutil create -volname "$vol" -srcfolder "$src" -ov -format UDZO "$out"
}

# Prints the mount point on stdout; diagnostics go to stderr.
_dmg_attach_once() {
  local out mount
  out=$(hdiutil attach "$1" -nobrowse -readonly) || return $?
  mount=$(printf '%s\n' "$out" | tail -1 | awk -F'\t' '{print $NF}' \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
  if [ -z "$mount" ] || [ ! -d "$mount" ]; then
    echo "hdiutil attach reported no usable mount point:" >&2
    printf '%s\n' "$out" >&2
    return 1
  fi
  printf '%s\n' "$mount"
}

dmg_attach() {
  _dmg_retry _dmg_attach_once "$1"
}

# Stop whatever still holds a file on the volume (an engine the smoke test
# spawned, a pytest child). Only this user's processes can be signalled; a
# system scanner is left to `detach -force`.
_dmg_kill_holders() {
  local mount="$1" pids pid
  # With a mount point as its argument, lsof lists every open file on that
  # file system.
  pids=$(lsof -t -- "$mount" 2>/dev/null | sort -u | grep -vx "$$" || true)
  [ -n "$pids" ] || return 0
  echo "processes still using $mount:" >&2
  # shellcheck disable=SC2086
  ps -o pid=,command= -p $(echo $pids | tr ' ' ',') >&2 || true
  for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in $pids; do kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true; done
}

_dmg_detach_once() {
  local mount="$1"
  [ -d "$mount" ] || return 0   # already gone
  _dmg_kill_holders "$mount"
  hdiutil detach "$mount"
}

dmg_detach() {
  local mount="${1:-}"
  [ -n "$mount" ] || return 0
  [ -d "$mount" ] || return 0
  if DMG_ATTEMPTS="${DMG_ATTEMPTS:-4}" _dmg_retry _dmg_detach_once "$mount"; then
    return 0
  fi
  echo "::warning::falling back to 'hdiutil detach -force $mount'" >&2
  _dmg_kill_holders "$mount"
  hdiutil detach -force "$mount" || { [ ! -d "$mount" ] && return 0; return 1; }
}
