#!/usr/bin/env sh
# List every Mach-O file under a directory, one path per line, detected by
# content rather than by name.
#
# The frozen onedir carries Mach-O files that no extension filter catches:
# PyInstaller ships the interpreter as _internal/Python.framework, whose
# binary is just `Python`, and the launcher itself has no suffix. Apple's
# notary service rejects a submission over a single unsigned Mach-O, so the
# signing and verification steps both enumerate through this script.
#
#   usage: scripts/list_macho.sh <dir>
set -eu
dir=${1:?usage: list_macho.sh <dir>}
[ -d "$dir" ] || { echo "list_macho.sh: not a directory: $dir" >&2; exit 2; }
# `file --mime-type` prints "<path>: <type>", column-aligned across each batch
# unless told not to pad (-N; the padding is what hid 484 of 487 Mach-O files
# on the first try). Anchoring on the type at end of line keeps a colon inside
# a path from splitting the line.
find "$dir" -type f -print0 \
  | xargs -0 file -N --mime-type -- \
  | sed -n 's/: *application\/x-mach-binary$//p'
