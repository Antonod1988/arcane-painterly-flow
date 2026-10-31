#!/usr/bin/env bash
# rt.sh <blender-script-basename> <tag> [args...]  -> renders via the GPU lock, prints timing, writes work/<tag>/cmp.png
B=$1; T=$2; shift 2
cd "$(dirname "$0")"
python gpu_run.py ../blender/$B -- tag=$T "$@" 2>&1 | grep -E "^GPU lock|^Time:|^blender exit|Error|Traceback|line [0-9]+" | grep -v "Not freed"
