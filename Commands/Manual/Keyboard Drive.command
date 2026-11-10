#!/bin/bash
# Keyboard drive: w/s speed, a/d steer, space stops, q quits.
source "$(dirname "$0")/../_common.sh"

s=$(ask "Maximum speed (Enter = 0.6): ")
$A drive-keys $s
finish
