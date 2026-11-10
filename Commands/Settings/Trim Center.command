#!/bin/bash
# Shift the saved servo center by a number of degrees.
source "$(dirname "$0")/../_common.sh"

d=$(ask "Degrees to shift (e.g. -0.5): ")
$A trim "$d"
finish
