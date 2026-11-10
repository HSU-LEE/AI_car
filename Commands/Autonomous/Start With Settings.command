#!/bin/bash
# One autonomous run with temporary settings, for example KP=1.8 KH=1.0.
source "$(dirname "$0")/../_common.sh"

v=$(ask "KEY=value (separate several with spaces): ")
$A go --set $v
finish
