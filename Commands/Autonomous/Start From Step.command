#!/bin/bash
# Start autonomous driving at course step N.
source "$(dirname "$0")/../_common.sh"

k=$(ask "몇 번 동작부터: ")
$A go --start "$k"
finish
