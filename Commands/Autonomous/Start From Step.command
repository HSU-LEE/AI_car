#!/bin/bash
# Start autonomous driving at course step N.
source "$(dirname "$0")/../_common.sh"

k=$(ask "Start from step number: ")
$A go --start "$k"
finish
