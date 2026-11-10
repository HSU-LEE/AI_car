#!/bin/bash
# Delete old recordings on the Pi. Asks before deleting.
source "$(dirname "$0")/../_common.sh"

k=$(ask "Number of recent recordings to keep (Enter = 5): ")
$A clean-runs $k
finish
