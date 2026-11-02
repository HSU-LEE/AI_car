#!/bin/bash
# Delete old recordings on the Pi. Asks before deleting.
source "$(dirname "$0")/../_common.sh"

k=$(ask "최근 몇 개를 남길까요 (엔터 = 5): ")
$A clean-runs $k
finish
