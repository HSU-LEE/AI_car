#!/bin/bash
# Drive straight for a chosen speed and time.
source "$(dirname "$0")/../_common.sh"

s=$(ask "Speed (-1 to 1, negative = reverse): ")
t=$(ask "Duration in seconds (Enter = 1): ")
$A drive "$s" $t
finish
