#!/bin/bash
# Delete a saved Wi-Fi network.
source "$(dirname "$0")/../_common.sh"

$A wifi
echo
w=$(ask "Wi-Fi network to delete: ")
$A wifi-del "$w"
finish
