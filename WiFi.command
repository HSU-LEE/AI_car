#!/bin/bash
# Save a new Wi-Fi network on the Pi USB adapter.
source "$(dirname "$0")/Commands/_common.sh"

$A wifi
echo
$A wifi-add
finish
