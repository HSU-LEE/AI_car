#!/bin/bash
# Pi status: address, Wi-Fi, temperature, disk, and running programs.
source "$(dirname "$0")/../_common.sh"

$A status
finish
