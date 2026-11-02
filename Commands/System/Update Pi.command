#!/bin/bash
# Upload code from the Mac pi/ folder to the Pi.
source "$(dirname "$0")/../_common.sh"

$A push
finish
