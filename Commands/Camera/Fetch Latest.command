#!/bin/bash
# Copy the latest recording into the Mac runs/ folder.
source "$(dirname "$0")/../_common.sh"

$A fetch latest
finish
