#!/bin/bash
# Re-analyze the latest recording on the Mac and write a video.
source "$(dirname "$0")/../_common.sh"

$A replay latest
finish
