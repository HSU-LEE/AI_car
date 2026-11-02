#!/bin/bash
# Tune the traffic light. In the browser: w/s saturation, e/d brightness, r/f size, Enter saves.
source "$(dirname "$0")/../_common.sh"

$A tune-light
finish
