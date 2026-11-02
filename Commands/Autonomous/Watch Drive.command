#!/bin/bash
# Follow the log of a drive that is already running. Ctrl+C stops the car.
source "$(dirname "$0")/../_common.sh"

$A log
finish
