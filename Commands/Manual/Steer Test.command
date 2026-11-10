#!/bin/bash
# Test a steer angle, then return to center after one second.
source "$(dirname "$0")/../_common.sh"

a=$(ask "Angle (-26 to 26, negative = left): ")
$A steer "$a"
finish
