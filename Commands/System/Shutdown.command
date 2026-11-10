#!/bin/bash
# Shut the Pi down before unplugging it.
source "$(dirname "$0")/../_common.sh"

y=$(ask "Shut down the Pi? [y/N] ")
[ "$y" = y ] && $A shutdown
finish
