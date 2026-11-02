#!/bin/bash
# Shut the Pi down before unplugging it.
source "$(dirname "$0")/../_common.sh"

y=$(ask "파이를 끌까요? [y/N] ")
[ "$y" = y ] && $A shutdown
finish
