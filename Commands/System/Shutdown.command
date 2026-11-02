#!/bin/bash
# 파이 끄기 (전원 뽑기 전에)
source "$(dirname "$0")/../_common.sh"

y=$(ask "파이를 끌까요? [y/N] ")
[ "$y" = y ] && $A shutdown
finish
