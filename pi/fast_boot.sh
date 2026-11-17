#!/bin/bash
# Faster Pi boot for AICAR, and no background jobs that wake up during a drive. Run on the Pi: bash fast_boot.sh
# (aicar fast-boot does it). Undo: bash fast_boot.sh undo
#
#   NetworkManager-wait-online  6 s waiting for "online": nothing here needs it (ssh and the hotspot come up anyway)
#   cloud-init                  first-boot setup (user, hostname, ssh), done long ago, run again at every boot (2.5 s, before ssh)
#   bluetooth, disable-bt       not used. The UART is left free as well
#   e2scrub_reap                cleans up LVM snapshots: this card has no LVM
#   apt-daily, apt-daily-upgrade, man-db timers
#                               package downloads and index rebuilds that start by themselves, possibly in the middle of a drive
#   disable_splash              the rainbow screen at power-on
set -euo pipefail

UNITS=(NetworkManager-wait-online.service e2scrub_reap.service bluetooth.service apt-daily.timer apt-daily-upgrade.timer man-db.timer)
CONFIG=/boot/firmware/config.txt
MARK="# Faster boot (aicar)"

if [ "${1:-}" = undo ]; then
    sudo systemctl enable "${UNITS[@]}" >/dev/null 2>&1 || true
    sudo rm -f /etc/cloud/cloud-init.disabled
    sudo sed -i "/^$MARK/,+2d" "$CONFIG"
    echo "Undone. Takes effect at the next boot (sudo reboot)."
    exit 0
fi

[ -f "$CONFIG.bak-boot" ] || sudo cp "$CONFIG" "$CONFIG.bak-boot"
sudo systemctl disable --now "${UNITS[@]}" >/dev/null 2>&1 || true
sudo touch /etc/cloud/cloud-init.disabled
if ! grep -q "^$MARK" "$CONFIG"; then
    printf '\n%s: no rainbow screen, no Bluetooth (the UART stays free)\ndisable_splash=1\ndtoverlay=disable-bt\n' "$MARK" | sudo tee -a "$CONFIG" >/dev/null
fi

echo "Done (previous config.txt: $CONFIG.bak-boot):"
for u in "${UNITS[@]}"; do printf '  %-36s %s\n' "$u" "$(systemctl is-enabled "$u" 2>/dev/null || true)"; done
echo "  cloud-init                           $([ -f /etc/cloud/cloud-init.disabled ] && echo disabled || echo enabled)"
echo "Takes full effect at the next boot (sudo reboot). Undo: bash fast_boot.sh undo"
