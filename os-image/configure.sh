#!/bin/bash
# Runs inside the Lima VM as root: writes hotspot/user/hardware config into the mounted image at /mnt/pi
set -euo pipefail
source "$(dirname "$0")/secrets.env"
R=/mnt/pi; B=$R/boot/firmware

# --- hotspot (NetworkManager) ---
cat > $R/etc/NetworkManager/system-connections/AICAR-AP.nmconnection <<EOF
[connection]
id=AICAR-AP
uuid=$(cat /proc/sys/kernel/random/uuid)
type=wifi
interface-name=wlan0
autoconnect=true
autoconnect-priority=100

[wifi]
mode=ap
ssid=${APSSID:-04}
band=bg
channel=6

[wifi-security]
key-mgmt=wpa-psk
proto=rsn
pairwise=ccmp
group=ccmp
pmf=1
psk=$APPW

[ipv4]
method=shared
address1=10.42.0.1/24

[ipv6]
method=disabled
EOF
chmod 600 $R/etc/NetworkManager/system-connections/AICAR-AP.nmconnection

# --- first-boot config via cloud-init ---
cat > $B/user-data <<EOF
#cloud-config
hostname: aicar
manage_etc_hosts: true
timezone: Asia/Seoul
ssh_pwauth: true
users:
- name: pi
  gecos: AICAR
  groups: [adm, dialout, cdrom, audio, users, sudo, video, plugdev, input, gpio, spi, i2c, netdev, render]
  shell: /bin/bash
  sudo: ALL=(ALL) NOPASSWD:ALL
  lock_passwd: false
  passwd: $HASH
runcmd:
- [raspi-config, nonint, do_wifi_country, KR]
- [rfkill, unblock, wifi]
- [systemctl, enable, --now, ssh]
- [nmcli, connection, up, AICAR-AP]
EOF
touch $B/ssh

# --- Wi-Fi regulatory domain, I2C/SPI ---
grep -q ieee80211_regdom $B/cmdline.txt || sed -i '1 s/$/ cfg80211.ieee80211_regdom=KR/' $B/cmdline.txt
grep -q '^dtparam=i2c_arm=on' $B/config.txt || printf '\n# AICAR\ndtparam=i2c_arm=on\ndtparam=spi=on\n' >> $B/config.txt
grep -q '^i2c-dev' $R/etc/modules || echo i2c-dev >> $R/etc/modules

chroot $R systemctl enable ssh avahi-daemon NetworkManager >/dev/null 2>&1

# --- cleanup ---
rm -f $R/usr/sbin/policy-rc.d $R/etc/ssh/ssh_host_*
: > $R/etc/resolv.conf
chroot $R apt-get clean
rm -rf $R/root/.cache $R/tmp/* $R/var/lib/apt/lists/*
echo configured
