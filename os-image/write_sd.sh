#!/bin/bash
# Write the AICAR image to a 128 GB SD card. Stops unless the disk is removable and 128 GB.
set -euo pipefail
DISK=$(diskutil list physical | awk '/^\/dev\/disk/ {d=$1} /\*128\.0 GB/ {print d; exit}' | sed "s#/dev/##")
[ -n "$DISK" ] || { echo "Stopped: no 128 GB SD card found"; exit 1; }
echo "Target disk: $DISK"
IMG="$(cd "$(dirname "$0")" && pwd)/aicar-2026-10-06-lite.img.gz"
INFO=$(diskutil info /dev/$DISK)
echo "$INFO" | grep -q "Removable Media:.*Removable" || { echo "Stopped: $DISK is not a removable disk"; exit 1; }
echo "$INFO" | grep -q "128.0 GB" || { echo "Stopped: $DISK is not 128 GB"; exit 1; }
diskutil unmountDisk /dev/$DISK
echo "Writing the SD card... (5-10 min, enter the Mac password)"
gunzip -c "$IMG" | sudo dd of=/dev/r$DISK bs=4m status=progress
sync
diskutil eject /dev/$DISK
echo "Done. Take out the SD card and put it in the Raspberry Pi."
