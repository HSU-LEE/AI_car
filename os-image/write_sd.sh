#!/bin/bash
# Write the AICAR image to a 128 GB SD card. Stops unless the disk is removable and 128 GB.
set -euo pipefail
DISK=$(diskutil list physical | awk '/^\/dev\/disk/ {d=$1} /\*128\.0 GB/ {print d; exit}' | sed "s#/dev/##")
[ -n "$DISK" ] || { echo "중단: 128GB SD 카드를 찾지 못했습니다"; exit 1; }
echo "대상 디스크: $DISK"
IMG="$(cd "$(dirname "$0")" && pwd)/aicar-2026-10-06-lite.img.gz"
INFO=$(diskutil info /dev/$DISK)
echo "$INFO" | grep -q "Removable Media:.*Removable" || { echo "중단: $DISK 가 이동식 디스크가 아닙니다"; exit 1; }
echo "$INFO" | grep -q "128.0 GB" || { echo "중단: $DISK 크기가 128GB가 아닙니다"; exit 1; }
diskutil unmountDisk /dev/$DISK
echo "SD 카드에 쓰는 중... (5~10분 소요, 맥 비밀번호를 입력하세요)"
gunzip -c "$IMG" | sudo dd of=/dev/r$DISK bs=4m status=progress
sync
diskutil eject /dev/$DISK
echo "완료! SD 카드를 빼서 라즈베리 파이에 꽂으세요."
