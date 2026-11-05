#!/bin/bash
# Rebuild the board, autoroute signals with Freerouting, fill zones and run DRC.
# Usage: ./route.sh <path to freerouting jar>
set -eo pipefail
cd "$(dirname "$0")"
if [ -f MANUAL_LAYOUT ]; then echo "Hand layout in progress: this script would overwrite it."; exit 1; fi
JAR=$1
K=/Applications/KiCad/KiCad.app/Contents
PY=$K/Frameworks/Python.framework/Versions/Current/bin/python3
CLI=$K/MacOS/kicad-cli
JAVA=/opt/homebrew/opt/openjdk/bin/java
mkdir -p out
$PY gen_pcb.py 2>&1 | { grep -v Fontconfig || true; }
$PY pcb_tool.py export aicar-hat.kicad_pcb out/board.dsn 2>&1 | { grep -v Fontconfig || true; }
"$JAVA" -jar "$JAR" -de out/board.dsn -do out/board.ses -mp 40 --gui.enabled=false > out/route.log 2>&1 || true
tail -3 out/route.log | cut -c1-160
$PY pcb_tool.py import aicar-hat.kicad_pcb out/board.ses 2>&1 | { grep -v Fontconfig || true; }
$PY pcb_tool.py repair aicar-hat.kicad_pcb 2>&1 | { grep -v Fontconfig || true; }
$CLI pcb drc --schematic-parity --severity-error --format json -o out/drc.json aicar-hat.kicad_pcb 2>/dev/null | tail -1 || true
$PY pcb_tool.py repair aicar-hat.kicad_pcb out/drc.json 2>&1 | { grep -v Fontconfig || true; }
$CLI pcb drc --schematic-parity --severity-error --format json -o out/drc.json aicar-hat.kicad_pcb 2>/dev/null | tail -1 || true
python3 pcb_report.py
