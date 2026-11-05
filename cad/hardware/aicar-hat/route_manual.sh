#!/bin/bash
# Route the hand-placed board on a work copy, starting from the unrouted backup. Usage: route_manual.sh <freerouting jar> [passes]
set -euo pipefail
cd "$(dirname "$0")"
JAR=$1; PASSES=${2:-40}
K=/Applications/KiCad/KiCad.app/Contents
PY=$K/Frameworks/Python.framework/Versions/Current/bin/python3
CLI=$K/MacOS/kicad-cli
JAVA=/opt/homebrew/opt/openjdk/bin/java
W=out/rt
mkdir -p $W
FR_HOME="$(pwd)/$W/freerouting-home"
mkdir -p "$FR_HOME"
cp out/before-routing.kicad_pcb $W/r.kicad_pcb
cp aicar-hat.kicad_pro $W/r.kicad_pro
$PY route_board.py power $W/r.kicad_pcb > $W/log 2>&1 || { echo "power failed"; tail -3 $W/log; exit 1; }
$PY route_board.py pretie $W/r.kicad_pcb >> $W/log 2>&1 || { echo "pretie failed"; tail -3 $W/log; exit 1; }
$PY route_board.py dsn $W/r.kicad_pcb $W/r.dsn >> $W/log 2>&1 || { echo "dsn failed"; tail -3 $W/log; exit 1; }
rm -f $W/r.ses
"$JAVA" -Duser.home="$FR_HOME" -Djava.awt.headless=true -jar "$JAR" -de $W/r.dsn -do $W/r.ses -mp "$PASSES" --gui.enabled=false > $W/route.log 2>&1
test -s $W/r.ses
$PY pcb_tool.py import $W/r.kicad_pcb $W/r.ses >> $W/log 2>&1 || { echo "import failed"; tail -3 $W/log; exit 1; }
$PY pcb_tool.py repair $W/r.kicad_pcb >> $W/log 2>&1
$CLI pcb drc --severity-error --format json -o $W/drc.json $W/r.kicad_pcb >/dev/null 2>&1
$PY pcb_tool.py repair $W/r.kicad_pcb $W/drc.json >> $W/log 2>&1
$PY pcb_tool.py link $W/r.kicad_pcb >> $W/log 2>&1
$PY standardize_widths.py $W/r.kicad_pcb >> $W/log 2>&1
$CLI pcb drc --severity-error --format json -o $W/drc.json $W/r.kicad_pcb >/dev/null 2>&1
{ grep -E "pad ties|link vias|cut off" $W/log || true; } | cut -c1-200
python3 - <<'P'
import json,collections
d=json.load(open('out/rt/drc.json'))
print('violations',dict(collections.Counter(v['type'] for v in d['violations'])),'unconnected',len(d['unconnected_items']))
for v in d['violations'][:14]: print(' ',v['type'],' / '.join(i['description'][:42]+str(tuple(round(c,1) for c in (i['pos']['x']-100,i['pos']['y']-100))) for i in v['items']))
for u in d['unconnected_items'][:14]: print(' U',' / '.join(i['description'][:36]+str(tuple(round(c,1) for c in (i['pos']['x']-100,i['pos']['y']-100))) for i in u['items']))
P
