#!/bin/bash
# Write JLCPCB manufacturing files into fab/: Gerbers and drills (zip), BOM, placement.
set -e
cd "$(dirname "$0")"
CLI=/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli
KIPY=/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3
rm -rf fab
mkdir -p fab/gerber
$CLI pcb export gerbers --layers F.Cu,B.Cu,F.Paste,B.Paste,F.SilkS,B.SilkS,F.Mask,B.Mask,Edge.Cuts \
    --no-x2 --use-drill-file-origin -o fab/gerber aicar-hat.kicad_pcb >/dev/null 2>&1
$CLI pcb export drill --format excellon --drill-origin plot --excellon-separate-th \
    --generate-map --map-format gerberx2 -o fab/gerber/ aicar-hat.kicad_pcb >/dev/null 2>&1
(cd fab/gerber && zip -q ../aicar-hat-gerber.zip * -x '*-drl_map.gbr' '*.gbrjob')
$CLI pcb export pos --format csv --units mm --side both --exclude-dnp --use-drill-file-origin -o fab/pos.csv aicar-hat.kicad_pcb >/dev/null 2>&1
$CLI sch export pdf -o fab/aicar-hat-schematic.pdf aicar-hat.kicad_sch >/dev/null 2>&1
python3 make_bom.py
rm fab/pos.csv
$KIPY orientation_check.py aicar-hat.kicad_pcb fab/orientation-check.csv
$CLI pcb render --side top --width 1600 --height 1400 -o fab/preview-top.png aicar-hat.kicad_pcb >/dev/null 2>&1
$CLI pcb render --side bottom --width 1600 --height 1400 -o fab/preview-bottom.png aicar-hat.kicad_pcb >/dev/null 2>&1
cp ORDER-GUIDE.md fab/
(cd fab && zip -q aicar-hat-order-package.zip aicar-hat-gerber.zip bom-jlcpcb.csv cpl-jlcpcb.csv hand-solder.csv \
    orientation-check.csv aicar-hat-schematic.pdf preview-top.png preview-bottom.png ORDER-GUIDE.md)
ls -la fab
