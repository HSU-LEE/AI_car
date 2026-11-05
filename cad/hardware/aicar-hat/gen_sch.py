"""Write aicar-hat.kicad_sch from design.py. Nets are global labels on each pin."""
import uuid
from pathlib import Path

from design import GROUPS, PARTS

HERE = Path(__file__).parent
ROOT = str(uuid.uuid5(uuid.NAMESPACE_DNS, "aicar-hat"))
G = 1.27


def uid(s):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, "aicar-hat/" + s))


def pin_layout(pins):
    """Left column gets the first half of the pins, right column the rest, as (name, x, y, angle)."""
    names = sorted(pins, key=int)
    half = (len(names) + 1) // 2
    rows = max(half, len(names) - half)
    w = 7.62
    out = []
    for i, p in enumerate(names):
        col, row = (0, i) if i < half else (1, len(names) - 1 - i)
        y = (rows - 1) * G - row * 2 * G
        x = -w - 2.54 if col == 0 else w + 2.54
        out.append((p, x, y, 0 if col == 0 else 180))
    return out, w, (rows - 1) * G + 2 * G


def symbol_def(ref, value, pins):
    layout, w, h = pin_layout(pins)
    s = [f'(symbol "aicar:{ref}" (in_bom yes) (on_board yes)',
         f'(property "Reference" "{ref}" (at 0 {h + 1.27} 0) (effects (font (size 1.27 1.27))))',
         f'(property "Value" "{value}" (at 0 {-h - 1.27} 0) (effects (font (size 1.27 1.27))))',
         '(property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))',
         '(property "Datasheet" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))',
         f'(symbol "{ref}_0_1" (rectangle (start {-w} {h}) (end {w} {-h}) (stroke (width 0.2) (type default)) '
         '(fill (type background))))',
         f'(symbol "{ref}_1_1"']
    for p, x, y, a in layout:
        s.append(f'(pin passive line (at {x} {y} {a}) (length 2.54) (name "{p}" (effects (font (size 1 1)))) '
                 f'(number "{p}" (effects (font (size 1 1)))))')
    s.append('))')
    return ' '.join(s)


def main():
    defs, body = [], []
    x0, y0, cell_w, col_x, y = 40.0, 40.0, 75.0, 40.0, 40.0
    page_h = 400
    for title, refs in GROUPS:
        need = 14 + sum(pin_layout(PARTS[r][3])[2] * 2 + 9 for r in refs)
        if y + need > page_h:
            y, col_x = y0, col_x + cell_w
        body.append(f'(text "{title}" (at {col_x - 12} {y} 0) (effects (font (size 2 2) bold) (justify left)) '
                    f'(uuid {uid("t" + title)}))')
        y += 8
        for ref in refs:
            value, fp, lcsc, pins = PARTS[ref]
            defs.append(symbol_def(ref, value, pins))
            layout, w, h = pin_layout(pins)
            y += h + 2
            X, Y = round(col_x / G) * G + 12.7, round(y / G) * G
            body.append(
                f'(symbol (lib_id "aicar:{ref}") (at {X:.2f} {Y:.2f} 0) (unit 1) (in_bom yes) (on_board yes) '
                f'(uuid {uid(ref)}) '
                f'(property "Reference" "{ref}" (at {X:.2f} {Y - h - 1.27:.2f} 0) (effects (font (size 1.27 1.27)))) '
                f'(property "Value" "{value}" (at {X:.2f} {Y + h + 1.27:.2f} 0) (effects (font (size 1.27 1.27)))) '
                f'(property "Footprint" "{fp}" (at {X:.2f} {Y:.2f} 0) (effects (font (size 1.27 1.27)) hide)) '
                f'(property "Datasheet" "" (at {X:.2f} {Y:.2f} 0) (effects (font (size 1.27 1.27)) hide)) '
                f'(property "LCSC" "{lcsc}" (at {X:.2f} {Y:.2f} 0) (effects (font (size 1.27 1.27)) hide)) '
                + ' '.join(f'(pin "{p}" (uuid {uid(ref + "/" + p)}))' for p in pins) +
                f' (instances (project "aicar-hat" (path "/{ROOT}" (reference "{ref}") (unit 1)))))')
            for p, px, py, a in layout:
                ax, ay = X + px, Y - py
                net = pins[p]
                if not net:
                    body.append(f'(no_connect (at {ax:.2f} {ay:.2f}) (uuid {uid("nc" + ref + p)}))')
                    continue
                angle, just = (180, "right") if a == 0 else (0, "left")
                body.append(f'(global_label "{net}" (shape passive) (at {ax:.2f} {ay:.2f} {angle}) '
                            f'(effects (font (size 1.27 1.27)) (justify {just})) (uuid {uid("l" + ref + p)}))')
            y += h + 5
        y += 6
    text = (f'(kicad_sch (version 20231120) (generator "aicar_gen") (generator_version "1") '
            f'(uuid {ROOT}) (paper "A0") '
            f'(title_block (title "AICAR Raspberry Pi 4 HAT") (rev "A")) '
            f'(lib_symbols {" ".join(defs)}) {" ".join(body)} '
            f'(sheet_instances (path "/" (page "1"))))\n')
    (HERE / "aicar-hat.kicad_sch").write_text(text)
    print("parts", len(PARTS), "nets", len({n for p in PARTS.values() for n in p[3].values() if n}))


if __name__ == "__main__":
    main()
