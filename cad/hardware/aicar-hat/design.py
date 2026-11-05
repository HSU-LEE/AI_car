"""AICAR HAT single source of truth: parts and nets.

gen_sch.py and gen_pcb.py both read this, so the schematic and board cannot disagree.
Part format: ref -> (value, footprint, lcsc, {pin: net}).
"""

R0603 = "Resistor_SMD:R_0603_1608Metric"
C0603 = "Capacitor_SMD:C_0603_1608Metric"
C1206 = "Capacitor_SMD:C_1206_3216Metric"
ELEC = "Capacitor_SMD:CP_Elec_8x10"
SO8 = "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"
SO8EP = "Package_SO:SOIC-8-1EP_3.9x4.9mm_P1.27mm_EP2.41x3.3mm"
TSOT6 = "Package_TO_SOT_SMD:TSOT-23-6"
SOT23 = "Package_TO_SOT_SMD:SOT-23"
SMB = "Diode_SMD:D_SMB"
SMA = "Diode_SMD:D_SMA"
WIRE_THIN = "Connector_Wire:SolderWire-0.5sqmm_1x02_P4.6mm_D0.9mm_OD2.1mm"
LED = "LED_SMD:LED_0603_1608Metric"
IND = "Inductor_SMD:L_APV_APH0650"
WIRE = "Connector_Wire:SolderWire-1sqmm_1x02_P5.4mm_D1.4mm_OD2.7mm"
XH2 = "Connector_JST:JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical"
PIN3 = "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical"
PIN4 = "Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical"
TP = "TestPoint:TestPoint_Pad_D1.5mm"

# Passive part numbers (0603; each number checked against its LCSC page).
RES = {"220": "C22962", "1k": "C21190", "2.2k": "C4190", "10k": "C25804", "8.2k": "C25981", "13k": "C22797",
       "15k": "C22809", "100k": "C25803", "110k": "C25805", "150k": "C22807",
       "470k": "C23178", "1M": "C22935"}
CAP = {"100pF": "C14858", "100nF": "C1591", "1uF": "C5199872"}
C22U = "C7432785"   # 22uF 25V X5R 1206.
C470U = "C72517"    # 470uF 16V 8x10 electrolytic.
TVS12 = "C113957"   # SMAJ12A.
TVS5 = "C113974"   # SMBJ5.0A.
FUSE = "C2924995"   # 7A 32V 1206 fast.

PARTS = {}


def add(ref, value, fp, lcsc, pins):
    PARTS[ref] = (value, fp, lcsc, {str(k): v for k, v in pins.items()})


def r(ref, value, a, b):
    add(ref, value, R0603, RES[value], {1: a, 2: b})


def c(ref, value, a, b):
    add(ref, value, C0603, CAP[value], {1: a, 2: b})


def c22(ref, a):
    add(ref, "22uF 25V", C1206, C22U, {1: a, 2: "GND"})


def elec(ref, a):
    add(ref, "470uF 16V", ELEC, C470U, {1: a, 2: "GND"})


# --- Raspberry Pi 40-pin socket (mounted on the bottom side). -----------------
GPIO = {1: "3V3", 17: "3V3", 2: "5V_PI", 4: "5V_PI", 3: "SDA", 5: "SCL",
        11: "GPIO17", 12: "GPIO18", 16: "GPIO23", 18: "GPIO24", 33: "GPIO13", 40: "GPIO21"}
for n in (6, 9, 14, 20, 25, 30, 34, 39):
    GPIO[n] = "GND"
add("J1", "Pi 2x20 socket", "Connector_PinSocket_2.54mm:PinSocket_2x20_P2.54mm_Vertical", "",
    {n: GPIO.get(n, "") for n in range(1, 41)})

# --- Battery inputs: JST-XH, TVS, ideal diode, voltage divider. ------------
for i, (jr, dr, ur, qr) in enumerate((("J2", "D1", "U1", "Q1"), ("J3", "D2", "U2", "Q2")), 1):
    bat, gate = f"BAT{i}", f"G{i}"
    add(jr, f"Pack {i} JST-XH", XH2, "", {1: bat, 2: "GND"})
    add(dr, "SMAJ12A", SMA, TVS12, {1: bat, 2: "GND"})
    add(ur, "LM5050MKX-1", TSOT6, "C473393",
        {1: bat, 2: "GND", 3: "GND", 4: bat, 5: gate, 6: "VOR"})
    add(qr, "AO4406A", SO8, "C2944312",
        {1: bat, 2: bat, 3: bat, 4: gate, 5: "VOR", 6: "VOR", 7: "VOR", 8: "VOR"})
    r(f"R{i}", "470k", bat, f"ADC{i}")
    r(f"R{i + 2}", "150k", f"ADC{i}", "GND")
    c(f"C{i}", "100nF", f"ADC{i}", "GND")

# --- Power switch and undervoltage cutoff: off near 6.0V, on above 6.8V. -------
add("Q3", "AO4407A", SO8, "C2841482",
    {1: "VOR", 2: "VOR", 3: "VOR", 4: "Q3G", 5: "VRAW", 6: "VRAW", 7: "VRAW", 8: "VRAW"})
add("U3", "TL431AIDBZR", SOT23, "C23892", {1: "UVK", 2: "UVREF", 3: "SWG"})
r("R5", "2.2k", "VOR", "UVK")
r("R6", "10k", "UVK", "Q3G")     # 10 ms gate ramp with C3: soft start, rides out hot-swap dips.
c("C3", "1uF", "Q3G", "VOR")
r("R7", "13k", "VOR", "UVREF")
r("R8", "8.2k", "UVREF", "SWG")
r("R9", "100k", "VRAW", "UVREF")
c("C4", "100nF", "UVREF", "SWG")
add("SW1", "Power switch", "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical", "",
    {1: "SWG", 2: "GND", 3: "GND"})     # Middle pin is the common; the off-side throw just stays at GND.

# --- Fuse and main bus. --------------------------------------------------------
add("F1", "7A 32V", "Fuse:Fuse_1206_3216Metric", FUSE, {1: "VRAW", 2: "VSW"})
add("D3", "SMAJ12A", SMA, TVS12, {1: "VSW", 2: "GND"})
elec("C5", "VSW")
elec("C6", "VSW")

# --- 5V bucks: one for the Pi, one for the servo. ------------------------------
for tag, ic, out, l, parts in (
        ("A", "U4", "5V_PI", "L1", ("C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18")),
        ("B", "U5", "5V_SRV", "L2", ("C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28"))):
    rs = {"A": ("R10", "R11", "R12", "R13"), "B": ("R14", "R15", "R16", "R17")}[tag]
    add(ic, "SY8205FCC", SO8EP, "C111875",
        {1: f"BS{tag}", 2: f"LX{tag}", 3: f"EN{tag}", 4: f"SS{tag}", 5: f"FB{tag}", 6: f"VCC{tag}",
         7: "VSW", 8: "VSW", 9: "GND"})
    add(l, "3.3uH 8A", IND, "C49073675", {1: f"LX{tag}", 2: out})
    r(rs[0], "100k", "VSW", f"EN{tag}")
    r(rs[1], "1M", f"EN{tag}", "GND")
    r(rs[2], "110k", out, f"FB{tag}")   # Vout = 0.6 * (1 + 110k / 15k) = 5.0V.
    r(rs[3], "15k", f"FB{tag}", "GND")
    c22(parts[0], "VSW")
    c22(parts[1], "VSW")
    c22(parts[2], out)
    c22(parts[3], out)
    c22(parts[4], out)
    c(parts[5], "100nF", f"BS{tag}", f"LX{tag}")
    c(parts[6], "1uF", f"VCC{tag}", "GND")
    c(parts[7], "100nF", f"SS{tag}", "GND")
    c(parts[8], "100pF", out, f"FB{tag}")
add("D4", "SMBJ5.0A", SMB, TVS5, {1: "5V_PI", 2: "GND"})    # Blows F1 if the Pi buck shorts high.
elec("C7", "5V_SRV")

# --- Motor driver: channel A only. Unused channel B inputs tied low. ------------
TB = {1: "MOT1", 2: "MOT1", 5: "MOT2", 6: "MOT2", 3: "GND", 4: "GND", 9: "GND", 10: "GND",
      7: "", 8: "", 11: "", 12: "", 13: "VSW", 14: "VSW", 24: "VSW", 15: "GND", 16: "GND", 17: "GND",
      18: "GND", 19: "3V3", 20: "3V3", 21: "GPIO23", 22: "GPIO24", 23: "GPIO13"}
add("U6", "TB6612FNG", "Package_SO:SSOP-24_5.3x8.2mm_P0.65mm", "C141517", TB)
c22("C30", "VSW")
c("C31", "100nF", "VSW", "GND")
c("C32", "100nF", "3V3", "GND")
add("J4", "Motor wires", WIRE_THIN, "", {1: "MOT1", 2: "MOT2"})

# --- Battery ADC: 0x48 (ADDR to GND). -------------------------------------------
add("U7", "ADS1115IDGSR", "Package_SO:MSOP-10_3x3mm_P0.5mm", "C37593",
    {1: "GND", 2: "", 3: "GND", 4: "ADC1", 5: "ADC2", 6: "GND", 7: "GND", 8: "3V3", 9: "SDA", 10: "SCL"})
c("C33", "100nF", "3V3", "GND")

# --- Servo. ----------------------------------------------------------------------
r("R20", "220", "GPIO18", "SERVO")
add("J5", "Servo", "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical", "",
    {1: "GND", 2: "5V_SRV", 3: "SERVO"})

# --- Sharp GP2Y0A21: red 5V, black GND, yellow Vo. --------------------------------
r("R21", "1k", "SHARP", "GPIO17")
r("R22", "100k", "GPIO17", "GND")   # Reads clear when the sensor is unplugged.
c22("C34", "5V_PI")
add("J6", "Sharp", PIN3, "",
    {1: "5V_PI", 2: "GND", 3: "SHARP"})

# --- VL53L0X: VIN must stay 3.3V because the module pulls I2C up to VIN. ------------
c("C35", "100nF", "3V3", "GND")
add("J7", "VL53L0X", PIN4, "",
    {1: "3V3", 2: "GND", 3: "SCL", 4: "SDA"})

# --- Status LEDs. --------------------------------------------------------------------
r("R23", "2.2k", "5V_PI", "LEDP")
add("D5", "Red PWR", LED, "C2286", {1: "GND", 2: "LEDP"})
r("R24", "1k", "GPIO21", "LEDS")
add("D6", "Red STATUS", LED, "C2286", {1: "GND", 2: "LEDS"})

# --- Test points. ---------------------------------------------------------------------
for ref, net in (("TP1", "VOR"), ("TP2", "VSW"), ("TP3", "5V_PI"), ("TP4", "5V_SRV"), ("TP5", "GND")):
    add(ref, net, TP, "", {1: net})

GROUPS = [
    ("Pi header", ["J1"]),
    ("Pack 1", ["J2", "D1", "Q1", "U1", "R1", "R3", "C1"]),
    ("Pack 2", ["J3", "D2", "Q2", "U2", "R2", "R4", "C2"]),
    ("Switch and undervoltage cutoff", ["Q3", "U3", "R5", "R6", "C3", "R7", "R8", "R9", "C4", "SW1"]),
    ("Fuse and bus", ["F1", "D3", "C5", "C6"]),
    ("Pi 5V buck", ["U4", "L1", "R10", "R11", "R12", "R13", "C10", "C11", "C12", "C13", "C14",
                    "C15", "C16", "C17", "C18", "D4"]),
    ("Servo 5V buck", ["U5", "L2", "R14", "R15", "R16", "R17", "C20", "C21", "C22", "C23", "C24",
                       "C25", "C26", "C27", "C28", "C7"]),
    ("Motor", ["U6", "C30", "C31", "C32", "J4"]),
    ("Battery ADC", ["U7", "C33"]),
    ("Servo", ["R20", "J5"]),
    ("Sharp", ["R21", "R22", "C34", "J6"]),
    ("VL53L0X", ["C35", "J7"]),
    ("LEDs", ["R23", "D5", "R24", "D6"]),
    ("Test points", ["TP1", "TP2", "TP3", "TP4", "TP5"]),
]
