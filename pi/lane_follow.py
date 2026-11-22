"""OpenCV lane detection with variable steering and a per-junction course.

The lower part of the frame (the ROI) is split into horizontal bands. Each band finds the white lines, and the left and right lines are tracked from frame to frame to get the lane center. A line fitted through those centers gives
  - position error (how far the car is from the lane center)
  - heading error (how much the lane is curving)
Those two are combined into a continuous steering angle. Larger steering slows the car. A line across the road (a junction) runs the next action listed in COURSE in course.py.

Usage (on the Pi)
  python lane_follow.py             steering only (motors off); push the car by hand to check detection
  python lane_follow.py --drive     drive for real
  python lane_follow.py --drive --speed 0.3
  python lane_follow.py --drive --start 3     start at COURSE action 3
  python lane_follow.py --record    save video under runs/ (analyze on a Mac with replay.py)
  python lane_follow.py --drive --set KP=1.8 KH=1.0   override settings for this run only

  The white-line threshold is calibrated automatically. Set it by hand with tune_white.py (saved in vision.json).

  Open http://<Pi IP>:8000 in a Mac browser to see the detection view. (On the hotspot, 10.42.0.1.)
  Stop with Ctrl+C (SIGINT) or SIGTERM. Driving continues if ssh drops (SIGHUP).
  On the Mac, aicar go/view starts this in the background on the Pi and writes output to drive.log.
"""
import argparse
import ast
import collections
import hashlib
import hmac
import json
import queue
import secrets
import select
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from car import STEER_LIMIT
from course import COURSE, RED_BACK_ON, PERSON_AT, PERSON_MAX_WAIT, PERSON_MIN_STOP, PERSON_ON, PERSON_RESUME_DELAY, PERSON_X
import course as _course
from person import AnyFigure, PERSON_FILE, PHOTO_DIR, PersonDetector, PersonWatch, figure_box
import landmarks
from overlay import Canvas, as_draw
from safe_io import read_json, write_json
PERSON_IN_LANE = getattr(_course, "PERSON_IN_LANE", False)      # Same for this one.
PERSON_LANE_MARGIN = 0.05   # PERSON_IN_LANE: a figure's center may be this far outside a lane line, as a share of the lane width, for one standing on the line.
PERSON_ANY = getattr(_course, "PERSON_ANY", False)      # A course saved with an older run has no such line, and looked only for the registered figure.

# --- Settings ---
FRAME_W, FRAME_H = 640, 360
# The camera uses the full sensor (about 66 degrees). The old 1536x864 mode (about 47 degrees) cropped the center two thirds because FrameRate was 60.
# Both modes share the same image center, so an old point (x, y) is now (320 + (x-320)/1.5, 180 + (y-180)/1.5).
# Pixel and screen-position settings below were converted by that ratio (the old value is noted on each line). Confirm on the track with Detect Only and refine.
CAMERA_SENSOR = (2304, 1296)    # Full sensor, read 2x2 binned (up to 56 fps). For a narrower view, use (1536, 864) at 60 fps.
CAMERA_FPS = 56
HFLIP = VFLIP = False   # Set True if the image looks upside down.
CAMERA_X = 0.5          # Where the car centerline falls across the frame. vision.json camera_x overrides this when present.

LINE_MODE = "lane"      # "lane": drive between both lines. "line": follow a single line.
ROI_TOP = 0.4           # Use only from this fraction of the frame height downward. A smaller value looks farther ahead.
LANE_MODEL = "bands"    # Driving always reads the lane this way: a point per line in each band, a straight line through them.
                        # course.py cannot switch it. "track" (lane_track.py: one lane model fitted to every line pixel up to
                        # near the horizon, tracked frame to frame) is experimental: tests and the simulator set it directly.
LANE_POINTS = getattr(_course, "LANE_POINTS", "bands")   # course.py. How the band method's lane position and heading are worked out (junctions and the rest are always the bands).
                        # "bands": a center point per band, a straight line through them. "pairs": both lines fitted at once as a
                        # perspective pair over thin sample rows (lane_pairs.py).
BAND_VIEW = lambda: LANE_MODEL != "track" and LANE_POINTS != "pairs"   # The band points and path drawn in the view.
JUNCTION_TRACK_ASSIST = getattr(_course, "JUNCTION_TRACK_ASSIST", False)   # course.py. The tracker only removes false junctions and
                        # false lane ends. It never steers and never adds one. Off: detection is exactly as before.
STEER_MODE = "pursuit"  # With LANE_MODEL = "track": "pursuit" steers onto an arc through the lane center LOOK_LW ahead (uses the bend of
                        # the lane); "pd" keeps the position / heading steering (KP, KH). The band method always uses "pd".
LOOK_LW = 0.5           # pursuit: target this many lane widths ahead of the camera. Longer = smoother, shorter = tighter in curves.
LOOK_SPEED_LW = 0.2     # pursuit: plus this much per unit of speed command (faster looks farther).
WHEELBASE_LW = 0.7      # The car's wheelbase in lane widths (rear axle to front axle over the lane width).
CAM_AHEAD_LW = 0.5      # How far the camera sits ahead of the rear axle, in lane widths.
TRACK_CONF_MIN = 0.0    # pursuit: below this tracker confidence, the position / heading steering is used instead. 0: the tracker's held
                        # lane carries through gaps (dashed lines); switching back and forth made the steering jump.
STYLE_LOOK = {"soft": 1.25, "normal": 1.0, "sharp": 0.8}    # set_style(): scales the look-ahead (soft looks farther, sharp nearer).
SLICES = 6              # Number of horizontal bands the ROI is split into.

# White-line detection. The default recalibrates every frame. tune_white.py stores a manual value in vision.json.
LINE_COLOR = "white"    # Track type. "white": white line on a black floor. "black": black line on a white floor.
                        # "black" inverts brightness so the black line is treated as a white line. The rest of the pipeline stays the same.
                        # The track-color menu (aicar track) saves the choice in vision.json.
WHITE_THR = None        # Brightness above the floor that counts as a white line. None means automatic.
WHITE_MIN = 35          # Auto mode never goes below this, so noise on an empty floor is not taken as a line.
TOPHAT_K = 23           # A bright blob thicker than this is not a line (reflection, floor outside the mat), in half-resolution px. [narrow FOV 35]
REFLECT_SHARP = 0.34    # Ceiling-light reflections on a glossy mat look like white lines but have soft, blurred edges. A blob whose edge steepness
                        # (90th percentile edge gradient / its contrast with the floor) is below this is dropped. None turns it off. Tape measured 0.37-0.5, reflections 0.18-0.33.
                        # calibrate.py measures it on the venue (vision.json reflect_sharp overrides this).
REFLECT_Y = 0.4         # Only blobs centered below this fraction of the ROI height are judged. Far tape is out of focus (focus is ~15 cm) and soft too.
REFLECT_MIN_AREA = 60   # Smaller blobs are not judged.
REFLECT_LINE_H = 0.6    # A blob at least this tall (fraction of the ROI height) is a lane line running up the view, not judged. Reflections are short.
# Motion test (works under any lighting): paint on the floor comes toward the car as fast as the car drives. A ceiling light seen in
# a glossy floor is a mirror image as far below the floor as the light is above it, so it barely moves (about 1/20 of the floor).
GLARE_WINDOW = 0.4      # Seconds back to the frame a blob is compared with. The car must have been driving forward all that time.
GLARE_PAINT_CM = 2.5    # Came this many cm or more closer in that time: paint (kept even if its edges are soft, such as a far crosswalk).
GLARE_STILL_CM = 1.0    # Came less than this closer: a reflection (dropped even if its edges are crisp).
GLARE_STEER_MAX = 12    # The test runs only while steering less than this. A turn sweeps everything sideways.
GLARE_MATCH = 0.6       # How well the blob must match its old self (normalized correlation) to be judged.
GLARE_MARGIN = 0.1      # The other answer must match at least this much worse, or there is no verdict.
GLARE_SIDE_PX = 60      # Sideways search in px. A piece of a long slanted line coming closer looks like it slides along itself.
GLARE_REACH_CM = 40     # Paint cannot have come closer than this in GLARE_WINDOW (100 cm/s), so the match looks no farther back.
GLARE_STILL_DX = 8      # "Stayed put" allows only this much sideways in px, so the next stripe of a crosswalk cannot pass for the same one standing still.
GLARE_SHARP_MAX = 0.4   # A blob this crisp or crisper is never dropped by the motion test: tape (0.37-0.5), not a reflection (up to about 0.33).
GLARE_KEEP = 3.0        # A spot judged a reflection stays one for this long, also once the car stops (the reflection stays put when it does).
REFLECT_FALLBACK_MAX = 0.36  # With no motion verdict (standing still, turning), the edge test cuts at most this. Tape and paint measured 0.37 and up.
USE_MAT_MASK = True     # Ignore the floor and wooden border outside the black mat.
MAT_MIN_CONTRAST = 70   # Below this brightness gap between the mat and the outside, there is no separate mat (tape on one floor), so the whole frame is used.
                        # Wood floor plus black tape differs by about 50 under lighting and vignetting, so splitting would erase the near tape in the bottom corners.
                        # A black mat plus a wood border differs by about 100.
LINE_W_MIN, LINE_W_MAX = 3, 87   # Width of one line inside a band, in px. [narrow FOV 4, 130]
MIN_PIXELS = 40         # Minimum pixels required to accept one line. [narrow FOV 60]
DOUBLE_GAP = 33         # Two lines closer than this gap (px) are merged into one double line. [narrow FOV 50]
CROSSWALK_INSIDE = 3    # With both lane lines known, a crosswalk band needs this many stripes between them.
CROSSWALK_LINES = 4     # This many lines or more in one band means a crosswalk, and that band is ignored.
STRIPE_MAX = 0.3        # Crosswalk stripe: a separate white blob no longer than this times the lane width at its height. Lane lines are long blobs.
                        # Run 20261103-202456: stripes 41x15..84x59 px (about 0.2 lane width), lane lines 97x183 px and longer. Far line bits about 0.38.
STRIPE_MIN_AREA = 120   # Smaller blobs are noise, not stripes.
SPECK_SIDE = 0.25       # A blob in the far half of the ROI, shorter than 1.5 bands, narrower than a horizontal line, centered in the outer SPECK_SIDE of the frame,
                        # and not cut by any ROI edge (a lane line leaving the frame) is not a lane line: a dark object or shadow off the track (run 20261103-210839).
SIGN_RED_SHARE = 0.08   # A white blob with this share or more of red paint around it is a number on a red sign, not a crosswalk stripe.
SIGN_S_MIN, SIGN_V_MIN = 90, 70     # How red a pixel must be to count.
SIGN_MIN_AREA = 400     # A red area on the floor at least this big (px) whose outline holds white is a sign: everything inside it is left out of
SIGN_WHITE_SHARE = 0.1  # the lane lines and the crosswalk test. The white inside must be at least this share of the outline (a sign face, not a red object).
STRIPES_MIN = 3         # This many stripes or more, each inside the lane (lane line on both its left and right), is a crosswalk.
                        # Then they are removed from lane detection, so a stripe is never taken as a lane line.
LANE_WIDTH_PX = 482     # Lane width in px at the bottom of the frame (y=342). Corrected automatically when both lines are visible.
                        # Converted from a measured 590 in a narrow-FOV recording. y=342 on the wide FOV is outside the old frame (closer floor).
HORIZON_Y = 49          # Height y where the two lane lines meet. Sets how quickly the lane narrows toward the top. [narrow FOV -16]
REF_LANE_PX = 590       # Lane width that position and heading error are measured in, from when KP and KH were tuned on the narrow FOV.
                        # Errors are multiplied by REF_LANE_PX / LANE_WIDTH_PX so they keep the old magnitude.
                        # Steering values such as KP, KH, OFFSET_PRIORITY, and the new-lane test stay unchanged.
LEARN_WIDTH = False     # True learns lane width while driving. False keeps the measured value above, so each run behaves the same.
OUTLIER_PX = 30         # A band center farther than this from the fitted line is a misdetect and is dropped. [narrow FOV 45]

KP = 1.6                # Position error to steering. How hard the car returns to the lane center. Larger is more aggressive.
KH = 0.9                # Heading error to steering. How early the car turns into a curve. Raise it if the car leaves the curve.
OFFSET_PRIORITY = 0.6   # Once position error exceeds this, heading is ignored and the car only returns to the lane, so the two do not cancel on an S-curve.
KD = 0.03               # Damping. Raise it a little if the zigzag is fast. Too much causes shake.
SMOOTH = 0.15           # Steering low-pass. 0 is none, 0.8 is very smooth. A larger value reacts late and leaves curves.
AUTO_TRIM = False       # True learns and saves a center trim when straight driving keeps steering to one side.
AUTO_TRIM_TAU = 15.0    # Auto center-trim time constant, in seconds. Learns only on straights where both lines are visible.
AUTO_STEER_MAX = 26     # Maximum steering angle while driving, in degrees. Does not exceed car.py STEER_LIMIT (26).
STEER_RATE = 150        # Maximum change in steering angle per second, in deg/s. A smaller value turns less abruptly.

BASE_SPEED = 0.3        # Speed going straight, when an action sets none.
STEER_SLOWDOWN = False  # True slows the car as steering grows while following the lane (speed_for). Off: the set speed is kept.
MIN_SPEED = 0.3         # Lowest forward speed: at full steering, at the end of slow=, and while the lane is lost.
HARD_STEER = 18         # Above this steering angle (degrees), wheel drag slows the car, so the motor command is raised. Below tr() 21 so right turns get it too.
HARD_STEER_BOOST = 0.15 # Motor command added then. 0 disables it.
JUMP_CONFIRM = 0.3      # A lane position that jumps by more than this from the last frame counts only when the next frame agrees
                        # (one stray point over broken lines flipped the lane for a frame). 0 = off.
JUMP_HOLD_MAX = 2       # At most this many frames in a row are held back.
LOST_HOLD = 1.0         # After the lane is lost, drive on for this many seconds, then back up (LOST_BACK_SEC), then stop.
LOST_CENTER_RATE = 25.0 # While following with the lane lost, the wheels return to straight at this many degrees per second
                        # instead of staying at the last angle (which kept the car turning off the mat). 0 = hold the last angle.
LOST_HOLD_SPEED = 0.95  # LOST_HOLD is for this speed. Slower, the car needs longer for the same distance (a gap in the lines), so it
LOST_HOLD_SCALE_MAX = 3.0   # waits up to this many times as long before backing up. Faster or equal: LOST_HOLD as it is.
LOST_BACK_SEC = 1.5     # Then reverse for up to this long, replaying the recent steering backwards so the car retraces its path, until the lane comes back. 0 = just stop.
LOST_BACK_SPEED = 0.5   # Motor command while backing up. Reverse needs a higher command than forward to move at all.
LOST_BACK_TRIES = 2     # Back-ups allowed per action. Each needs a fresh loss of the lane, so the car cannot rock back and forth forever.
LOST_BACK_MEMORY = 4.0  # Seconds of forward steering kept for the retrace.
RED_LOOK_SEC = 2.0      # wait_green: look for the red light this long, standing still, before backing up to see more.
RED_BACK_SEC = 1.0      # wait_green: back up this long along the recent path. The light may be out of view because the car stopped too close.
RED_BACK_TRIES = 2      # wait_green: back-ups allowed per action. After them it waits out max_sec, as before.
STEP_MAX_SEC = 20       # If an until action sets no max_sec, warn and advance after this many seconds so a step cannot run forever.
LIGHT_VIEW_EVERY = 8    # Not waiting for green: the traffic light is detected for the view only once every this many frames (about 3 times a second).
                        # Display only: driving reads the light every frame while waiting for green. It is the costliest part of a watched frame.
PERSON_EVERY = 2        # The LEGO figure is looked for once every this many frames. Its stop position and delays are in course.py.
FAR_TOP = 0.3           # slow= far detection also looks for a horizontal line between this frame height and FAR_BOT. [narrow FOV 0.2]
FAR_BOT = 0.6           # A far line that is still thin can sit below ROI_TOP before the near (band) test can see it, so the far search reaches down to here.
FAR_HLINE = 0.8         # A far horizontal line counts only if connected white pixels on one row cover at least this times the lane width at that height.
                        # A thin far line shows only its middle in the mask (1.2 missed the T-junction line in runs 20261103-1939/1940 until it was at 0.5).
                        # Used only to slow down, not for the junction decision (until), so a curve mistaken for a line only slows the car briefly.
SLOW_START_CM = 30.0    # Lane-end stops: drive at the set speed until the lane end is this close (cm, as shown), then slow= eases in.
APPROACH_MIN = 0.325      # slow= never slows the car below this, whatever slow= asks. A speed set lower with speed= or set_speed() is kept.
APPROACH_HOLD = 0.5     # slow=: keep slowing for this many seconds after the line disappears, so one or two false frames do not react.
MOTOR_MIN = 0.3         # Lowest forward motor command. Every lower command is raised to this, whatever speed= or slow= asks for.
MOTOR_MIN_TURN = 0.3    # Lowest motor command at full steer, because turned wheels add drag. Values in between scale with the steering angle.
                        # Fixed here. motor_min / motor_min_turn in calib.json on the Pi are no longer read.
LANE_PX_PER_CM = REF_LANE_PX / 20.0     # How many position-error px equal 1 cm, for a 20 cm lane width. Used by shift=.
                                        # Position error is in REF_LANE_PX units, so this uses the same reference.
# style= steering character: (KP scale, KH scale, SMOOTH).
STYLES = {"soft": (0.8, 0.7, 0.35), "normal": (1.0, 1.0, None), "sharp": (1.3, 1.6, 0.05)}

HLINE_LANE_SHARE = 0.4  # With both lane lines known, a horizontal line must cover this share of the lane between them to be a line across.
HLINE_RATIO = 0.27      # A run covering at least this fraction of the band width is a horizontal line (a junction). [narrow FOV 0.4]
JUNCTION_Y = 0.5        # The junction action starts once the horizontal line drops below this frame height. A smaller value starts earlier.
CAMERA_F = 492          # Focal length in px of the full-sensor view (about 66 degrees wide). Turns screen height into floor distance for cm=.
LANE_CM = 20.0          # Real lane width.
END_BASE_CM = 38.0      # cm=0 is where the line sits this far ahead of the camera, about where to_junction() ends without cm=.
END_SPEED_SEC = 0.25    # cm=: the car's speed is measured from how the line moved over the last this many seconds. Short, because the car is still slowing down.
END_LEAD_SEC = 0.05     # cm=: end this much early (at the measured speed). The reading is about 2 frames old by the time it is acted on.
STOP_LEAD = True        # go_to(): count the stop point as reached a little early, by the measured speed: the frames it must hold
                        # in a row and the reading's age would otherwise carry the car past it (about 2-4 cm at 0.7).
STOP_LEAD_MAX_CM = 4.0  # ... never more than this early,
STOP_LEAD_SCORE = 0.85  # ... and only on a clear match (a borderline one may be a look-alike just short of the place).
END_MIN_SPEED = 3.0     # cm=: a measured speed below this (cm/s), on a car that is being driven, is a bad reading. END_DEFAULT_SPEED is used instead.
END_DEFAULT_SPEED = 10.0    # cm=: speed assumed (cm/s) when none could be measured. About what the lowest motor command gives.
END_SWITCH_FRAMES = 3   # cm=: before the end line is reached, this many readings in a row that agree with each other replace a reading they jump away from.
END_FRESH_SEC = 0.15    # cm=: a reading older than this when the line is reached does not count as that line's position.
JUNCTION_REF = "laneend" # to_junction(cm=): what it measures to. "laneend": where the lane's own line ends at the junction, the nearer side.
                        # Every junction has one: an L on its inside, a T and a + on both sides. It lies straight ahead of the car (+-10 cm),
                        # so it stays in view down to the frame bottom from either direction. "line": the line across the lane (before 2026-11-09).
LANEEND_OFFSET_CM = -14.5   # Added to every line-end distance read from the camera: real minus read, in cm. -14.5 makes it the distance from
                        # the car front (2026-11-09: read 19.5 cm, tape 5 cm), since the camera sits about 15 cm behind it.
                        # calibrate.py --end CM measures it on the car and saves it in vision.json (laneend_offset_cm), which wins.
# The settings below are all distances from the car front, as shown on screen (LANE END) and written as cm=.
LANEEND_BASE_CM = 10.0  # to_junction() with no cm=: ends with the line end this far ahead. to_junction(cm=20) ends at 20 cm.
LANEEND_MIN_CM = 4.5    # The nearest it can still be read: the frame bottom is about 2.6 cm, plus room for the 2 frames a reading is old.
LANEEND_MAX_CM = 28.0   # The farthest it can be read and still be confirmed: the top of the view is about 31 cm. A larger cm= would end on first sight.
LANEEND_LINE_GAP = 20.0 # When no line end is found, it is taken this far before the line across (cm), unless the two were seen together and
                        # the real gap measured. Recordings 20261109: 19-22 cm.
LANEEND_FRAMES = 4      # The line end must be found this many frames in a row before it counts.
LANEEND_GAP = 4         # Walking up along the lane line, this many empty rows in a row end it.
LANEEND_TOL = 0.06      # The line is looked for this far either side of where it should be, in lane widths (at least 4 px).
LANEEND_DRIFT = 3.0     # While walking up a lane line, it may bend this many px per row and still be the same line (a curve).
LANEEND_BEND_ROWS = 30  # Where a lane line leaves its straight path, the other line is looked at this many rows higher up.
LANEEND_BEND_PX = 15.0  # The other line that far off its straight path, the same way: the whole lane curves (an S-bend, not a side road).
LANEEND_BEND_GROW = 10.0    # Or bending away that much more over those rows. Measured: curves 16-36 px, side roads under 5 px.
LANEEND_CURVE_HOLD = 0.6    # The other line out of sight just after a curve was seen: still the curve for this long (s).
LANEEND_CURVE = 0.35    # Above its end, line pixels this close to where it would go on (in lane widths) mean it bends on: a curve, not an end.
LANEEND_MAX_SEC = 12.0  # The last distance takes at most this long. It starts as soon as the line end is confirmed, as far off as that is.
END_MAX_SEC = 5.0       # cm=: the last distance takes at most this long. It is bounded because lost-lane back-up is off during it.
EDGE_STOP_Y = 0.63      # Stop when the mat in front of the car ends at this frame height (the mat edge). [narrow FOV 0.7]

JUNCTION_BY_BRANCH = False  # True also treats three or more visible lines as a junction (a fork).
JUNCTION_BEND_DEG = 55      # A lane line that turns off its straight path by at least this many degrees on the floor, within a short
                            # stretch, is a junction too (an L corner, a T, a side road), even with no line across seen. A curve turns
                            # gradually: the tightest the car can drive reads about 15. 0 = off (only the line across, as before).
BEND_ON_LW = 0.08           # sharp_bend: a piece this far behind the line's end still counts as beside it, not behind.
BEND_OFF_LW = 0.2           # A piece of the tape this far sideways of the line's direction at its end has turned away.
BEND_REACH_LW = 0.7         # Only the turned-away part this near the line's end counts (a far, other line does not).
BEND_THICK_LW = 0.06        # ...and they must lie along a line: their spread across it at most this (lane widths; tape is about 0.07 wide).
BEND_RUN_LW = 0.15          # ...and run on sideways by at least this (lane widths), past BEND_OFF_LW, or out of the side of the view.
BRIDGE_LW = 0.15            # sharp_bend: a break in a line this short (lane widths ahead, on the floor; about 3 cm) is a tape joint,
BRIDGE_SIDE_LW = 0.12       # with the line going on within this sideways of where it was heading. A junction gap is far longer.
BRIDGE_MAX = 3              # Joints bridged along one line.
BEND_MIN_PIX = 25           # Turned-away pixels needed (every other row and column): a speck does not make a corner.
STOP_WHOLE_FRAMES = 2   # go_to(): frames a road-area stop point must match in a row (strict match; a mark box needs JUNCTION_FRAMES).
JUNCTION_FRAMES = 3     # Consecutive frames required before a detection counts. This rejects false hits. gap and green use the same count.
JUNCTION_LATCH_FRAMES = 2   # A junction line seen this many frames in a row, low in the frame, is remembered.
JUNCTION_GONE_FRAMES = 2    # Remembered, then missing this many frames: it went under the car, so it counts the same as JUNCTION_FRAMES.
BLIND_NEW_LINE_Y = 0.9      # blind_sec: a line first seen above this frame height (about 7 cm ahead of the car) that goes under the car during blind is the junction.
                            # Lower down it may be the start line the car stands on. Not when the last action ended on a junction: that line is still in front.
                            # One missing frame is a flicker. The count resets if the line comes back.
JUNCTION_LOST_Y = 0.8       # The remembered line must have reached this far down the frame. A line that disappears higher up was a false line.
BLIND_REARM_FRAMES = 5  # blind_sec: a line gone for this many frames in a row during the blind period arms its event, so it can end the action once blind is over.
GAP_MAX_HEADING = 0.5   # Skip the gap test when heading error is larger than this, because the car is badly misaligned with the lane.
GAP_TIP_LOST = 5        # A confirmed gap is followed by its line tips until they are gone for this many frames in a row.
                        # The band test needs both sides, so when one line ends well before the other it may hold only 1-2 frames near the car (runs 20261104-230925, 231241).
GAP_BANDS = 2           # Lane gap: one empty band is enough if the line resumes above it.
                        # If it does not resume, the empty run must be at least this many bands, because one empty top band may just be a missed far line.
CROSSWALK_HOLD_MAX = 1.5    # Maximum seconds to hold steering on a crosswalk right in front of the car. After that, follow the lane again.
                            # Only for a crosswalk found by the band test. Stripes found as blobs are already out of the lane, so the lane is followed.
TURN_CLEAR_FRAMES = 8   # Turn: once the new lane is held for this many frames, steering moves fully from the turn angle to lane following.
TURN_END_OFFSET = 0.4   # Turn end: a line is visible just ahead and the position error is within this.
TURN_END_HEADING = 0.5  # Heading-error limit while a corner horizontal line is still visible. Corner tape is rejected here.
TURN_YAW_PX = 330       # Turns with deg= also end when the camera view has swung this far sideways for 90 degrees (YawMeter).
                        # Lane-ended 90-degree turns in runs 20261103-1939..2014 swung 331-352 px; overshooting ones went on to 420-710.
YAW_TOP = 0.3           # YawMeter uses the frame above this height (the room, not the floor).
TURN_END_OFFSET_WIDE = 0.65   # Turn end, second way: a lane straight ahead (heading within TURN_END_HEADING_WIDE) that is off to one side by up to this.
TURN_END_HEADING_WIDE = 0.3
TURN_EXIT_HEADING = 0.92  # Exit line after the horizontal line disappears. The turn ends when that line comes in front, even if heading has not fallen to 0.5.
TURN_EXIT_CLEAR = 8     # The horizontal line must stay gone for this many frames before it counts as the exit. A one- or two-frame flicker is still corner tape.
TURN_END_FRAMES = 4     # Held for this many frames, the turn ends. Later alignment handles any heading that is still wobbling.
APPROACH_STEER_MAX = 10  # When a junction line is visible ahead, hold steering inside this angle in degrees so the car does not cut into the corner.
HEADING_STEER_MAX = 0.35  # Cap on the heading error used for steering. Align and turn-end tests still use the raw value.
ALIGN_OFFSET = 0.2      # until="aligned": position error is within this (the lane center).
ALIGN_HEADING = 0.2     # Heading error is within this (pointed along the lane). 0.12 almost never happens on the real track.
ALIGN_FRAMES = 5        # Held for this many frames, the car counts as aligned. Used when tr() or tl() aligns and then stops after a turn.
ALIGN_HOLD_HEADING = False  # True: while aligning, a heading error above TURN_END_HEADING is never steered into (old behavior).
                        # False: held only for a real horizontal line (ALIGN_HLINE_HEADING), whose heading is unreliable. Otherwise steered into.
                        # Off: after a corner the exit lane can keep curving (heading -0.79 in run 20261103-201450); holding drove the car straight off the line.
ALIGN_HLINE_HEADING = 0.8   # While aligning, a horizontal line in view really is one only if it drives heading to about +-1.
                            # A lane line curving across the view also trips the band test, at 0.5-0.62, and must be steered along (run 20261104-230540).
ALIGN_HLINE_HOLD = 0.3      # A real line's heading dips below that for 1-3 frames (runs 200649, 201450), so the hold lasts this many seconds after it was last above.
ALIGN_STEER_MAX = 8     # While aligning, if heading error exceeds TURN_END_HEADING, ignore heading and correct position only within this angle.
STRAIGHT_CORR = 6       # straight(): when the lane is visible, correct only within ± this many degrees.
CURVE_ASSIST = 0.5      # curve(): blend of the set angle and lane-follow steering. 0 is a fixed angle, 1 is the lane only.

# Traffic light. tune_light.py adjusts these for the venue lighting and saves them in vision.json.
LIGHT_Y = (0.0, 0.7)    # Height range searched for a lamp, as a fraction of frame height. Stopped just before the crosswalk, the lamp is at 0.54-0.59.
LIGHT_SIDE = 0.5        # Look in this fraction of the width from the left and from the right. 0.5 is the whole width: the light may stand beside the lane or straight ahead at its end. [narrow FOV 0.35]
LIGHT_S_MIN = 110       # Minimum saturation that counts as a color (0-255).
LIGHT_V_MIN = 170       # Minimum value that counts as a color (0-255). An LED emits light, so it is brighter than nearby objects.
LIGHT_AREA_MIN = 11     # Minimum lamp-blob size, in px. Small reflections are ignored. [narrow FOV 25]
LIGHT_AREA_MAX = 1778   # Maximum size, in px. Large red or green objects are ignored. [narrow FOV 4000]
LIGHT_G_HUE = (35, 100)  # wait_green: wide green range used only at the spot where red was seen. A green LED also reads cyan, H 80-95.
LIGHT_LOCAL_S_MIN = 60  # Minimum saturation at that spot. Looser than full-frame LIGHT_S_MIN, because the position is known and a false hit matters less.
LIGHT_LOCAL_V_MIN = 100  # Minimum value at that spot. A green LED tends to read darker than red.
LIGHT_PART = 2.5        # A lamp candidate inside a colored area more than this many times its size is paint (a red speed-limit sign on the floor), not a lamp.
                        # Measured: a piece of a sign ring 2.8-3.9 times, a lamp and its glow about 1.
LIGHT_PART_S_MIN, LIGHT_PART_V_MIN = 70, 60     # How colored a pixel must be to count toward that area.
LIGHT_NEAR = 4          # wait_green: green must lie within (red lamp size times this) of where the red lamp was.
                        # That allows the gap between lamps on the same light and ignores green objects elsewhere.

LENS_POSITION = 6.67    # Manual focus in diopters (1/distance in m). 6.67 is about 15 cm, the near floor where the lane and its ends are read. Always this value: a lens_position in vision.json is ignored. None means autofocus.
                        # calibrate.py measures sharpness over the lane region and saves it in vision.json.
LUX_WARN_RATIO = 1.6    # Warn at the start when the room is this many times brighter or darker than at calibration (vision.json camera_lock Lux).
CAMERA_LOCK = None      # Locked exposure, gain, and white balance {"ExposureTime": us, "AnalogueGain": x, "ColourGains": [r, b]}.
                        # None leaves the camera automatic. calibrate.py measures the venue lighting and saves it in vision.json.
VISION_FILE = Path(__file__).with_name("vision.json")
# ---


def load_vision(path=None):
    """Load saved values from vision.json (or a copy saved with a recording). Missing keys keep their defaults.

    Track color (line_color), the white-line threshold (tune_white.py), traffic-light
    thresholds (tune_light.py), the camera center measured by calibrate.py (camera_x),
    lane width and vanishing point, and exposure / white balance. Focus stays at LENS_POSITION.
    """
    global LINE_COLOR, WHITE_THR, CAMERA_X, LIGHT_S_MIN, LIGHT_V_MIN, LIGHT_AREA_MIN
    global LANE_WIDTH_PX, HORIZON_Y, CAMERA_LOCK, REFLECT_SHARP, LANEEND_OFFSET_CM
    try:
        data = read_json(path or VISION_FILE)
    except (OSError, ValueError):
        return

    def pick(key, default, kind):
        """Keep the current value when the file omits the key, stores null, or stores the wrong type."""
        value = data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, kind):
            return default
        return value

    LINE_COLOR = pick("line_color", LINE_COLOR, str)
    WHITE_THR = pick("white_thr", WHITE_THR, (int, float))
    CAMERA_X = pick("camera_x", CAMERA_X, (int, float))
    LIGHT_S_MIN = pick("light_s_min", LIGHT_S_MIN, (int, float))
    LIGHT_V_MIN = pick("light_v_min", LIGHT_V_MIN, (int, float))
    LIGHT_AREA_MIN = pick("light_area_min", LIGHT_AREA_MIN, (int, float))
    LANE_WIDTH_PX = pick("lane_width_px", LANE_WIDTH_PX, (int, float))
    HORIZON_Y = pick("horizon_y", HORIZON_Y, (int, float))
    CAMERA_LOCK = pick("camera_lock", CAMERA_LOCK, dict)
    REFLECT_SHARP = pick("reflect_sharp", REFLECT_SHARP, (int, float))
    LANEEND_OFFSET_CM = pick("laneend_offset_cm", LANEEND_OFFSET_CM, (int, float))     # calibrate.py --end: measured on this car.


def calibration_note(path=None, today=None):
    """One line on when the venue calibration (calibrate.py) was made and whether its check passed, for the start of a drive."""
    try:
        data = read_json(path or VISION_FILE)
    except (OSError, ValueError):
        return "Calibration: none saved (defaults). Run it on the arena: aicar calibrate"
    info = data.get("calibrated")
    if not isinstance(info, dict) or not isinstance(info.get("at"), str):
        return "Calibration: saved (date unknown: made before dates were kept)"
    try:
        made = time.strptime(info["at"], "%Y-%m-%d %H:%M")
        days = (time.mktime(time.strptime((today or time.strftime("%Y-%m-%d")), "%Y-%m-%d")) - time.mktime(made[:3] + (0, 0, 0) + made[6:])) / 86400
        age = "today" if round(days) <= 0 else "yesterday" if round(days) == 1 else f"{round(days)} days ago"
    except (ValueError, OverflowError):
        age = "?"
    failed = info.get("check") != "ok"
    return (f"Calibration: {info['at']} ({age})" + (", its check FAILED: the lane model is from an earlier one" if failed else "")
            + ("" if age in ("today", "?") else ". Light and place change: calibrate again at a new venue"))


def course_problem(course, start=1, no_stream=False):
    """Why this course cannot be driven as it is, or None. Checked before the motor and the camera are opened."""
    if not 1 <= start <= max(len(course), 1):
        return f"--start {start}: the course has steps 1 to {len(course)}"
    wanted = {n for s in course if s.get("until") == "_stop" for n in ([s["stop"]] if isinstance(s["stop"], str) else s["stop"])}
    missing = sorted(wanted - set(STOP_MODEL.names() if STOP_MODEL else []))
    if missing:
        return (f"go_to(): stop point(s) {', '.join(missing)} not registered (registered: {', '.join(STOP_MODEL.names()) if STOP_MODEL else 'none'}). "
                "Register: aicar stoppoint")
    if XWALK_MODEL is None and any(s.get("until") in ("_crosswalk", "_crossed") for s in course):
        return "_to_crosswalk / _pass_crosswalk need a registered crosswalk: aicar crosswalk (menu: register crosswalk)"
    if no_stream and any(s.get("until") == "enter" for s in course):
        return "a course with next() needs the web page (leave out --no-stream)"
    return None


def lost_hold(speed):
    """How long the lane may stay lost before backing up, at this motor speed: the same distance driven, not the same time.

    A gap in the lines (a crossing, a side road) is a distance. At 0.5 the car took twice as long over it as at 0.95, and
    backed up in the middle of the gap. Below LOST_HOLD_SPEED the time grows as the speed falls, up to LOST_HOLD_SCALE_MAX."""
    scale = LOST_HOLD_SPEED / max(float(speed), 0.05)
    return LOST_HOLD * min(max(scale, 1.0), LOST_HOLD_SCALE_MAX)


def motor_floor(angle):
    """Minimum motor command that actually rolls the car at this steering angle."""
    r = min(abs(angle) / STEER_LIMIT, 1.0)
    return MOTOR_MIN + (MOTOR_MIN_TURN - MOTOR_MIN) * r


def save_vision(**values):
    """Update only the given keys in vision.json. Other keys are left as they are."""
    try:
        data = read_json(VISION_FILE)
    except (OSError, ValueError):
        data = {}
    data.update(values)
    write_json(VISION_FILE, data)


load_vision()

LIGHT_MODEL = None      # Traffic light registered with aicar light (landmarks.LightModel), or None: the color thresholds above are used.
XWALK_MODEL = None      # Crosswalk registered with aicar crosswalk (landmarks.CrosswalkModel), for _to_crosswalk / _pass_crosswalk.
LINE_MODEL = None       # Lane lines registered with aicar line (landmarks.LineModel): sets the reflection cut to this venue's paint.
STOP_MODEL = None       # Stop points registered with aicar stoppoint (landmarks.StopModel), for go_to("A").
LANDMARK_FILES = (landmarks.LIGHT_FILE, landmarks.XWALK_FILE, landmarks.LINE_FILE, landmarks.STOP_FILE)


def load_landmarks(folder=None):
    """Load light.json, crosswalk.json, line.json and stops.json from this folder (a recording's, for a replay), or beside this file."""
    global LIGHT_MODEL, XWALK_MODEL, LINE_MODEL, STOP_MODEL
    folder = Path(folder) if folder else Path(__file__).parent
    LIGHT_MODEL = landmarks.LightModel.load(folder / landmarks.LIGHT_FILE.name)
    XWALK_MODEL = landmarks.CrosswalkModel.load(folder / landmarks.XWALK_FILE.name)
    LINE_MODEL = landmarks.LineModel.load(folder / landmarks.LINE_FILE.name)
    STOP_MODEL = landmarks.StopModel.load(folder / landmarks.STOP_FILE.name)


load_landmarks()


_ELLIPSE_RECTS = {}


def ellipse_rects(size):
    """Split an ellipse kernel into rectangles that share its center. Their union is the original ellipse.

    An ellipse is convex and symmetric, so every row of width w fits in a rectangle of width w and the full height of rows at least that wide. Each of those rectangles lies inside the ellipse, and together they are exactly the ellipse.
    """
    if size not in _ELLIPSE_RECTS:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        c = size // 2
        reach = {}      # Half-width to the greatest distance from the center of a row with that half-width.
        for r in range(size):
            cols = np.flatnonzero(k[r])
            assert cols[0] + cols[-1] == 2 * c, "the ellipse kernel is not symmetric"
            hw = c - cols[0]
            reach[hw] = max(reach.get(hw, 0), abs(r - c))
        _ELLIPSE_RECTS[size] = [np.ones((2 * hh + 1, 2 * hw + 1), np.uint8) for hw, hh in reach.items()]
    return _ELLIPSE_RECTS[size]


REFLECT_K9, REFLECT_K5 = np.ones((9, 9), np.uint8), np.ones((5, 5), np.uint8)


_COLUMNS = np.arange(4096)      # Column numbers, made once (find_lines).
_OPEN_K3 = np.ones((3, 3), np.uint8)


MAT_BRIDGE_K = np.ones((15, 1), np.uint8)    # mat_mask: bridges a line across the mat up to 60 px tall (quarter size x 4).
MAT_BRIDGE_Y = 0.55     # mat_mask: the bridge works below this frame height only (the mat-edge stop looks at EDGE_STOP_Y 0.63).


def tophat_ellipse(img, size):
    """Faster equivalent of cv2.morphologyEx(img, MORPH_TOPHAT, ellipse kernel).

    Erosion with a union kernel is the minimum of the rectangle erosions, and dilation is the maximum. OpenCV separates rectangles and processes them quickly.
    """
    rects = ellipse_rects(size)
    eroded = cv2.erode(img, rects[0])
    for r in rects[1:]:
        eroded = cv2.min(eroded, cv2.erode(img, r))
    opened = cv2.dilate(eroded, rects[0])
    for r in rects[1:]:
        opened = cv2.max(opened, cv2.dilate(eroded, r))
    return cv2.subtract(img, opened)


def pct(a, q):
    """np.percentile(a, q) of a 1-D array (linear), by partial sort: much quicker than a full np.percentile on the Pi."""
    n = a.size
    pos = (n - 1) * (q / 100)
    lo = int(pos)
    t = pos - lo
    if t == 0 or lo + 1 >= n:
        return a.dtype.type(np.partition(a, lo)[lo])
    part = np.partition(a, (lo, lo + 1))
    v0, v1 = part[lo], part[lo + 1]
    if a.dtype == np.float32 and _LERP_OK:
        return _lerp32(v0, v1, t)
    return np.percentile(np.array([v0, v1], a.dtype), t * 100)       # The two around it, mixed exactly as numpy does.


def _lerp32(v0, v1, t):
    """np.percentile(np.array([v0, v1], float32), t * 100), worked out directly (much quicker on the Pi): numpy divides
    t*100 by 100 in float32, and mixes the two in float32, from the far side once past half way."""
    T = np.float32
    q = T(t * 100) / T(100)
    k = np.floor(q)
    lo2, hi2 = (v0, v1) if k == 0 else (v1, v1)
    g = T(q - k)
    d = T(hi2 - lo2)
    return T(hi2 - d * (1 - g)) if g >= 0.5 else T(lo2 + d * g)


def _lerp_check():
    """True when _lerp32 gives exactly what this numpy's percentile gives (checked once at start). Numpy versions mix
    differently in the last bit, so with another numpy pct() keeps calling numpy itself."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        v = np.sort((rng.random(2) * rng.choice([1.0, 255.0, 1000.0])).astype(np.float32))
        t = float(rng.random())
        want = np.percentile(np.array(v, np.float32), t * 100)
        got = _lerp32(v[0], v[1], t)
        if not (got == want and type(got) is type(want)):
            return False
    return True


_LERP_OK = _lerp_check()


def percentile_u8(img, q):
    """np.percentile(img, q) with linear interpolation, computed from a histogram without sorting. For uint8 images."""
    n = img.size
    cum = np.cumsum(cv2.calcHist([img], [0], None, [256], [0, 256]).ravel().astype(np.int64))
    pos = (n - 1) * (q / 100)                       # Same position calculation as numpy's default linear method.
    lo = int(np.floor(pos))
    t = pos - lo
    a = float(np.searchsorted(cum, lo, side="right"))                 # The lo-th smallest value.
    b = float(np.searchsorted(cum, min(lo + 1, n - 1), side="right"))
    return b - (b - a) * (1 - t) if t >= 0.5 else a + (b - a) * t   # Same interpolation formula as numpy.


# --- Detection Overlay (Browser Live View) ---
# Colors (BGR) drawn on the debug view.
VIEW_DETAIL = False     # The view shows only what drives the car (lane, events, steering). True: also reflection boxes, points
                        # left out, the pair fit's pieces, for checking detection.
VIEW_TINT = 0.35        # How strongly the detected line pixels are colored in the view (0 to 1).
LANE_FILL = 0.1         # The followed lane between its two lines, filled this strongly (LANE_POINTS = "pairs"). 0 = not filled.
C_LANE = (255, 210, 60)     # Pixels detected as the white line (translucent).
# The view paints line pixels 0.45 * pixel + 0.55 * C_LANE: per channel, as a lookup table (much faster than float math on the Pi).
LANE_TINT = np.stack([((1 - VIEW_TINT) * np.arange(256) + VIEW_TINT * c).astype(np.uint8) for c in C_LANE], axis=1).reshape(1, 256, 3)
C_LEFT, C_RIGHT = (255, 140, 40), (60, 60, 255)    # Left line / right line.
C_PATH = (80, 230, 80)      # Lane center the car follows.
C_JUNC = (255, 0, 255)      # Junction (horizontal line).
C_XWALK = (240, 240, 240)   # Crosswalk.
C_GAP = (255, 128, 0)       # Lane gap.
C_AT = (0, 230, 255)        # at= point, the line where this action ends.
C_PERSON = (60, 220, 60)    # LEGO figure.
C_DIM = (110, 110, 110)


def chip(img, text, x, y, color, scale=0.42, dark=None):
    """Small label: text on a colored background. (x, y) is the top left. Returns the right-edge x.

    img: a Canvas (overlay.py, drawn later by the browser or render) or an ndarray (painted at once). Same for the other draw helpers.
    """
    d = as_draw(img)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(min(max(x, 0), d.shape[1] - tw - 8))
    y = int(min(max(y, 0), d.shape[0] - th - 8))
    d.rectangle((x, y), (x + tw + 8, y + th + 8), color, -1)
    if dark is None:
        dark = sum(color) > 380     # Black text on a bright background.
    d.putText(text, (x + 4, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
              (20, 20, 20) if dark else (255, 255, 255), 1, cv2.LINE_AA)
    return x + tw + 8


def dashed_hline(img, y, color, x0=0, x1=None, dash=10, gap=6, thick=1):
    d = as_draw(img)
    y = int(min(max(y, 0), d.shape[0] - 1))
    x1 = d.shape[1] if x1 is None else x1
    d.dashed(x0, y, x1, color, dash, gap, thick)


def draw_person(dbg, detector, at, detail=None):
    """LEGO figure on the view: the line its feet must pass to count, blobs of the right size with the wrong colors, and the match.
    detail: the line and the near misses too (default VIEW_DETAIL; the tuning tools always want them)."""
    detail = VIEW_DETAIL if detail is None else detail
    dbg = as_draw(dbg)
    if at is not None and detail:
        dashed_hline(dbg, int(at * FRAME_H), C_PERSON, dash=4, gap=8)
    for x, y, w, h, color, shape in detector.near if detail else ():     # Candidates that did not match.
        dbg.rectangle((x, y), (x + w, y + h), C_DIM, 1)
        chip(dbg, f"c{color:.2f} s{shape:.2f}", x, y - 18, C_DIM, 0.35)
    if detector.found:
        f = detector.found
        x, y, w, h = figure_box(f["x"], f["y"], f["w"], f["h"], f["feet"])      # The whole figure, though only its colored part was matched.
        dbg.rectangle((x, y), (x + w, y + h), C_PERSON, 2, cv2.LINE_AA)
        look = "" if f.get("look") is None else f" L{f['look']:.2f}"     # L: how much it looks like the registered figure (EMBED_MIN).
        chip(dbg, f"PERSON c{f['score']:.2f} s{f['shape']:.2f}{look}" if detail else "PERSON", f["x"], f["y"] - 22, C_PERSON, 0.38)


class LaneDetector:
    def __init__(self):
        self.reflections, self.reflect_boxes = [], []     # Blobs dropped as light reflections this frame, and their boxes in the ROI.
        self.sign_ids = []                      # Blobs dropped as part of a red sign on the floor this frame.
        self.signs = None                       # Mask of red-rimmed floor signs this frame (ROI size), or None.
        self.xwalk2_y, self.xwalk2_boxes = None, []     # The registered crosswalk (aicar crosswalk): near edge y, stripe boxes (ROI px).
        self.roi = None
        self.curve_t = None                     # Last time a line leaving its straight path was judged a curve of the whole lane.
        self.now, self.moving = None, False     # Set by Driver before each frame: time, and driving forward without a big steer (motion test).
        self.tracker = None         # LANE_MODEL = "track": the lane tracker (lane_track.py), made on first use.
        self.assist_turn = False
        self.det_bands = None       # The band method's own answer this frame, whichever LANE_MODEL drives.
        self.pairs = None           # LANE_POINTS = "pairs": the pair fit (lane_pairs.py), made on first use.
        self.bend = None            # sharp_bend: the nearest sharp turn of a lane line in the last frame, or None.
        self.jump_last = self.jump_pending = None   # confirm_jump: the lane last given out, and a jump waiting for a second frame.
        self.jump_held = False      # This frame's jump was held back.
        self.jump_count = 0         # Frames held back in a row.
        self.jump_gone = 0          # Frames with no lane in a row (confirm_jump keeps the last lane through a short loss).
        self.band_line = None       # Its center line this frame, (x at near_y, x at far_y) before clipping, or None.
        self.moving_since = None
        self.history = collections.deque(maxlen=40)     # (time, gray ROI) of recent frames for the motion test.
        self.glare_spots = []                   # (x0, y0, x1, y1, time) of blobs judged reflections by motion, in ROI px.
        self.paint_boxes = []                   # Boxes judged paint by motion this frame (for the view).
        self.left = [None] * SLICES     # Left and right line x in each band from the previous frame.
        self.right = [None] * SLICES
        self.center = [None] * SLICES
        self.width = [None] * SLICES    # Learned lane width for each band.
        self.auto_thr = None            # White-line threshold from automatic calibration.
        self.thr = 0                    # Threshold actually used for this frame.
        self.misses = 0                 # Consecutive frames with no lane found.
        self.edge = False               # The mat ends right in front of the car (driving stops).
        self.edge_hits = 0
        self.mat_top = 0                # Highest y where the mat starts in front of the car, in screen coordinates.
        self.crosswalk = False          # A crosswalk is visible in front of the car (below JUNCTION_Y) this frame.
        self.crosswalk_any = False      # A crosswalk is visible anywhere in the frame (while passing it).
        self.stripes = []               # Crosswalk stripes found as short blobs this frame: [(x, y, w, h)].
        self.blobs, self.band_h = (np.zeros((1, 1), np.int32), np.zeros((1, 5), np.int32)), 1   # Mask blobs this frame, for assign().
        self.junction_y = self.crosswalk_y = None
        self.gap_y = None               # Where the lane ends ahead: y of the last band that still showed a line. None if it does not end.
        self.gap_why = ""               # Why gap_y is None (diagnostics).
        self.near_line = False          # Whether a line is visible just in front of the car (the bottom band). False while over a gap.
        self.far_y = None               # y of a horizontal line seen above the ROI (far). For slow= braking. Driver fills it only when needed.
        self.mask = np.zeros((1, 1), np.uint8)  # Line mask this frame.
        self.hrun = None                # Nearest horizontal line this frame, in the mask: (first column, last column, center row).
        self.tip_y = None               # Where the lane lines that end ahead end (lane_tip).
        self.end_y = self.end_side = None   # Where the lane's line ends ahead (lane_end): y of its top, "left"/"right".
        self.tip_diff = None            # How far apart the two line tips are, while both are in view.
        self.gap_hits = 0               # Frames the band test saw a gap, while its line tips stayed in view.
        self.tip_lost = 0               # Frames in a row the line tips were not seen.
        self.gap_held = False           # A gap was confirmed and its line tips are being followed.
        top = int(FRAME_H * ROI_TOP)
        h = (FRAME_H - top) // SLICES
        self.band_y = [FRAME_H - (i + 0.5) * h for i in range(SLICES)]

    def lane_span(self, y):
        """(left, right) x of the lane at this height, from the lines found. A line not found is taken one lane width from the other. None if neither was found in any band."""
        lefts, rights, ys = [], [], []
        for i in range(SLICES):
            left, right = self.left[i], self.right[i]
            if left is None and right is None:
                continue
            ys.append(self.band_y[i])
            lefts.append(left if left is not None else right - self.lane_width(i))
            rights.append(right if right is not None else left + self.lane_width(i))
        if not ys:
            return None
        order = np.argsort(ys)
        ys, lefts, rights = np.array(ys, dtype=np.float64)[order], np.array(lefts, dtype=np.float64)[order], np.array(rights, dtype=np.float64)[order]

        def at(xs):
            if len(ys) == 1:
                return float(xs[0])
            if y <= ys[0] or y >= ys[-1]:     # Beyond the bands: straight on from the nearest two.
                a, b = (0, 1) if y <= ys[0] else (-2, -1)
                return float(xs[a] + (xs[b] - xs[a]) * (y - ys[a]) / (ys[b] - ys[a]))
            return float(np.interp(y, ys, xs))
        return at(lefts), at(rights)

    def in_lane(self, x, feet):
        """Whether a figure with its center at x and its feet at this height stands between the lane lines. True when no line is known, because then it cannot be told."""
        span = self.lane_span(feet)
        if span is None:
            return True
        margin = PERSON_LANE_MARGIN * (span[1] - span[0])
        return span[0] - margin <= x <= span[1] + margin

    def model_width(self, i):
        """Perspective model: lane width is 0 at the vanishing point (HORIZON_Y) and widens linearly downward."""
        return LANE_WIDTH_PX * (self.band_y[i] - HORIZON_Y) / (max(342 - HORIZON_Y, 1.0))

    def lane_width(self, i):
        return self.width[i] or self.model_width(i)

    def learn_width(self, i, w):
        if not LEARN_WIDTH:
            return
        # Keep the learned value from drifting far from the model. A bad learn pulls the car to one side.
        m = self.model_width(i)
        w = float(np.clip(w, 0.8 * m, 1.25 * m))
        self.width[i] = w if self.width[i] is None else 0.9 * self.width[i] + 0.1 * w

    @staticmethod
    def mat_mask(gray, bridge=False):
        """Black-mat region: the convex hull of the largest dark region connected to the bottom. None when the whole frame is mat.

        bridge=True also joins the mat across a line running across it near the car (see MAT_BRIDGE_K). Only the mat-edge stop
        uses that one: the line pixels keep the mat as it always was."""
        small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4), interpolation=cv2.INTER_AREA)
        bg = cv2.medianBlur(small, 7)       # Floor brightness with the lines removed, used to test whether the mat and the outside really differ.
        bthr, _ = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo, hi = bg[bg < bthr], bg[bg >= bthr]
        if not len(lo) or not len(hi) or float(hi.mean()) - float(lo.mean()) < MAT_MIN_CONTRAST:
            return None     # One kind of floor, split only by lighting, so the whole frame is mat.
        thr, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = (small < thr).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))  # Fill in the white lines so the mat is one blob. [narrow FOV 9x9]
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((1, 21), np.uint8))  # Also bridge the gaps between crosswalk stripes. [narrow FOV 1x31]
        # bridge: also across a line that runs across the lane (a junction). Near the car it is thicker than the 6x6 closing, split the
        # mat in two, and the mat "ended" at it: the lane was lost and the mat-edge stop held the car there for good. Only near the car
        # (below MAT_BRIDGE_Y): far away the bright floor beyond the mat's far edge is a thin strip, and bridging it joined the mat to
        # dark furniture behind it.
        near_row = max(int((FRAME_H * MAT_BRIDGE_Y - (FRAME_H - gray.shape[0])) / 4), 0)
        if bridge and near_row < dark.shape[0]:
            bridged = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, MAT_BRIDGE_K)
            dark[near_row:] = bridged[near_row:]
        n, labels = cv2.connectedComponents(dark)
        if n <= 1:
            return None
        bottom = labels[-3:].ravel()
        counts = np.bincount(bottom[bottom > 0], minlength=n)
        if counts.max() == 0:
            return None
        blob = (labels == counts.argmax()).astype(np.uint8)
        # The hull of the outline is the hull of every pixel, and the outline is a few points instead of the whole mat.
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = np.zeros_like(small)
        cv2.fillConvexPoly(hull, cv2.convexHull(contours[0]), 255)
        hull = cv2.dilate(hull, np.ones((3, 3), np.uint8))
        return cv2.resize(hull, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def stripe_size(stats, i, top):
        """A blob's width, height (in lane widths at its height) and how much of its box it fills."""
        x, y, w, h, area = stats[i]
        cy = top + y + h / 2
        lane_w = max(LANE_WIDTH_PX * (cy - HORIZON_Y) / max(342 - HORIZON_Y, 1.0), 1.0)
        return w / lane_w, h / lane_w, area / max(w * h, 1)

    def find_registered_crosswalk(self, comps, top):
        """The crosswalk registered with aicar crosswalk: enough blobs shaped like its stripes, between the lane lines.

        Sets xwalk2_y (frame y of the near edge of the nearest stripe, or None) and xwalk2_boxes. Returns the stripe ids,
        which are left out of the lane lines. The older crosswalk test (crosswalk_y) is not touched.
        """
        self.xwalk2_y, self.xwalk2_boxes = None, []
        model = XWALK_MODEL
        if model is None or model.w is None:
            return []
        n, labels, stats, cents = comps
        H = labels.shape[0]
        cand = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < STRIPE_MIN_AREA or h >= REFLECT_LINE_H * H:
                continue
            if not model.like(*self.stripe_size(stats, i, top)):
                continue
            # Between the lane lines of the last frame at this height, when both are known.
            band = min(max((H - (y + h // 2)) // max(H // SLICES, 1), 0), SLICES - 1)
            l, r = self.left[band], self.right[band]
            cx = x + w / 2
            if l is not None and r is not None and not l < cx < r:
                continue
            cand.append(i)
        if len(cand) < model.count:
            return []
        if model.sharp is not None and self.roi is not None:     # Crisp paint, not a soft ceiling-light reflection.
            cand = [i for i in cand if (self.blob_sharpness(self.roi, comps, i) or 0.0) >= model.sharp]
        rows = []
        for i in cand:
            x, y, w, h, _ = stats[i]
            cy = top + y + h / 2
            lane_w = max(LANE_WIDTH_PX * (cy - HORIZON_Y) / max(342 - HORIZON_Y, 1.0), 1.0)
            rows.append((i, (x + w / 2) / lane_w, cy, h))
        found = model.row(rows)
        if len(found) < model.count:
            return []
        self.xwalk2_boxes = [tuple(int(v) for v in stats[i][:4]) for i in found]
        self.xwalk2_y = max(top + stats[i][1] + stats[i][3] for i in found)
        return found

    @staticmethod
    def sign_area(roi):
        """Mask (ROI size) of red-rimmed signs painted on the floor, rim and face, such as a speed limit "30" or "70".

        Their white face and numbers would otherwise read as lane lines or as crosswalk stripes.
        Half size: a sign is big, so this is plenty and cheap.
        """
        H, W = roi.shape[:2]
        small = cv2.resize(roi, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        hue, sat, val = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
        red = (((hue < 10) | (hue > 170)) & (sat >= SIGN_S_MIN) & (val >= SIGN_V_MIN)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
        out = None
        for i in range(1, n):
            if st[i][4] * 4 < SIGN_MIN_AREA:
                continue
            ys, xs = np.nonzero(lab == i)
            hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))
            face = np.zeros_like(red)
            cv2.fillConvexPoly(face, hull, 1)
            inside = face.astype(bool) & (red == 0)
            if (val[inside] >= 150).sum() < SIGN_WHITE_SHARE * face.sum():
                continue        # Nothing white inside: a red object, not a sign face.
            out = face if out is None else (out | face)
        if out is None:
            return None
        out = cv2.dilate(out, np.ones((3, 3), np.uint8))
        return cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    @staticmethod
    def on_red_sign(roi, box):
        """Whether a white blob sits on a red sign (a speed-limit number on the floor): red paint right around it."""
        x, y, w, h = box
        pad = int(1.5 * max(w, h))
        H, W = roi.shape[:2]
        crop = roi[max(y - pad, 0):min(y + h + pad, H), max(x - pad, 0):min(x + w + pad, W)]
        if crop.size == 0:
            return False
        hue, sat, val = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
        red = ((hue < 10) | (hue > 170)) & (sat >= SIGN_S_MIN) & (val >= SIGN_V_MIN)
        return red.mean() >= SIGN_RED_SHARE

    def find_stripes(self, comps, shape, top, roi=None):
        """Crosswalk stripes: short separate white blobs (STRIPE_MAX of the lane width) inside the lane. Lane lines are long blobs.

        Inside the lane: at the stripe's rows there is lane-line (long blob) mask on both its left and its right, in this same frame.
        comps is connectedComponentsWithStats of the mask. Returns (blob ids to clear, [(center x, center y, w, h)] in frame y),
        both empty unless STRIPES_MIN or more are found, so a lone short piece of a lane line stays in.
        Catches a crosswalk of a few staggered stripes that the band test (CROSSWALK_LINES in one band) misses.
        """
        n, labels, stats, cents = comps
        found = []
        self.sign_ids = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < STRIPE_MIN_AREA:
                continue
            cy = top + float(cents[i][1])
            lane_w = LANE_WIDTH_PX * (cy - HORIZON_Y) / (max(342 - HORIZON_Y, 1.0))
            # Cut off by an ROI edge: the end of a lane line (before a gap, at the ROI top, at a side), not a whole stripe.
            cut = y <= 0 or x <= 0 or x + w >= shape[1] or y + h >= shape[0]
            if not cut and lane_w > 0 and max(w, h) < STRIPE_MAX * lane_w:
                if roi is not None and self.on_red_sign(roi, (int(x), int(y), int(w), int(h))):
                    self.sign_ids.append(i)     # Part of a red speed-limit sign: neither a stripe nor a lane line.
                    continue
                found.append((i, int(cents[i][0]), int(cy), int(w), int(h)))
        if len(found) < STRIPES_MIN:
            return [], []
        short = np.zeros(n, bool)
        short[[f[0] for f in found]] = True
        inside = []
        for i, cx, cy, w, h in found:
            band = labels[max(0, cy - top - h // 2):cy - top + h // 2 + 1]
            cols = np.flatnonzero(((band > 0) & ~short[band]).any(axis=0))     # Columns with a long blob (lane line) at these rows.
            if len(cols) and cols[0] < cx < cols[-1]:
                inside.append((i, cx, cy, w, h))
        if len(inside) < STRIPES_MIN:
            return [], []
        # A crosswalk's stripes are alike in size. Pieces of very different sizes (a thin far line end, a tall reflection
        # near the car) are not one. Sizes are compared after scaling to the same distance (the lane width at their height).
        def size(f):
            lane_w = max(LANE_WIDTH_PX * (f[2] - HORIZON_Y) / max(342 - HORIZON_Y, 1.0), 1.0)
            return f[3] / lane_w, f[4] / lane_w
        best = []
        for f in inside:
            w0, h0 = size(f)
            alike = [g for g in inside if 0.5 <= size(g)[0] / w0 <= 2 and 0.5 <= size(g)[1] / h0 <= 2]
            if len(alike) > len(best):
                best = alike
        if len(best) < STRIPES_MIN:
            return [], []
        return [f[0] for f in best], [f[1:] for f in best]

    @staticmethod
    def near_blobs(comps, short_only=False):
        """Ids of blobs big enough and centered in the near part of the ROI (REFLECT_Y). short_only leaves out tall ones (lane lines, REFLECT_LINE_H)."""
        n, labels, stats, cents = comps
        H = labels.shape[0]
        y_min, tall = REFLECT_Y * H, REFLECT_LINE_H * H
        return [i for i in range(1, n) if stats[i][4] >= REFLECT_MIN_AREA and cents[i][1] >= y_min
                and not (short_only and stats[i][3] >= tall)]

    def blob_sharpness(self, roi, comps, i):
        """Edge steepness of blob i: 90th percentile edge gradient over its contrast with the floor around it. None if it cannot be measured.

        Tape has crisp edges (about 0.37-0.5). A ceiling-light reflection on a glossy mat fades out softly (about 0.18-0.33).
        """
        _, labels, stats, _ = comps
        H, W = labels.shape
        x, y, w, h, _ = stats[i]
        x0, y0, x1, y1 = max(x - 5, 0), max(y - 5, 0), min(x + w + 5, W), min(y + h + 5, H)
        blob = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(blob, REFLECT_K9) > 0) & (blob == 0)
        edge = (cv2.dilate(blob, REFLECT_K5) > 0) & (cv2.erode(blob, REFLECT_K5) == 0)
        if not ring.any() or not edge.any():
            return None
        g = self._roi_gray(roi)[y0:y1, x0:x1].astype(np.float32)    # The ROI's gray, made once per frame (the same pixels as converting the crop).
        if LINE_COLOR == "black":
            g = 255 - g
        # The Sobel of the crop alone: every edge pixel is at least 3 px inside it (5 px margin, 2 px dilation), or on the ROI's own
        # border, so it is the same as a Sobel of the whole ROI, on a few blobs' area instead of all of it.
        mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
        contrast = pct(g[blob > 0], 90) - pct(g[ring], 50)
        return float(pct(mag[edge], 90) / max(contrast, 1.0))

    def _roi_gray(self, roi):
        """Gray of this picture (uint8). Inside a detection it is made once per frame and shared (line mask, reflections);
        outside one (tools calling a part directly) it is made each time."""
        cache = getattr(self, "_sharp_cache", None)
        if cache is None:
            return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        key = (roi.__array_interface__["data"][0], roi.shape, roi.strides)
        if key not in cache:
            cache[key] = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cache[key]

    def motion_verdict(self, gray, old, box):
        """"paint", "glare", or None: how far the blob came toward the car since the old frame, on the floor.

        The blob is looked up in the old frame at its place and higher up (farther away), where paint would have been.
        """
        x, y, w, h = box
        if h < 6 or w < 6 or y <= 1:      # Cut off by the ROI top: where it came from cannot be seen.
            return None
        top = FRAME_H - 2 * gray.shape[0]       # gray and old are half size: matching costs a quarter.
        x2, y2, w2, h2 = x // 2, y // 2, max(w // 2, 3), max(h // 2, 3)
        H, W = gray.shape
        patch = gray[y2:y2 + h2, x2:x2 + w2]
        sx0, sx1 = max(x2 - GLARE_SIDE_PX // 2, 0), min(x2 + w2 + GLARE_SIDE_PX // 2, W)    # Wide sideways: a slanted line coming closer slides sideways too.
        sy1 = min(y2 + h2 + 3, H)
        # Only as far up as paint could have come from in the time (GLARE_REACH_CM farther away): the rest costs time for nothing.
        scale = CAMERA_F * LANE_CM / LANE_WIDTH_PX * max(342 - HORIZON_Y, 1.0)
        far_y = HORIZON_Y + scale / (floor_cm(top + 2 * (y2 + h2)) + GLARE_REACH_CM)       # Bottom edge back then, frame px.
        sy0 = int(min(max((far_y - top) / 2 - h2 - 2, 0), max(sy1 - h2, 0)))
        region = old[sy0:sy1, sx0:sx1]
        if region.shape[0] < h2 or region.shape[1] < w2 or patch.shape != (h2, w2):
            return None
        full = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
        res = full.max(axis=1)      # Best match per old height, anywhere sideways (paint may slide sideways as it comes closer).
        dx = GLARE_STILL_DX // 2
        c0 = x2 - sx0               # Column of res where the blob would be with no sideways shift.
        res_still = full[:, max(c0 - dx, 0):c0 + dx + 1].max(axis=1)     # A reflection stays where it is sideways too.
        bottoms = top + 2 * (sy0 + np.arange(len(res)) + h2)      # Bottom edge (where the blob meets the floor) in the old frame, per row of res.
        came = scale / np.maximum(bottoms - HORIZON_Y, 1.0) - floor_cm(top + 2 * (y2 + h2))
        still = res_still[np.abs(came) < GLARE_STILL_CM]
        moved = res[came >= GLARE_PAINT_CM]
        if not len(still) or not len(moved):      # Both answers must be testable.
            return None
        still, moved = float(still.max()), float(moved.max())
        # Only a clear answer counts. A repeating pattern (crosswalk stripes) matches both, and is left to the other tests.
        if moved >= GLARE_MATCH and still < moved - GLARE_MARGIN:
            return "paint"
        if still >= GLARE_MATCH and moved < still - GLARE_MARGIN:
            return "glare"
        return None

    def find_reflections(self, roi, comps):
        """Ids of blobs that are light reflections, not paint.

        1. Motion, while driving forward: a blob that did not come toward the car is a reflection, one that did is paint.
        2. Otherwise, standing still, a spot judged a reflection in the last GLARE_KEEP seconds of driving is still one.
        3. Otherwise the edge test (REFLECT_SHARP, capped at REFLECT_FALLBACK_MAX) in the near part of the ROI.
        """
        n, labels, stats, cents = comps
        now = self.now
        gray = None
        old = None
        if now is not None and getattr(_course, "REFLECT_MOTION_ON", True):     # course.py switch: off skips the motion test (faster).
            if self.moving:
                self.moving_since = now if self.moving_since is None else self.moving_since
            else:
                self.moving_since = None
            gray = cv2.resize(self._roi_gray(roi), (roi.shape[1] // 2, roi.shape[0] // 2), interpolation=cv2.INTER_AREA)
            if self.moving_since is not None and now - self.moving_since >= GLARE_WINDOW:
                past = [g for t, g in self.history if now - t >= GLARE_WINDOW and t >= self.moving_since]
                old = past[-1] if past else None
            self.history.append((now, gray))
            if self.moving:     # A spot is let go only while driving, when it would have shown itself to be paint by now.
                self.glare_spots = [g for g in self.glare_spots if now - g[4] <= GLARE_KEEP]
        H = labels.shape[0]
        near = set(self.near_blobs(comps, short_only=True))
        cut = None if REFLECT_SHARP is None else min(REFLECT_SHARP, REFLECT_FALLBACK_MAX)
        crisp = GLARE_SHARP_MAX
        if LINE_MODEL is not None and REFLECT_SHARP is not None:
            # Lane lines registered here (aicar line): cut just under this venue's softest paint, and protect anything as crisp as it.
            cut = LINE_MODEL.reflect_cut()
            crisp = min(GLARE_SHARP_MAX, LINE_MODEL.sharp * 0.95)
        out, self.paint_boxes = [], []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < REFLECT_MIN_AREA or h >= REFLECT_LINE_H * H:      # Long lane lines are not judged: that also keeps the cost down.
                continue
            cx, cy = cents[i]
            known = any(gx0 <= cx <= gx1 and gy0 <= cy <= gy1 for gx0, gy0, gx1, gy1, _ in self.glare_spots)
            verdict = self.motion_verdict(gray, old, (x, y, w, h)) if old is not None else None
            if verdict == "glare" and not known:     # A spot already checked needs no second edge test: that keeps the cost down.
                sharp = self.blob_sharpness(roi, comps, i)
                if sharp is not None and sharp >= crisp:     # Crisp like tape: a piece of a lane line that only seemed to stand still.
                    verdict = None
            if verdict == "glare":
                self.glare_spots.append((x, y, x + w, y + h, now))
                out.append(i)
                continue
            if verdict == "paint":
                self.paint_boxes.append((x, y, w, h))
                continue
            if not self.moving and known:
                # Standing still: a reflection seen while driving is still there. While driving, paint can pass through the same spot.
                out.append(i)
                continue
            if cut is not None and i in near:
                sharp = self.blob_sharpness(roi, comps, i)
                if sharp is not None and sharp < cut:
                    out.append(i)
        return out

    @staticmethod
    def clear_blobs(mask, labels, stats, ids):
        """Remove these blobs from the mask and from the labels, in place. A blob lies inside its own box, so only the
        boxes are touched: the same result as testing every pixel of the picture, at a fraction of the cost."""
        for i in ids:
            x, y, w, h = (int(v) for v in stats[i][:4])
            box = labels[y:y + h, x:x + w]
            hit = box == i
            mask[y:y + h, x:x + w][hit] = 0
            box[hit] = 0

    def drop_reflections(self, roi, mask):
        """Remove reflection blobs from the mask in place. Returns the connected components of what is left."""
        comps = cv2.connectedComponentsWithStats(mask, 8)       # The mask is 0 / 255: anything not 0 is a blob pixel.
        self.reflections = self.find_reflections(roi, comps)
        self.reflect_boxes = []
        if self.reflections:        # Gone before anything else looks at the mask, so they cannot make a lane line or a crosswalk.
            self.reflect_boxes = [tuple(int(v) for v in comps[2][i][:4]) for i in self.reflections]
            self.clear_blobs(mask, comps[1], comps[2], self.reflections)     # Cheaper than labelling the mask again.
            stats = comps[2].copy()
            stats[self.reflections, 4] = 0      # Gone: no area, so nothing below takes it for a stripe or a line.
            comps = (comps[0], comps[1], stats, comps[3])
        return comps

    def find_specks(self, comps, shape, band_h):
        """Ids of small blobs near the frame sides (SPECK_SIDE). They are off-track objects, not lane lines."""
        n, _, stats, cents = comps
        return [i for i in range(1, n)
                if stats[i, 3] < 1.5 * band_h and stats[i, 2] < HLINE_RATIO * FRAME_W and stats[i, 1] > 0
                and stats[i, 1] + stats[i, 3] < shape[0] // 2 and stats[i, 0] > 0 and stats[i, 0] + stats[i, 2] < shape[1]
                and not SPECK_SIDE * FRAME_W <= cents[i][0] <= (1 - SPECK_SIDE) * FRAME_W]

    def blob_height(self, i, x):
        """Height in px of the mask blob that band i's line at x belongs to (0 if none). A real lane line spans many bands."""
        labels, stats = self.blobs
        y1 = labels.shape[0] - (i + 1) * self.band_h
        col = labels[max(0, y1):y1 + self.band_h, int(np.clip(x, 0, labels.shape[1] - 1))]
        ids = col[col > 0]
        return int(stats[ids, 3].max()) if len(ids) else 0

    def line_mask(self, roi):
        gray = cv2.GaussianBlur(self._roi_gray(roi), (5, 5), 0)
        if LINE_COLOR == "black":
            # Black line on a white floor: inverting brightness makes it look like a white line on a black floor (the mat is the white floor).
            gray = cv2.bitwise_not(gray)
        # Top-hat keeps only what is brighter than the nearby floor and thin, so thick reflections and bright floor drop out.
        half = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
        # INTER_LINEAR_EXACT: the same pixels on every OpenCV version. Plain INTER_LINEAR rounds differently on 4.x (the Pi)
        # and 5.x (the Mac), by one grey level, which made a replay on the Mac decide slightly differently from the drive.
        top = cv2.resize(tophat_ellipse(half, TOPHAT_K), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)
        mat = self.mat_mask(gray) if USE_MAT_MASK else None
        if mat is None:     # No separate mat (or the mask is off): nothing is cut, and the mat runs to the ROI top.
            self.mat_top = int(FRAME_H * ROI_TOP)
            self.edge_hits, self.edge = 0, False
        else:
            top = cv2.bitwise_and(top, mat)
            # How far the mat continues in front of the car (center 1/6 of the width, which was 1/4 on the narrow FOV). Ending too near the car means the mat edge.
            cx0 = int(FRAME_W * CAMERA_X)
            front = mat[:, max(cx0 - FRAME_W // 12, 0):cx0 + FRAME_W // 12]
            rows = np.flatnonzero(front.mean(axis=1) > 128)
            mat_top = int(FRAME_H * ROI_TOP) + (rows[0] if len(rows) else front.shape[0])
            self.mat_top = mat_top
            if mat_top > FRAME_H * EDGE_STOP_Y:
                # The mat seems to end near the car. A line across the lane can split the mat the same way: look again with the
                # mat joined across such a line, and stop only if it still ends there (otherwise the car stopped at a junction for good).
                joined = self.mat_mask(gray, bridge=True)
                if joined is not None:
                    front = joined[:, max(cx0 - FRAME_W // 12, 0):cx0 + FRAME_W // 12]
                    rows = np.flatnonzero(front.mean(axis=1) > 128)
                    mat_top = int(FRAME_H * ROI_TOP) + (rows[0] if len(rows) else front.shape[0])
            self.edge_hits = self.edge_hits + 1 if mat_top > FRAME_H * EDGE_STOP_Y else 0
            self.edge = self.edge_hits >= 3     # Only after 3 frames in a row, so one bad frame does not stop the car.

        if WHITE_THR is not None:
            self.thr = WHITE_THR
        else:
            # Automatic: half the brightest line pixels (top 1%), then follow that target slowly from frame to frame.
            peak = percentile_u8(top, 99)
            target = max(WHITE_MIN, 0.5 * peak)
            self.auto_thr = target if self.auto_thr is None else 0.9 * self.auto_thr + 0.1 * target
            self.thr = max(WHITE_MIN, self.auto_thr)
        # top >= thr, as 0 / 255. For whole-number pixels that is top > ceil(thr) - 1, which cv2.threshold does in one pass.
        mask = cv2.threshold(top, int(np.ceil(self.thr)) - 1, 255, cv2.THRESH_BINARY)[1]
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_K3)

    def far_hline(self, frame):
        """Nearest (lowest) y of a horizontal line from FAR_TOP to FAR_BOT. None if there is none.

        Used to start slow= braking early. Brightness uses this frame's ROI threshold, self.thr.
        """
        top, bot = int(FRAME_H * FAR_TOP), int(FRAME_H * FAR_BOT)
        if bot - top < 4:
            return None
        gray = cv2.GaussianBlur(cv2.cvtColor(frame[top:bot], cv2.COLOR_BGR2GRAY), (5, 5), 0)
        half = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
        mask = tophat_ellipse(half, TOPHAT_K) >= self.thr
        hits = 0
        for r in range(mask.shape[0] - 1, -1, -1):      # From the bottom (the near side).
            row = np.concatenate(([0], mask[r].astype(np.int8), [0]))
            edges = np.flatnonzero(np.diff(row))
            longest = (edges[1::2] - edges[::2]).max() * 2 if len(edges) else 0
            y = top + 2 * r
            need = FAR_HLINE * LANE_WIDTH_PX * (y - HORIZON_Y) / (max(342 - HORIZON_Y, 1.0))
            hits = hits + 1 if longest >= need else 0
            if hits >= 2:                               # At least two rows in a row, so noise is excluded.
                return y + 2
        return None

    @staticmethod
    def find_lines(band, between=None):
        """One band: vertical-line x positions, a horizontal-line candidate (start, end) or None, and a crosswalk flag.

        between: (left line x, right line x) from the last frame, if both are known. A crosswalk then needs CROSSWALK_INSIDE
        stripes inside them, so the two lane lines plus a reflection or two between them are not a crosswalk.
        """
        cols = band.sum(axis=0).astype(np.float64)
        on = (cols > band.shape[0] * 0.2 * 255).astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], on, [0]))))
        runs, hline = [], None
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= FRAME_W * HLINE_RATIO:
                hline = (a, b)
            elif LINE_W_MIN <= b - a <= LINE_W_MAX and cols[a:b].sum() >= MIN_PIXELS * 255:
                runs.append([a, b])
        if len(runs) >= CROSSWALK_LINES:
            # A crosswalk is a tight repeat of stripes of similar width. Lane lines sit far apart.
            widths = np.array([b - a for a, b in runs], dtype=float)
            gaps = np.array([runs[k + 1][0] - runs[k][1] for k in range(len(runs) - 1)], dtype=float)
            med = np.median(widths)
            if np.all((widths > 0.5 * med) & (widths < 2 * med)) and np.median(gaps) < 2.5 * med:
                if between is None:
                    return [], hline, True
                lo, hi = between[0] + LINE_W_MAX / 2, between[1] - LINE_W_MAX / 2
                if sum(lo < (a + b) / 2 < hi for a, b in runs) >= CROSSWALK_INSIDE:
                    return [], hline, True
        merged = []     # Double lines become one.
        for a, b in runs:
            if merged and a - merged[-1][1] < DOUBLE_GAP:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        # Brightness-weighted center of each run. The arithmetic of np.average(x, weights=w), which spends about 0.1 ms
        # per call on the Pi in argument checks: sum(x * w) / sum(w) in float64.
        lines = []
        for a, b in merged:
            w = cols[a:b] + 1e-6
            lines.append(float(np.multiply(_COLUMNS[a:b], w, dtype=np.float64).sum() / w.sum(dtype=np.float64)))
        return lines, hline, False

    @staticmethod
    def nearest(x, prev):
        return abs(x - prev) if prev is not None else 1e9

    def assign(self, i, lines, side, ref):
        """Pick this band's left and right lane lines. Returns (left, right, line held back for a later decision)."""
        if not lines:
            return None, None, None
        if side == "left":
            return min(lines), None, None
        if side == "right":
            return None, max(lines), None
        lefts = [x for x in lines if x < ref]
        rights = [x for x in lines if x >= ref]
        l = max(lefts) if lefts else None
        r = min(rights) if rights else None
        w = self.lane_width(i)
        pl, pr = self.left[i], self.right[i]
        if l is not None and r is not None:
            if not 0.6 * w <= r - l <= 1.5 * w:
                # If the width is implausible, trust only the line closer to the previous position.
                # With no previous position (just after a turn) keep the longer one: the lane line, not a short object off the track.
                dl, dr = self.nearest(l, pl), self.nearest(r, pr)
                if dl == dr:
                    dl, dr = -self.blob_height(i, l), -self.blob_height(i, r)
                if dl <= dr:
                    r = None
                else:
                    l = None
        if l is None or r is None:
            # One visible line: if it clearly continues the previous left or right line, use that side.
            # Otherwise hold the decision and settle it from the whole frame (see __call__).
            x = l if l is not None else r
            dl, dr = self.nearest(x, pl), self.nearest(x, pr)
            if min(dl, dr) >= w * 0.25:
                return None, None, x
            l, r = (x, None) if dl <= dr else (None, x)
        return l, r, None

    def find_gap(self, bands, ls, rs):
        """Y of the last band where the lane is still visible ahead, or None if it does not end. ls and rs are unused and kept for compatibility.

        A gap is a band with no lines after lines have continued up from the bottom.
          - One empty band is a gap if the line resumes above it. A short break on the track is about one band on screen.
          - An empty top band with no resume is not a gap. The far line may just be too faint. A gap then needs at least GAP_BANDS empty bands.
        False-hit guards:
          - The band just before the gap and the band below it, taken together, must show both a left and a right line. The two sides may end at different heights.
          - A line in the band just before the gap that sits at the frame edge is not a gap. On a sharp curve the line leaves the side of the frame.
          - Anything in the outer 5% of the frame is not counted as a line. Those are objects off the floor.
          - A mat edge, an empty band outside the mat, or a horizontal-line or crosswalk band is not a gap.
        """
        self.gap_why = ""      # Why this is not a gap (diagnostics: gap_probe.py and the view).
        inner = [[x for x in lines if 0.05 * FRAME_W <= x <= 0.95 * FRAME_W] for _, lines, _ in bands]
        self.near_line = bool(inner[0])
        if self.edge or any(special for _, _, special in bands):
            self.gap_why = "mat edge" if self.edge else "junction/crosswalk"
            return None
        if not inner[0]:
            self.gap_why = "no near line"
            return None
        k = 0
        while k + 1 < len(bands) and inner[k + 1]:      # Last band whose lines still continue up from the bottom.
            k += 1
        if k == len(bands) - 1:
            self.gap_why = "lines continue to top"
            return None
        e = 0
        while k + 1 + e < len(bands) and not inner[k + 1 + e]:
            e += 1
        resumed = k + 1 + e < len(bands)
        if not resumed and e < GAP_BANDS:
            self.gap_why = "only far band empty"
            return None
        if any(not 0.08 * FRAME_W <= x <= 0.92 * FRAME_W for x in inner[k]):
            self.gap_why = f"band{k}: line at edge (curve)"
            return None
        cx0 = FRAME_W * CAMERA_X
        near = [x for j in range(max(0, k - 1), k + 1) for x in inner[j]]
        if not any(x < cx0 for x in near) or not any(x >= cx0 for x in near):
            self.gap_why = f"band{k}: one side only"
            return None
        if USE_MAT_MASK and any(bands[j][0] < self.mat_top for j in range(k + 1, k + 1 + e)):
            self.gap_why = "empty area outside mat"
            return None     # An empty area outside the mat is the mat ending, not a lane gap.
        return int(bands[k][0])

    def hline_edge(self):
        """Exact y of the near edge of the nearest horizontal line, in front of the car. None if there is none or it reached the frame bottom.

        Unlike junction_y (a band center, 36 px steps), this resolves about one pixel, for cm=.
        It reads the columns ahead of the car, or those of the band test's line when that line does not cross them (a corner off to one side).
        """
        cx0 = int(FRAME_W * CAMERA_X)
        strip = self.mask[:, max(cx0 - 60, 0):cx0 + 60] > 0
        rows = np.flatnonzero(strip.mean(axis=1) > 0.6)
        if len(rows):
            row = rows[-1]
        elif self.hrun is not None:
            strip, row = self.mask[:, self.hrun[0]:self.hrun[1]] > 0, self.hrun[2]
        else:
            return None
        r0, r1 = max(row - 20, 0), min(row + 20, strip.shape[0])
        near = strip[r0:r1]
        cols = near.any(axis=0)
        if not cols.any():
            return None
        low = near.shape[0] - 1 - np.argmax(near[::-1], axis=0)     # Lowest line pixel per column. The median reads a slanted line at its middle.
        y = r0 + float(np.median(low[cols]))
        return None if y >= strip.shape[0] - 2 else int(FRAME_H * ROI_TOP) + y + 1

    def walk_line(self, side, mask, top):
        """Follow one lane line up the view, pixel by pixel, so it bends with a curve. Returns {row: x} of where it was found, and the straight fit."""
        xs = self.left if side == "left" else self.right
        pts = [(self.band_y[i] - top, xs[i]) for i in range(SLICES) if xs[i] is not None]
        if len(pts) < 2:
            return None, None
        slope, base = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
        rows_n, cols_n = mask.shape
        width = lambda r: LANE_WIDTH_PX * (r + top - HORIZON_Y) / (max(342 - HORIZON_Y, 1.0))
        row0 = int(max(p[0] for p in pts))
        cs = self._row_sums         # Made by lane_end for this frame's mask.
        if self._row_xsums is None:     # Same, weighted by column: for the mean column in a window. Once per frame, when needed.
            self._row_xsums = np.zeros((rows_n, cols_n + 1), np.int32)
            np.cumsum(mask * np.arange(cols_n, dtype=np.int32), axis=1, out=self._row_xsums[:, 1:])
        csx = self._row_xsums
        step = 2        # Every other row: plenty for a lane line, and half the cost.
        x, dx, path, miss = slope * row0 + base, -slope * step, {}, 0
        drift = LANEEND_DRIFT * step
        for row in range(row0, -1, -step):
            if not 0 <= x < cols_n:
                break
            tol = max(4.0, LANEEND_TOL * width(row))
            lo = max(int(x - tol), 0)
            hi = min(int(x + tol) + 1, cols_n)
            count = int(cs[row, hi] - cs[row, lo]) if hi > lo else 0
            if count:
                # Mean column of the line pixels in the window, from row prefix sums (no per-row array): same value as before.
                mean = float(np.float64(int(csx[row, hi] - csx[row, lo]) - lo * count) / count)
                shift = min(max(lo + mean - x, -drift), drift)
                x += shift
                dx = 0.7 * dx - 0.3 * shift
                path[row], miss = x, 0
            else:
                miss += step
                if miss > 3 * LANEEND_GAP:
                    break
            x -= dx
        return path, (slope, base)

    def lane_end(self):
        """Where the lane's own line ends ahead, at a junction: the y of its top, of the nearer side if both end. None if neither ends in view.

        Each lane line is walked up from the near bands along where it runs, until it stops. At an inside corner it stops where it turns off
        sideways; on the outside of an L it goes on to the line across, farther, so the nearer end is the junction's near edge either way.
        A line that bends on (a curve) is not an end: its pixels go on above, near where it would be.
        A line that leaves the straight path while the other line bends the same way is a curve of the whole lane, not an end either.
        At a side road only one line turns off, and the other goes on straight.
        """
        self.end_y = self.end_side = None
        mask = self.mask > 0
        top = int(FRAME_H * ROI_TOP)
        rows_n, cols_n = mask.shape
        # Row prefix sums: "any line pixel between columns a and b on this row" becomes one subtraction.
        # cv2.integral sums down the rows as well; the difference of two rows of it is that row alone (the same integers
        # as a cumsum along the row, about 10x faster).
        cs = np.diff(cv2.integral(mask.view(np.uint8)), axis=0)
        hit = lambda r, a, b: cs[r, min(max(b, 0), cols_n)] > cs[r, min(max(a, 0), cols_n)]
        self._row_sums = cs
        self._row_xsums = None      # Made by walk_line only if it is needed this frame.
        ends = []
        walks = {}
        for side, xs in (("left", self.left), ("right", self.right)):
            pts = [(self.band_y[i] - top, xs[i]) for i in range(SLICES) if xs[i] is not None]
            if len(pts) < 2:
                continue
            slope, base = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
            width = lambda r: LANE_WIDTH_PX * (r + top - HORIZON_Y) / (max(342 - HORIZON_Y, 1.0))
            last, miss = None, 0
            for row in range(int(max(p[0] for p in pts)), -1, -1):
                x, tol = slope * row + base, max(4.0, LANEEND_TOL * width(row))
                if 0 <= x < cols_n and hit(row, int(x - tol), int(x + tol) + 1):
                    last, miss = row, 0
                else:
                    miss += 1
                    if miss > LANEEND_GAP:
                        break
            if last is None or last <= LANEEND_GAP or last >= rows_n - 2:
                continue        # Runs on to the top of the view, or has already left by the bottom.
            if sum(p[0] > last + LANEEND_GAP for p in pts) < 2:
                continue        # Not seen running up to that end over two bands: two different lines joined by the fit, not one that ends.
            above = range(max(last - 30, 0), max(last - LANEEND_GAP, 0))
            bends = [hit(r, int(slope * r + base - LANEEND_CURVE * width(r)), int(slope * r + base + LANEEND_CURVE * width(r)) + 1)
                     for r in above]
            if bends and np.mean(bends) > 0.3:
                continue
            curve = self.lane_bends(side, last, mask, top, walks)
            now = self.now if self.now is not None else 0.0
            if curve:
                self.curve_t = now
                continue        # The whole lane curves here: both lines bend the same way.
            if curve is None and self.curve_t is not None and 0 <= now - self.curve_t <= LANEEND_CURVE_HOLD:
                continue        # The other line is out of sight, and the lane was curving a moment ago: still the curve.
            ends.append((last, side, curve))
        # The nearer end first. One the tracker shows running on is dropped, and the other side's end still counts.
        for last, side, curve in sorted(ends, reverse=True):
            if not self.end_continues(top, last, side, curve):
                self.end_y, self.end_side = top + last, side
                break
        return self.end_y

    def end_continues(self, top, last, side, curve):
        """Whether a lane end the other line could not confirm (curve None) is false: the ending line runs on along the tracked lane.

        lane_bends needs the other line. With it out of sight the end stands, and a curve is read as the lane ending.
        The line has to still lie on the tracked lane above the end, including the rows just above it where a curve
        has left the straight fit but not the lane (runs 20261110-103523 frame 109, 183806 frame 132)."""
        t = self.tracker
        if not JUNCTION_TRACK_ASSIST or curve is not None or t is None or not t.ready(self.assist_turn):
            return False
        import lane_track
        mask = self.mask > 0
        which = -1 if side == "left" else 1

        def along(start, stop, tol_share, need, least):
            hit = rows = 0
            for dy in range(start, stop, 2):
                yy = last - dy
                if yy < 0:
                    break
                s = top + yy - HORIZON_Y
                if s <= 4:
                    break
                rows += 1
                left, right, _ = t.lines_at(t.p, s)
                x = left if which < 0 else right
                tol = max(4.0, tol_share * t.p[2] * s)
                if mask[yy, max(int(x - tol), 0):min(int(x + tol) + 1, mask.shape[1])].any():
                    hit += 1
            return rows >= least and hit / rows >= need

        return along(2, 16, 0.35, lane_track.TRACK_END_NEAR, 6) or along(6, 36, 0.2, lane_track.TRACK_END_CONTINUE, 8)

    def sharp_bend(self, top):
        """Where a lane line turns sharply: (frame y, side, degrees, frame x) of the nearest such turn, or None.

        Each lane line is followed up the view as it runs (walk_line, which bends with a curve). A curve is followed to the
        top of the view or out of its side, so it never turns sharply here. Where the following stops inside the view, the
        rest of the line's own piece of tape near that point is compared with the line's direction just before it, on the
        floor (perspective makes far angles look flatter), in lane widths. An L corner, a T or a side road gives about 90.
        Where the line only breaks (two pieces of tape joined slightly apart) and runs on just ahead, the following carries
        on past the break (_bridge), so the corner beyond it is found and the joint is not taken for the line's end.
        """
        labels, stats = self.blobs
        mask = self.mask > 0
        if labels.shape[0] < 2 or labels.shape != mask.shape:
            return None
        # walk_line's row sums of this frame's mask (lane_end makes them again later in the frame, the same).
        self._row_sums, self._row_xsums = np.diff(cv2.integral(mask.view(np.uint8)), axis=0), None
        H, cx = HORIZON_Y, FRAME_W * CAMERA_X
        k0 = max(342 - H, 1.0) / LANE_WIDTH_PX      # Floor lane widths per px across, one row below the horizon.
        floor = lambda x, row: ((np.asarray(x, float) - cx) * k0 / np.maximum(np.asarray(row, float) + top - H, 1.0),
                                CAMERA_F * k0 / np.maximum(np.asarray(row, float) + top - H, 1.0))
        best = None
        for side, xs in (("left", self.left), ("right", self.right)):
            path, _ = self.walk_line(side, mask, top)
            if not path or len(path) < 4:
                continue
            rows = np.array(sorted(path), dtype=float)          # Far (small row) to near.
            px = np.array([path[r] for r in rows])
            for _ in range(BRIDGE_MAX + 1):
                if rows[0] <= 4 or px[0] < 2 or px[0] > mask.shape[1] - 3:
                    break       # Followed to the top of the view, or out of its side (a curve going out of sight): it runs on.
                deg = self._turn_at(rows, px, mask, labels, stats, floor)
                if deg is not None:
                    y = int(rows[0] + top)
                    if best is None or y > best[0]:
                        best = (y, side, deg, float(px[0]))
                    break
                more = self._bridge(rows, px, mask, top, floor)
                if more is None:
                    break       # Ends here and does not turn: a lane end or a gap, not a corner.
                rows, px = more
        return best

    def junction_point(self):
        """go(cm=)'s junction point this frame (frame y), or None: where a lane line turns sharply (sharp_bend), else where
        a lane line ends (lane_end). The turn wins: a joint in the tape can stop a line short of the junction, and the corner
        is where the junction really is."""
        end = self.lane_end()
        if self.bend is not None:
            self.end_y, self.end_side = self.bend[0], self.bend[1]      # What the view and the ruler show as the lane end.
            return self.bend[0]
        return end

    def _turn_at(self, rows, px, mask, labels, stats, floor):
        """sharp_bend: how sharply the line's tape turns away at the end of the followed path (rows far to near), or None."""
        fx, fz = floor(px, rows)
        # The line's direction just before its end: the stretch within 0.3 lane widths (on the floor) of it.
        near_end = np.hypot(fx - fx[0], fz - fz[0]) <= 0.3
        if near_end.sum() < 3:
            return None
        dz, dx = fz[0] - fz[near_end][-1], fx[0] - fx[near_end][-1]
        if np.hypot(dx, dz) < 0.05:
            return None
        dirx, dirz = dx / np.hypot(dx, dz), dz / np.hypot(dx, dz)
        # The line's piece of tape at its end.
        r_e, x_e = int(rows[0]), int(np.clip(px[0], 0, mask.shape[1] - 1))
        near = labels[max(r_e - 3, 0):r_e + 4, max(x_e - 4, 0):x_e + 5]
        ids = near[near > 0]
        if not len(ids):
            return None
        lab = int(np.bincount(ids).argmax())
        bx, by_, bw, bh, _ = stats[lab]
        yy, xx = np.nonzero(labels[by_:by_ + bh:2, bx:bx + bw:2] == lab)
        yy, xx = yy * 2 + by_, xx * 2 + bx
        if len(yy) < BEND_MIN_PIX:
            return None
        gx, gz = floor(xx, yy)
        rx, rz = gx - fx[0], gz - fz[0]
        t = rx * dirx + rz * dirz                       # Ahead along the line's own direction at its end.
        u = rx * dirz - rz * dirx                       # Sideways from it.
        off = (np.abs(u) > BEND_OFF_LW) & (np.hypot(t, u) < BEND_REACH_LW) & (t > -BEND_ON_LW)
        if off.sum() < BEND_MIN_PIX:
            return None
        # The turned-away part must be a line: thin across its own direction. A crosswalk block or a blob joined to the
        # line is thick (the smaller spread of its pixels is far wider than a line).
        thick = float(np.sqrt(max(np.linalg.eigvalsh(np.cov(np.stack([t[off], u[off]])))[0], 0.0)))
        if thick > BEND_THICK_LW:
            return None
        if np.ptp(np.abs(u[off])) < BEND_RUN_LW and not ((xx[off] <= 3) | (xx[off] >= mask.shape[1] - 4)).any():
            return None     # It must run on sideways (or out of the view), not be a corner of something else poking past the line.
        deg = float(np.degrees(np.arctan2(np.abs(u[off]).mean(), np.maximum(t[off], 0.0).mean())))
        return deg if deg >= JUNCTION_BEND_DEG else None

    def _bridge(self, rows, px, mask, top, floor):
        """sharp_bend: past a short break in the line (a tape joint), where it runs on just ahead along its own direction.

        Returns the path with the piece beyond added (far to near), or None when nothing lies within BRIDGE_LW ahead and
        BRIDGE_SIDE_LW sideways of where the line was heading (a real end, or a gap at a junction, which is far longer)."""
        H = HORIZON_Y
        lane_px = lambda r: LANE_WIDTH_PX * (r + top - H) / max(342 - H, 1.0)
        end_r, end_x = rows[0], px[0]
        sel = rows <= end_r + 24
        slope = float(np.polyfit(rows[sel], px[sel], 1)[0]) if sel.sum() >= 3 else 0.0     # px across per row, near the end.
        z_end = floor(end_x, end_r)[1]
        found = None
        for r in range(int(end_r) - 1, -1, -1):
            if floor(end_x, r)[1] - z_end > BRIDGE_LW:
                break
            x_pred = end_x + slope * (r - end_r)
            tol = max(4.0, BRIDGE_SIDE_LW * lane_px(r))
            lo, hi = max(int(x_pred - tol), 0), min(int(x_pred + tol) + 1, mask.shape[1])
            hit = np.flatnonzero(mask[r, lo:hi]) if hi > lo else []
            if len(hit):
                found = (r, lo + float(hit.mean()))
                break
        if found is None:
            return None
        # Follow the piece beyond up the view, the way walk_line does (a window that drifts with the line).
        r, x = found
        new, miss = [], 0
        while r >= 0 and miss <= LANEEND_GAP:
            tol = max(4.0, LANEEND_TOL * lane_px(r))
            lo, hi = max(int(x - tol), 0), min(int(x + tol) + 1, mask.shape[1])
            hit = np.flatnonzero(mask[r, lo:hi]) if hi > lo else []
            if len(hit):
                x = lo + float(hit.mean())
                new.append((r, x))
                miss = 0
            else:
                miss += 1
            r -= 1
        if len(new) < 4:
            return None
        new.reverse()       # Far to near, like rows.
        return np.concatenate([[q[0] for q in new], rows]), np.concatenate([[q[1] for q in new], px])

    def lane_bends(self, side, last, mask, top, walks):
        """Whether the whole lane curves where this side's line left its straight path (row last): the other line bends the same way.

        Returns True (a curve), False (the other line goes on straight: a side road), or None (the other line is not seen that high).
        Measured as how far the other line has moved off its own straight path, outward on this side, at rows above last.
        """
        other = "right" if side == "left" else "left"
        if other not in walks:
            walks[other] = self.walk_line(other, mask, top)
        path, fit = walks[other]
        if not path or fit is None:
            return None
        outward = -1.0 if side == "left" else 1.0
        dev = lambda q: (path[q] - (fit[0] * q + fit[1])) * outward
        at = [q for q in path if last - 3 <= q <= last + 3]
        high = [q for q in path if q <= last - LANEEND_BEND_ROWS // 2]
        if not at or not high:
            return None
        q0, q1 = min(at, key=lambda q: abs(q - last)), max(min(high), last - LANEEND_BEND_ROWS)
        q1 = min(high, key=lambda q: abs(q - q1))
        return dev(q1) >= LANEEND_BEND_PX or dev(q1) - dev(q0) >= LANEEND_BEND_GROW

    def lane_tip(self):
        """y where the lane ends ahead: halfway between the tips of the left and right lines. None if the lane does not end in view.

        Halfway does not depend on which line ends first, which a slightly turned car changes. A line must sit about where a lane line does at
        that height, so other tape near the frame side does not count. Once one tip has left the frame, the other one stands in, by the distance they had.
        """
        labels, stats = self.blobs
        cx0 = FRAME_W * CAMERA_X
        top_row = int(FRAME_H * ROI_TOP)
        tips = {}
        for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
            # Comes from the near half of the view or the frame side (a near line can be cut off there), and stops below the ROI top.
            # A piece that lies wholly in the far half is the lane resuming beyond a gap, not the end of this one.
            ends_in_view = 0 < y and (y + h >= self.mask.shape[0] / 2 or x <= 0 or x + w >= FRAME_W)
            if not (ends_in_view and h >= 6 and area >= MIN_PIXELS):
                continue
            top = np.flatnonzero(labels[y] == i)
            if not len(top):
                continue
            tip_x = float(top.mean())
            lane_w = LANE_WIDTH_PX * (top_row + y - HORIZON_Y) / (max(342 - HORIZON_Y, 1.0))
            # At the frame side, it is a line leaving by the side (a curve), not one that ends.
            if 0.08 * FRAME_W <= tip_x <= 0.92 * FRAME_W and 0.2 * lane_w <= abs(tip_x - cx0) <= 0.85 * lane_w:
                side = tip_x < cx0
                tips[side] = max(tips.get(side, 0), int(y))
        if len(tips) == 2:
            self.tip_diff = abs(tips[True] - tips[False])
            return top_row + (tips[True] + tips[False]) / 2
        if not tips:
            self.tip_diff = None
        if len(tips) == 1 and self.tip_diff is not None:
            return top_row + next(iter(tips.values())) + self.tip_diff / 2
        return None

    def follow_gap(self, bands):
        """Keep a confirmed gap in gap_y while the band test cannot see it but its line tips are still in view."""
        self.tip_y = self.lane_tip()
        if self.gap_y is not None and self.tip_y is None:
            # A lane that really ends shows where its lines end. A curved line after a turn can look like a gap to the bands (run 20261104-230540, frames 979-984).
            self.gap_y, self.gap_why = None, "no line tips"
        self.tip_lost = 0 if self.tip_y is not None else self.tip_lost + 1
        if self.edge or any(special for _, _, special in bands) or self.tip_lost >= GAP_TIP_LOST:
            self.gap_hits, self.gap_held = 0, False
        elif self.gap_y is not None:
            # Counted over the whole time the tips are followed, not in a row: uneven tips let the band test see the gap only in short spells.
            self.gap_hits += 1
            self.gap_held = self.gap_hits >= JUNCTION_FRAMES
        if self.gap_y is None and self.gap_held and self.tip_y is not None:
            self.gap_y = int(self.tip_y)
            self.gap_why = "held by line tips"

    def __call__(self, frame, side="center", draw=True):
        """Return ((position error, heading error) or None, junction detected, debug image).

        draw=False skips the debug drawing. Detection is unchanged, and the debug image is None, which is faster when the view is not being sent.
        """
        self._sharp_cache = {}      # This frame's gray and edge pictures, shared by every blob_sharpness() call inside it.
        self.band_line = None
        try:
            det, junction, dbg = self._detect(frame, side, draw)
        finally:
            self._sharp_cache = None
        self.det_bands = det
        if LANE_MODEL == "track" or JUNCTION_TRACK_ASSIST:
            if self.tracker is None:
                from lane_track import LaneTracker
                self.tracker = LaneTracker()
            self.tracker.use_far = LANE_MODEL == "track"     # Steering by it needs the far lines; the junction check does not.
            if side != "center":        # side= (follow one line only) is the band method's job: the tracker starts again after it.
                self.tracker.reset()
            elif LANE_MODEL == "track":
                det = self.tracker.update(sys.modules[__name__], self, frame, self.now)
                if dbg is not None:
                    self.draw_track(dbg)
            else:
                # Junction and lane-end checks only: the band reading stands. A fault here must not cost the drive a frame:
                # the tracker starts again and the checks are off until it holds the lane again.
                try:
                    self.tracker.update(sys.modules[__name__], self, frame, self.now)
                except Exception as e:
                    self.tracker.reset()
                    if not getattr(self, "_assist_warned", False):
                        self._assist_warned = True
                        print(f"\n[warning] junction check skipped a frame: {type(e).__name__}: {e}")
        if LANE_MODEL != "track" and LANE_POINTS == "pairs":
            det = self.pair_reading(det, side, dbg)
        if LANE_MODEL != "track":
            det = self.confirm_jump(det, side)
        return det, junction, dbg

    def pair_reading(self, det, side, dbg):
        """LANE_POINTS = "pairs": the lane position and heading from both lines fitted as one pair (lane_pairs.py).
        Lost stays lost (the bands decide that), and a turn or one line only (side=) keeps the band answer."""
        if self.pairs is None:
            from lane_pairs import LanePairs
            self.pairs = LanePairs()
        pairs = self.pairs
        top = int(FRAME_H * ROI_TOP)
        h = (FRAME_H - top) // SLICES
        if det is None or side != "center" or self.assist_turn:
            pairs.reset()
            pairs.used = False
        else:
            sure = sum(l is not None and r is not None for l, r in zip(self.left, self.right)) >= 3
            det = pairs(sys.modules[__name__], det, self.mask, top, FRAME_H - h // 2, top + h // 2, int(FRAME_W * CAMERA_X),
                        self.band_line, sure)
        if dbg is not None:
            self.draw_pairs(dbg, top, h)
        return det

    def draw_pairs(self, dbg, top, h):
        """The pair fit: its pieces (left out: grey), both lines and the center. The band path when the fit was not used."""
        pairs = self.pairs
        if pairs.used:
            lf_ = sys.modules[__name__]
            left, right = pairs.curve_points(lf_, -1, top), pairs.curve_points(lf_, 1, top)   # As the lines really run.
            if LANE_FILL > 0:     # The lane the car follows, lightly filled over the whole band area.
                yy = np.linspace(FRAME_H - 1, top, 24)
                edge = []
                for which, pts in ((-1, left), (1, right)):
                    # The drawn line where it is drawn; above its end, the pair line, moved to meet it there.
                    line = pairs.line_points(lf_, which, top)
                    x_line = np.interp(yy, line[::-1, 1], line[::-1, 0])
                    end = pts[:, 1].min()
                    shift = np.interp(end, pts[::-1, 1], pts[::-1, 0]) - np.interp(end, line[::-1, 1], line[::-1, 0])
                    x = np.where(yy >= end, np.interp(yy, pts[::-1, 1], pts[::-1, 0]), x_line + shift)
                    edge.append(np.stack([x, yy], axis=1))
                dbg.fill_poly(np.vstack([edge[0], edge[1][::-1]]), C_PATH, LANE_FILL)
            ys, xs, sides = pairs.pieces
            for y, x, sd in zip(ys[::2], xs[::2], sides[::2]) if VIEW_DETAIL else ():     # Every other piece: the view stays light.
                dbg.circle((int(x), int(y)), 2, C_LEFT if sd < 0 else C_RIGHT if sd > 0 else C_DIM, -1)
            # Both lines as they run; the center is the straight line the car steers by.
            for pts, col, thick in ((left, C_LEFT, 2), (right, C_RIGHT, 2), (pairs.line_points(lf_, 0, top), C_PATH, 3)):
                dbg.polyline(pts, (20, 20, 20), thick + 3, cv2.LINE_AA)
                dbg.polyline(pts, col, thick, cv2.LINE_AA)
        elif self.band_line is not None:
            near_y, far_y = FRAME_H - h // 2, top + h // 2
            x_near, x_far = self.band_line
            dbg.line((int(x_near), near_y), (int(x_far), far_y), (20, 20, 20), 6, cv2.LINE_AA)
            dbg.line((int(x_near), near_y), (int(x_far), far_y), C_PATH, 3, cv2.LINE_AA)

    def confirm_jump(self, det, side):
        """A lane position that jumps by more than JUMP_CONFIRM counts only once the next frame agrees; until then the last one stands.

        Over broken lines (a gap, a junction, a crosswalk) one stray point (a stripe, a piece of the next lane's line) can turn
        the whole lane over for a single frame, and the car steered after it. Off during a turn and on one line (side=):
        there the lane really does move fast."""
        last, pending = self.jump_last, self.jump_pending
        if det is None and not self.assist_turn and side == "center":
            # A frame or two with no lane (a gap, glare) keeps the last lane to compare with: otherwise the first reading after
            # it counted whatever it was, a one-frame misread included. A longer loss forgets it.
            self.jump_gone += 1
            if self.jump_gone > JUMP_HOLD_MAX:
                self.jump_last, self.jump_pending, self.jump_held = None, None, False
            return det
        self.jump_gone = 0
        if JUMP_CONFIRM <= 0 or det is None or last is None or self.assist_turn or side != "center" \
                or abs(det[0] - last[0]) <= JUMP_CONFIRM:
            self.jump_last, self.jump_pending, self.jump_held = det, None, False
            return det
        if pending is not None and abs(det[0] - pending[0]) <= JUMP_CONFIRM:
            self.jump_last, self.jump_pending, self.jump_held = det, None, False     # Two frames agree: it is real.
            return det
        self.jump_count = self.jump_count + 1 if self.jump_held else 1
        if self.jump_count > JUMP_HOLD_MAX:      # Still far off and not settling: give it out rather than hold an old lane.
            self.jump_last, self.jump_pending, self.jump_held = det, None, False
            return det
        self.jump_pending, self.jump_held = det, True
        return last

    def draw_track(self, dbg):
        """The tracked lane: both lines and the center, over the whole height it was fitted on."""
        t = self.tracker
        if t is None or t.p is None:
            return
        color = C_PATH if t.conf >= 0.5 else C_AT
        for which, col, thick in ((-1, C_LEFT, 2), (1, C_RIGHT, 2), (0, color, 3)):
            pts = t.curve_points(sys.modules[__name__], which)
            if pts is not None:
                dbg.polyline(pts, (20, 20, 20), thick + 3, cv2.LINE_AA)
                dbg.polyline(pts, col, thick, cv2.LINE_AA)
        relock = t.relocked_at is not None and self.now is not None and self.now - t.relocked_at < 0.5
        chip(dbg, "TRACK RE-LOCK" if relock else f"TRACK {t.conf:.0%}", FRAME_W - 130, FRAME_H - 30, C_AT if relock else color, 0.4)

    def _detect(self, frame, side="center", draw=True):
        """Return ((position error, heading error) or None, junction detected, debug image).

        draw=False skips the debug drawing. Detection is unchanged, and the debug image is None, which is faster when the view is not being sent.
        """
        top = int(FRAME_H * ROI_TOP)
        mask = self.line_mask(frame[top:])
        self.signs = self.sign_area(frame[top:])
        if self.signs is not None:      # A red-rimmed sign on the floor: its white face and numbers are not lines or stripes.
            mask[self.signs] = 0
        h = mask.shape[0] // SLICES
        comps = self.drop_reflections(frame[top:], mask)     # Once per frame, shared below.
        self.roi = frame[top:]
        reg_ids = self.find_registered_crosswalk(comps, top)
        stripe_ids, self.stripes = self.find_stripes(comps, mask.shape, top, frame[top:])
        gone = stripe_ids + reg_ids + self.sign_ids + self.find_specks(comps, mask.shape, h)
        labels, stats = comps[1], comps[2]
        if gone:        # Clear crosswalk stripes and off-track specks from the mask, and from the blob labels assign() reads.
            self.clear_blobs(mask, labels, stats, gone)
        self.blobs, self.band_h = (labels, stats), h
        self.mask = mask

        dbg = Canvas(frame) if draw else None   # Drawing commands only: the browser paints them (overlay.py).
        if draw:
            dbg.tint(top, mask > 0, LANE_TINT, C_LANE, VIEW_TINT)   # White-line pixels translucent, so the original image still shows.
            dashed_hline(dbg, top, C_DIM)       # Only below this line is used for lane detection.
            for bx, by, bw, bh in self.reflect_boxes if VIEW_DETAIL else ():      # Light reflections left out of the lane lines.
                dbg.rectangle((bx, top + by), (bx + bw, top + by + bh), C_DIM, 1)
            for bx, by, bw, bh in self.xwalk2_boxes:       # Stripes of the registered crosswalk.
                dbg.rectangle((bx, top + by), (bx + bw, top + by + bh), C_AT, 2)
        cx0 = int(FRAME_W * CAMERA_X)
        jy = int(FRAME_H * JUNCTION_Y)
        marks = {}                          # Junction and crosswalk labels, one per kind, drawn on top at the end.
        if draw and self.xwalk2_y is not None:
            marks["_CROSSWALK"] = (C_AT, self.xwalk2_y)

        junction = False
        self.crosswalk = self.crosswalk_any = False
        self.junction_y = self.crosswalk_y = None   # Lowest (nearest) band y of a horizontal line or crosswalk. Used by course at=.
        self.bend = None                            # sharp_bend: the nearest sharp turn of a lane line this frame, or None.
        if self.stripes:
            self.crosswalk_any = True
            self.crosswalk_y = max(y + sh // 2 for _, y, _, sh in self.stripes)    # Near edge of the nearest stripe.
            self.crosswalk = self.crosswalk_y >= jy
            if draw:
                for sx, sy, sw, sh in self.stripes:
                    dbg.rectangle((sx - sw // 2, sy - sh // 2), (sx + sw // 2, sy + sh // 2), C_XWALK, 2 if VIEW_DETAIL else 1)
                marks.setdefault("CROSSWALK", (C_XWALK, self.crosswalk_y))
        ys, ls, rs, pending = [], [], [], []
        bands = []      # Per band: (y, line x list, horizontal line or crosswalk). Used for the lane-gap test.
        ref = self.center[0] if self.center[0] is not None else cx0
        self.hrun = None
        self.end_y = self.end_side = None          # lane_end, likewise.
        for i in range(SLICES):     # Band 0 is the nearest. Tracking walks from the bottom upward.
            y1 = mask.shape[0] - (i + 1) * h
            cy = top + y1 + h // 2
            # The lane lines at this height in the last frame, or at the nearest height where both were seen.
            both = [k for k in range(SLICES) if self.left[k] is not None and self.right[k] is not None]
            k = min(both, key=lambda k: abs(k - i)) if both else None
            prev = (self.left[k], self.right[k]) if k is not None and abs(k - i) <= 1 else None
            lines, hrun, crosswalk = self.find_lines(mask[y1:y1 + h], prev)
            if hrun is not None and prev is not None:
                # A line across must cross the lane in front of the car. A run off to one side is a lane line bending flat
                # (a curve seen from inside it), not a junction.
                inside = min(hrun[1], prev[1]) - max(hrun[0], prev[0])
                if inside < HLINE_LANE_SHARE * (prev[1] - prev[0]):
                    hrun = None
            # The lane-end test sees the line as the band method judged it. The assist below only takes lines off the junction.
            gap_hline = hrun is not None
            if hrun is not None and prev is None and JUNCTION_TRACK_ASSIST and self.tracker is not None:
                # No memory of both lines: the same test, using the tracked lane. It only ever drops a line.
                guessed = self.tracker.assist_lane(sys.modules[__name__], cy, self.assist_turn)
                if guessed is not None:
                    inside = min(hrun[1], guessed[1]) - max(hrun[0], guessed[0])
                    if inside < HLINE_LANE_SHARE * (guessed[1] - guessed[0]):
                        hrun = None
            hline = hrun is not None
            bands.append((cy, lines, gap_hline or crosswalk))
            if hline and cy >= jy or (JUNCTION_BY_BRANCH and i < 2 and len(lines) >= 3):
                junction = True
            if hline:
                self.junction_y = max(self.junction_y or 0, cy)
                if self.hrun is None:
                    self.hrun = (int(hrun[0]), int(hrun[1]), y1 + h // 2)
            if crosswalk:
                self.crosswalk_any = True
                self.crosswalk_y = max(self.crosswalk_y or 0, cy)
                if cy >= jy:
                    self.crosswalk = True
            if hline or crosswalk:
                color, name = (C_JUNC, "JUNCTION") if hline else (C_XWALK, "CROSSWALK")
                if draw:
                    dbg.rectangle((0, top + y1 + 2), (5, top + y1 + h - 2), color, -1)   # Marks on both edges.
                    dbg.rectangle((FRAME_W - 6, top + y1 + 2), (FRAME_W - 1, top + y1 + h - 2), color, -1)
                marks.setdefault(name, (color, cy))

            if LINE_MODE == "line":
                l = min(lines, key=lambda x: abs(x - ref)) if lines else None
                r = None
            else:
                l, r, undecided = self.assign(i, lines, side, ref)
                if undecided is not None:
                    pending.append((i, cy, undecided))
            ys.append(cy)
            ls.append(l)
            rs.append(r)
            if LINE_MODE == "line":
                if l is not None:
                    ref = l          # Stay on the one line. Shifting by half a lane would chase the other side.
            elif l is not None and r is not None:
                ref = (l + r) / 2   # The next (higher) band judges left and right from this band's center.
            elif l is not None or r is not None:
                w = self.lane_width(i) / 2
                ref = l + w if l is not None else r - w

        if pending:
            # A held-back line is a left line if extending it to the bottom of the frame lands on the car's left, and a right line if it lands on the right.
            if len(pending) >= 2:
                fit = np.polyfit([p[1] for p in pending], [p[2] for p in pending], 1)
                xb = np.polyval(fit, FRAME_H)
            else:
                xb = pending[0][2]
            for i, _, x in pending:
                if xb < cx0 and ls[i] is None:
                    ls[i] = x
                elif xb >= cx0 and rs[i] is None:
                    rs[i] = x
        self.left, self.right = list(ls), list(rs)

        # A lane line that turns sharply (an L corner, a T, a side road) is a junction even when no line across was seen.
        self.bend = self.sharp_bend(top) if JUNCTION_BEND_DEG and LINE_MODE != "line" else None
        if self.bend is not None:
            by = self.bend[0]
            if by >= jy:
                junction = True
                self.junction_y = max(self.junction_y or 0, by)
            if draw:
                marks.setdefault("JUNCTION", (C_JUNC, by))
                chip(dbg, f"BEND {self.bend[2]:.0f} deg", int(self.bend[3]) - 40, by - 22, C_JUNC, 0.38)

        self.gap_y = self.find_gap(bands, ls, rs)
        self.follow_gap(bands)
        if self.gap_y is not None and draw:
            dashed_hline(dbg, self.gap_y, C_GAP, thick=2)
            chip(dbg, "LANE ENDS", FRAME_W // 2 - 40, self.gap_y - 22, C_GAP)

        pts = []
        for i, (cy, l, r) in enumerate(zip(ys, ls, rs)):
            half_w = self.lane_width(i) / 2
            if LINE_MODE == "line":
                c = l
            elif l is not None and r is not None:
                c = (l + r) / 2
                self.learn_width(i, r - l)
            elif l is not None:     # Lane width is measured horizontally, so shift by half of it as-is.
                c = l + half_w
            elif r is not None:
                c = r - half_w
            else:
                c = None
            self.center[i] = c
            if draw and BAND_VIEW():     # The band points: with the tracker or the pair fit their own lines are drawn instead.
                for x, color in ((l, C_LEFT), (r, C_RIGHT)):
                    if x is not None:
                        dbg.circle((int(x), cy), 5, (20, 20, 20), -1, cv2.LINE_AA)
                        dbg.circle((int(x), cy), 4, color, -1, cv2.LINE_AA)
            if c is not None:
                if draw and BAND_VIEW():
                    dbg.circle((int(c), cy), 3, C_PATH, -1, cv2.LINE_AA)
                pts.append((cy, c))

        if len(pts) == 1 and pts[0][0] < (top + FRAME_H) / 2:
            # One far point alone is not a lane: a dark baseboard or object at the far edge (run 20261103-204527 drove
            # off the mat on one point at the top-left corner, and even counted it as "aligned").
            pts = []
        if not pts:
            self.misses += 1
            if self.misses > 5:     # After a long loss (the car was picked up, for example), clear the tracking history.
                self.left, self.right, self.center = [None] * SLICES, [None] * SLICES, [None] * SLICES
            if draw:
                self.draw_marks(dbg, marks)
            return None, junction, dbg
        self.misses = 0

        # Drop center points far from the others (misdetects): of the lines through each pair of points, keep the one that fits the most points.
        if len(pts) >= 3:
            best = []
            for a in range(len(pts)):
                for b in range(a + 1, len(pts)):
                    (ya, xa), (yb, xb) = pts[a], pts[b]
                    k = (xb - xa) / (yb - ya)
                    inl = [(y, x) for y, x in pts if abs(xa + k * (y - ya) - x) < OUTLIER_PX]
                    if len(inl) > len(best):
                        best = inl
            if len(best) >= 2:
                if draw and BAND_VIEW() and VIEW_DETAIL:
                    for y, x in pts:
                        if (y, x) not in best:
                            dbg.circle((int(x), y), 4, C_DIM, -1, cv2.LINE_AA)    # Point dropped as a misdetect.
                pts = best

        near_y, far_y = FRAME_H - h // 2, top + h // 2
        if len(pts) >= 2:
            fit = np.polyfit([y for y, _ in pts], [x for _, x in pts], 1)
            x_near, x_far = np.polyval(fit, near_y), np.polyval(fit, far_y)
        else:
            x_near = x_far = pts[0][1]
        self.band_line = (x_near, x_far)    # Before the clip below: where the tracker (LANE_MODEL "track") starts from.
        if draw and BAND_VIEW():      # With the tracker or the pair fit, their own lane is drawn instead.
            dbg.line((int(x_near), near_y), (int(x_far), far_y), (20, 20, 20), 6, cv2.LINE_AA)
            dbg.line((int(x_near), near_y), (int(x_far), far_y), C_PATH, 3, cv2.LINE_AA)

        # Error unit: half the frame width times (REF_LANE_PX / LANE_WIDTH_PX). A changed field of view can make the lane look smaller.
        # The same lane departure stays the same value, so the steering gains (KP, KH) and the thresholds stay as they are.
        half = FRAME_W / 2 * LANE_WIDTH_PX / REF_LANE_PX
        offset = float(np.clip((x_near - cx0) / half, -1, 1))
        heading = float(np.clip((x_far - x_near) / half, -1, 1))
        if self.gap_y is not None and abs(heading) > GAP_MAX_HEADING:
            # If the car is badly misaligned with the lane, the lines leave the side of the frame and look like a gap, so this is not treated as a gap.
            # Also drop a confirmation built up during that misalignment, or it would end the action once the car straightens.
            self.gap_y, self.gap_why = None, "car not aligned with lane"
            self.gap_hits, self.gap_held = 0, False
        if draw:
            self.draw_marks(dbg, marks)
        return (offset, heading), junction, dbg

    @staticmethod
    def draw_marks(dbg, marks):
        """Draw junction and crosswalk labels above the other marks, at the nearest band's height."""
        for name, (color, cy) in marks.items():
            chip(dbg, name, 12, cy - 10, color)


class TrafficLight:
    """Lamp color each frame: "red", "green", or None.

    Looks only in the upper part of the frame (LIGHT_Y) and at the left and right edges (LIGHT_SIDE). Two methods are used together:
      1. Highly saturated, bright, nearly square blobs of red (H<10 or >170) or green (H 40-80).
         Green stops at 80 so a cyan obstacle block (H 85-95) is not taken as a green light.
      2. An LED blows out white in the center and keeps color only in a ring, so the ring color is used.
    """

    RING_S_MIN = 60         # Minimum saturation used to read the ring color. The ring is paler than the blob.
    CORE_V_MIN = 230        # Value that counts as a blown-out white LED center.
    CORE_S_MAX = 50         # Maximum saturation for the center. Nearly white only, so a pale colored rim is not pulled into the center.

    def __init__(self):
        self.state = None
        self.boxes = []         # (x, y, w, h, color) for the on-screen display.
        self.floor_signs = None     # (ROI top, mask) of red signs painted on the floor this frame (LaneDetector.signs), set by Driver.
        self.local = []
        self.lamps = {}         # Largest lamp box per color {"red": (x, y, w, h), "green": ...}. Compared by position in wait_green.
        self.mask_vis = None    # Color mask shown by tune_light.py, only when keep_mask=True.
        self.painted = []
        self.keep_mask = False

    @staticmethod
    def regions():
        y0, y1 = int(FRAME_H * LIGHT_Y[0]), int(FRAME_H * LIGHT_Y[1])
        sw = int(FRAME_W * LIGHT_SIDE)
        return [(0, y0, sw, y1), (FRAME_W - sw, y0, FRAME_W, y1)]

    @staticmethod
    def is_red(hue):
        return (hue < 10) | (hue > 170)

    @staticmethod
    def is_green(hue):
        return (hue >= 40) & (hue <= 80)

    @staticmethod
    def lamp_like(w, h, area, area_min):
        """Lamp shape: inside the size range, nearly square, and filling much of its box (a circle or a square)."""
        return area_min <= area <= LIGHT_AREA_MAX and 0.6 <= w / h <= 1.6 and area >= 0.45 * w * h

    def __call__(self, frame):
        self.boxes = []
        self.lamps = {}
        self.local = []         # Spots re-read by color_at (for the on-screen display).
        biggest = {"red": 0, "green": 0}

        signs = self.floor_signs

        def add(color, x, y, w, h, area):
            if signs is not None:       # On a red floor sign (a "30" or "70" ring): paint, not a lamp.
                top, on = signs
                cx, cy = int(x + w / 2), int(y + h / 2) - top
                if 0 <= cy < on.shape[0] and 0 <= cx < on.shape[1] and on[cy, cx]:
                    self.painted.append((x, y, w, h))
                    return
            self.boxes.append((x, y, w, h, color))
            if area > biggest[color]:
                biggest[color] = area
                self.lamps[color] = (int(x), int(y), int(w), int(h))

        if self.keep_mask:
            self.mask_vis = np.zeros_like(frame)
        self.painted = []       # Candidates dropped as part of a big painted area (for the view).

        for x0, y0, x1, y1 in self.regions():
            hue, sat, val = cv2.split(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV))
            strong = (sat >= LIGHT_S_MIN) & (val >= LIGHT_V_MIN)

            def painted(color, x, y, w, h, x0=x0, y0=y0):
                """Whether the candidate is a piece of a much bigger colored area: a sign or marking, not a lamp.

                Looked at in the whole frame around it, so an area cut by the search-region edge still counts whole.
                """
                fx, fy = x0 + x, y0 + y
                reach = int(max(w, h) * (LIGHT_PART + 1))
                cx0, cy0 = max(fx - reach, 0), max(fy - reach, 0)
                cx1, cy1 = min(fx + w + reach, FRAME_W), min(fy + h + reach, FRAME_H)
                ch, cs, cv = cv2.split(cv2.cvtColor(frame[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2HSV))
                loose = (cs >= LIGHT_PART_S_MIN) & (cv >= LIGHT_PART_V_MIN) & (self.is_red(ch) if color == "red" else self.is_green(ch))
                _, lab, st, _ = cv2.connectedComponentsWithStats(loose.astype(np.uint8), connectivity=8)
                pad = max(2, max(w, h) // 2)
                ids = np.unique(lab[max(fy - pad - cy0, 0):fy + h + pad - cy0, max(fx - pad - cx0, 0):fx + w + pad - cx0])
                big = max(w, h) * LIGHT_PART
                if any(st[i][2] > big or st[i][3] > big for i in ids if i > 0):
                    self.painted.append((fx, fy, w, h))
                    return True
                return False

            # 1. Color blobs. A registered light (aicar light) sets the colors to look for, instead of the thresholds.
            model = LIGHT_MODEL
            masks = []
            for color, plain in (("red", strong & self.is_red(hue)), ("green", strong & self.is_green(hue))):
                reg = model.color_mask(color, hue, sat, val) if model is not None else None
                masks.append((color, plain if reg is None else reg))
            for color, mask in masks:
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                if self.keep_mask:
                    self.mask_vis[y0:y1, x0:x1][mask > 0] = (0, 0, 255) if color == "red" else (0, 255, 0)
                _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for x, y, w, h, area in stats[1:]:
                    if model is not None and color in model.colors():
                        # Registered: its size and place, and a filled, roundish blob (a sign's ring is hollow and wide).
                        ok = (0.5 <= w / h <= 2.0 and area >= 0.4 * w * h and model.fits(color, x0 + x, y0 + y, w, h, area, FRAME_W, FRAME_H))
                    else:
                        ok = self.lamp_like(w, h, area, LIGHT_AREA_MIN)
                    if ok and not painted(color, x, y, w, h):
                        add(color, x0 + x, y0 + y, w, h, area)

            # 2. LED with a blown-out white center: color of the surrounding ring.
            core = ((val >= self.CORE_V_MIN) & (sat <= self.CORE_S_MAX)).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if not self.lamp_like(w, h, area, max(LIGHT_AREA_MIN // 2, 4)):
                    continue
                pad = max(2, min(w, h) // 2)
                sy = slice(max(y - pad, 0), y + h + pad)
                sx = slice(max(x - pad, 0), x + w + pad)
                blob = (labels[sy, sx] == i).astype(np.uint8)
                ring = cv2.dilate(blob, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)).astype(bool) & (blob == 0)
                colored = ring & (sat[sy, sx] >= self.RING_S_MIN) & (val[sy, sx] >= LIGHT_V_MIN // 2)
                if colored.sum() < 0.3 * ring.sum():
                    continue
                h_ring = hue[sy, sx][colored]
                color = ("red" if self.is_red(h_ring).mean() > 0.6 else
                         "green" if self.is_green(h_ring).mean() > 0.6 else None)
                if color and model is not None and color in model.colors() and not model.fits(color, x0 + x, y0 + y, w, h, area, FRAME_W, FRAME_H, check_area=False):
                    color = None        # Registered: only near where the lamp was.
                if color and not painted(color, x, y, w, h):
                    add(color, x0 + x, y0 + y, w, h, area)

        # Keep every red-lamp candidate so wait_green can re-read the spots that were red.
        self.lamps["red_all"] = [(int(x), int(y), int(w), int(h)) for x, y, w, h, c in self.boxes if c == "red"]
        if biggest["red"] or biggest["green"]:
            self.state = "red" if biggest["red"] >= biggest["green"] else "green"
        else:
            self.state = None
        return self.state

    def color_at(self, frame, box):
        """Re-read only the box where red was, including the glow around it. Returns "red", "green", or None.

        The position is known, so the color test is wide: green includes cyan (LIGHT_G_HUE), and the saturation and value limits are lower.
        A center blown out to white still leaves color in the surrounding glow.
        """
        x, y, w, h = box
        pad = max(w, h)
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, FRAME_W), min(y + h + pad, FRAME_H)
        hue, sat, val = cv2.split(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV))
        colored = (sat >= LIGHT_LOCAL_S_MIN) & (val >= LIGHT_LOCAL_V_MIN)
        red = int((colored & self.is_red(hue)).sum())
        green = int((colored & (hue >= LIGHT_G_HUE[0]) & (hue <= LIGHT_G_HUE[1])).sum())
        need = max(LIGHT_AREA_MIN // 2, int(0.15 * w * h))
        color = "green" if green >= need and green > 2 * red else "red" if red >= need and red > 2 * green else None
        self.local.append((x0, y0, x1, y1, color))
        return color

    def draw(self, dbg, regions=True):
        """regions: also draw the lamp search areas (upper left and right). The drive view draws them only while waiting for a light."""
        dbg = as_draw(dbg)
        if regions:
            for x0, y0, x1, y1 in self.regions():
                dbg.rectangle((x0, y0), (x1 - 1, y1 - 1), C_DIM, 1)
        for x0, y0, x1, y1, color in self.local:    # Spot wait_green re-reads, where red was (yellow).
            dbg.rectangle((x0, y0), (x1 - 1, y1 - 1), C_AT, 1, cv2.LINE_AA)
            if color:
                chip(dbg, color.upper(), x0, y1 + 2, C_AT, 0.35)
        for x, y, w, h, color in self.boxes:
            bgr = (40, 40, 255) if color == "red" else (60, 220, 60)
            dbg.rectangle((x - 4, y - 4), (x + w + 4, y + h + 4), bgr, 2, cv2.LINE_AA)
            chip(dbg, color.upper(), x - 4, y - 22, bgr, 0.38)


class YawMeter:
    """How far the car has turned, from how far the room above the floor slides sideways between frames. A gyro substitute.

    Phase correlation of the top YAW_TOP of the frame at half size. Returns the total shift in px since reset (right turn is negative).
    A plain wall gives no match (low response), and such frames add nothing, so the turn then ends on the lane or max_sec as before.
    """

    def __init__(self):
        self.prev, self.win, self.total = None, None, 0.0

    def reset(self):
        self.prev, self.total = None, 0.0

    def __call__(self, frame):
        h = int(frame.shape[0] * YAW_TOP)
        g = cv2.resize(cv2.cvtColor(frame[:h], cv2.COLOR_BGR2GRAY), (frame.shape[1] // 2, h // 2),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.win is None or self.win.shape != g.shape:
            self.win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        if self.prev is not None:
            (dx, _), response = cv2.phaseCorrelate(self.prev, g, self.win)
            if response > 0.05:
                self.total += dx * 2
        self.prev = g
        return self.total


def stop_lead(hist, now, left, need):
    """go_to(): how far before the stop (cm) to count it as reached, so the car stops on it, not past it.

    hist: [(t, cm left)] of this stop point, newest last (this frame's added here). need: frames it must hold in a row.
    The speed comes from how the cm left fell over the last END_SPEED_SEC; with too few readings there is no lead.
    """
    hist.append((now, left))
    while len(hist) > 2 and now - hist[0][0] > END_SPEED_SEC:
        hist.pop(0)
    if not STOP_LEAD or len(hist) < 3 or now - hist[0][0] < 0.05:
        return 0.0
    speed = (hist[0][1] - left) / (now - hist[0][0])
    if not END_MIN_SPEED <= speed <= 80:
        return 0.0
    steps = [a[1] - b[1] for a, b in zip(hist, hist[1:])]
    if min(steps) < -1.0 or max(steps) > 2 * (sum(steps) / len(steps)) + 1.5:
        return 0.0      # Readings that jump about (a look-alike, a misread): no speed to trust.
    frame = (now - hist[0][0]) / (len(hist) - 1)
    return min(speed * (END_LEAD_SEC + (need - 1) * frame), STOP_LEAD_MAX_CM)


def floor_cm(y):
    """Floor distance ahead of the camera for a point at height y on screen: the lane is LANE_CM wide, and narrows to nothing at the horizon."""
    return CAMERA_F * LANE_CM / LANE_WIDTH_PX * (max(342 - HORIZON_Y, 1.0)) / max(y - HORIZON_Y, 1.0)


class EndRuler:
    """cm=: where the line this action ends on is, using the line itself as a ruler. It needs no floor texture and adds up no error.

    Before that line is reached, whatever is read may be some other line, or a misread. Readings that jump away are refused,
    but END_SWITCH_FRAMES of them in a row take over. Once it is reached (locked), nothing can replace it: the next line ahead is not it.
    When the line leaves the frame or stops being seen, its distance is carried on from the speed measured while it was visible.
    """

    def __init__(self):
        self.speed = None
        self.reset(None)

    def reset(self, kind):
        """A different line. The car's speed still holds."""
        self.kind, self.z, self.t, self.read_t, self.guess = kind, None, None, None, False
        self.raw, self.track, self.jump = [], [], []

    def feed(self, kind, y, now, locked=False, moving=True):
        if kind != self.kind:
            self.reset(kind)
        dt = 0.0 if self.t is None else min(now - self.t, 0.1)      # Capped: a stop for the LEGO figure is not driving.
        self.t = now
        z = None if y is None else floor_cm(y)
        if z is not None and self.z is not None and not -6 < z - self.z < 3:
            self.jump = (self.jump + [z])[-END_SWITCH_FRAMES:]
            if locked or len(self.jump) < END_SWITCH_FRAMES or max(self.jump) - min(self.jump) >= 3:
                z = None    # Jumped: another line, or a misread.
            else:
                self.reset(kind)
                self.t = now
        elif z is not None:
            self.jump = []
        if z is None:
            if self.z is not None and moving:
                speed = self.speed if self.speed is not None and self.speed >= END_MIN_SPEED else END_DEFAULT_SPEED
                self.z -= speed * dt
                self.guess = True
            return
        if self.guess:              # Back from carrying on: the old readings would pull the position backwards.
            self.raw, self.track = [], []
        self.raw = (self.raw + [z])[-3:]
        self.z = float(np.median(self.raw))     # A median of 3, so one misread frame cannot end the action.
        self.guess = False
        self.read_t = now
        self.track = [p for p in self.track if now - p[0] <= END_SPEED_SEC] + [(now, self.z)]
        if len(self.track) >= 5 and self.track[-1][0] - self.track[0][0] >= END_SPEED_SEC / 2:
            ts, zs = zip(*self.track)
            self.speed = max(0.0, -float(np.polyfit(np.array(ts) - ts[0], zs, 1)[0]))


class EndWatch:
    """Measurement only, changes no decision: where the line was when a to_junction / to_end action ended, and where it was once the car had stopped.

    The stop is seen only if a stop or wait action follows. Each result is printed and kept in rows, which a recording saves as ends.csv.
    """
    SETTLE_SEC = 0.4    # The line moving less than SETTLE_CM over this long means the car has stopped.
    SETTLE_CM = 0.3
    MAX_SEC = 3.0       # A stop not seen by then is reported as not seen.
    HEADER = "t,step,target_cm,how,end_z,end_y,speed,cmd,steer,offset,heading,stop_z,slid_cm,stop_sec,ref,ref_z"

    def __init__(self):
        self.rows, self.open, self.track = [], None, []

    def ended(self, idx, step, ruler, y, cmd, steer, det, now, stops):
        z = floor_cm(y) if y is not None else ruler.z       # Without a reading this frame, what the ruler carried on with.
        how = "line" if y is not None else ("guess" if z is not None else "none")
        row = {"t": now, "step": idx + 1, "target_cm": step.get("cm"), "how": how, "end_z": z, "end_y": y, "speed": ruler.speed, "cmd": cmd, "steer": steer,
               "offset": det[0] if det else None, "heading": det[1] if det else None, "stop_z": None, "slid_cm": None, "stop_sec": None,
               "ref": ruler.kind, "ref_z": None if ruler.z is None else ruler.z + (LANEEND_OFFSET_CM if ruler.kind == "laneend" else 0.0)}     # What it ended on, and where, as shown on screen for a lane end.
        self.rows.append(row)
        self.open = row if stops else None
        self.track = []
        fmt = lambda v, f=".1f": "-" if v is None else format(v, f)
        print(f"\n[measure {idx + 1}] end: line at {fmt(z)}cm ({how}, +{fmt(None if z is None else END_BASE_CM - z)}cm past the {END_BASE_CM:g}cm base, target {fmt(step.get('cm'), 'g')})"
              f" speed {fmt(ruler.speed)}cm/s command {cmd:.2f} steer {steer:+.1f} offset {fmt(row['offset'], '+.2f')} heading {fmt(row['heading'], '+.2f')}")

    def watch(self, y, now):
        row = self.open
        if row is None:
            return
        if y is not None:
            self.track = [p for p in self.track if now - p[0] <= self.SETTLE_SEC] + [(now, floor_cm(y))]
            zs = [p[1] for p in self.track]
            if now - self.track[0][0] >= self.SETTLE_SEC * 0.75 and max(zs) - min(zs) < self.SETTLE_CM:
                row["stop_z"] = float(np.median(zs))
                row["slid_cm"] = None if row["end_z"] is None else row["end_z"] - row["stop_z"]
                row["stop_sec"] = self.track[0][0] - row["t"]
                slid = "-" if row["slid_cm"] is None else f"{row['slid_cm']:.1f}"
                print(f"\n[measure {row['step']}] stopped: line at {row['stop_z']:.1f}cm, slid {slid}cm after the end ({row['stop_sec']:.2f}s)")
                self.open = None
                return
        if now - row["t"] >= self.MAX_SEC:
            print(f"\n[measure {row['step']}] stop position not read (the line left the view)")
            self.open = None

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.HEADER + "\n")
            for row in self.rows:
                f.write(",".join("" if row[k] is None else (f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k])) for k in self.HEADER.split(",")) + "\n")


class Steering:
    """Turn (position error, heading error) into a steering angle in degrees. Variable steering from P, filtered D, and a low-pass."""

    def __init__(self):
        self.prev_err = 0.0
        self.prev_t = None
        self.d = 0.0
        self.angle = 0.0

        self.out_t = None

    def lane(self, offset, heading, now, kp=1.0, kh=1.0):
        """Lane-follow steering target in degrees. Does not output it. kp and kh are the style= scales."""
        # As the car gets farther from the lane center, heading error counts less, so a car stuck on the lane edge returns first.
        err = kp * KP * offset + kh * KH * heading * max(0.0, 1 - abs(offset) / OFFSET_PRIORITY)
        if self.prev_t is not None and now > self.prev_t:
            # Frame-to-frame derivatives are noisy and cause zigzag if used raw, so filter them.
            self.d = 0.7 * self.d + 0.3 * (err - self.prev_err) / (now - self.prev_t)
        self.prev_err, self.prev_t = err, now
        return float(np.clip(err + KD * self.d, -1, 1)) * AUTO_STEER_MAX

    def output(self, target, now, smooth=None):
        """Turn a target angle into the actual steering angle, smoothed and rate-limited. Every course action goes through this."""
        if self.out_t is None:
            dt = 1 / 30
        elif now > self.out_t:
            dt = min(now - self.out_t, 0.1)    # A stall, or a second of holding the last angle while the lane is lost,
        else:                                  # must not be spent as one snap when the next command arrives.
            dt = 0.0
        self.out_t = now
        smooth = SMOOTH if smooth is None else smooth
        target = smooth * self.angle + (1 - smooth) * target
        step = STEER_RATE * dt      # Maximum turn in one frame.
        self.angle += float(np.clip(target - self.angle, -step, step))
        return self.angle

    def __call__(self, offset, heading, now):
        return self.output(self.lane(offset, heading, now), now)


class CourseRunner:
    """Run COURSE actions from top to bottom.

    Each action ends on sec (a fixed time) or until (junction, crosswalk, new lane, lane gap, or green light),
    and stays inside min_sec to max_sec. speed applies only during that action.
    """
    stop_whole = False      # Before the first action begins.

    EVENT_NAMES = {"junction": "junction", "crosswalk": "crosswalk", "lane": "new lane",
                   "gap": "lane gap", "end": "end (junction line or lane gap)", "crossed": "crosswalk passed", "green": "green light", "obstacle": "obstacle", "clear": "obstacle cleared",
                   "lost": "lane lost", "aligned": "aligned", "enter": "Enter",
                   "_crosswalk": "registered crosswalk", "_crossed": "registered crosswalk passed", "_stop": "stop point"}
    STATE_EVENTS = ("lane", "green", "clear", "aligned", "crossed", "_crossed")   # Judged as the current state: using one does not require it to count again. An obstacle counts again only after it goes away and comes back.
    AT_EVENTS = ("junction", "crosswalk", "gap", "end", "_crosswalk")    # Events that can use at= (how far down the frame the line must come).
    # end: a junction line or a lane gap, whichever comes first (to_end). Its y is the nearer of the two.

    def __init__(self, course, start=1):
        from course import _flat
        self.course = _flat(course)     # Expand even if groups (lists) are mixed in, so COURSE does not have to call _flat.
        # go() / to_junction() with neither cm= nor at=: the default lane-end distance, filled here (course.py cannot import this file).
        self.course = [dict(s, cm=LANEEND_BASE_CM) if s.get("cm_default") and s.get("cm") is None and JUNCTION_REF == "laneend" else s
                       for s in self.course]
        if isinstance(start, bool) or not isinstance(start, int) or start < 1 or start > max(len(self.course), 1):
            raise ValueError(f"the start number must be 1 to {max(len(self.course), 1)} ({start})")
        for n, step in enumerate(self.course, 1):      # A cm= the camera cannot read to is held at the nearest one it can.
            if self.uses_laneend(step) and not LANEEND_MIN_CM <= step["cm"] <= LANEEND_MAX_CM:
                print(f"[warning] step {n} to_junction(cm={step['cm']:g}): the lane end can be read only from {LANEEND_MIN_CM:g} to "
                      f"{LANEEND_MAX_CM:g} cm ahead. It ends at {self.laneend_target(step):g}cm")
        self.idx = start - 1        # Index of the action running now (from 0).
        self.step_start = None
        self.done = False
        self.hold = None            # straight(angle="hold"): steering angle at the moment it started.
        self.red_hits = 0           # wait_green: consecutive frames of red.
        self.red_seen = False       # wait_green: whether this action has definitely seen red. The car starts only on a change from red to green.
        self.red_box = None         # wait_green: last red lamp box. Green counts only near it.
        self.red_boxes = []         # wait_green: every red-lamp candidate seen while the light was red. Those spots are re-read.
        self.local_light = None     # wait_green: result of re-reading those spots. Driver fills this every frame.
        # Per event: consecutive detection frames, and whether it is armed. Something visible from the start (the start line) or just consumed must disappear once before it counts again.
        # Green, and obstacle present or clear, must be accepted even if that was already the state, so they stay armed.
        self.hits = {"junction": 0, "crosswalk": 0, "lane": 0, "gap": 0, "end": 0, "crossed": 0, "green": 0, "obstacle": 0, "clear": 0,
                     "lost": 0, "aligned": 0, "enter": 0, "_crosswalk": 0, "_crossed": 0, "_stop": 0}
        self.armed = {"junction": False, "crosswalk": False, "lane": True, "gap": False, "end": False, "crossed": True, "green": True,
                      "obstacle": True, "clear": True, "lost": True, "aligned": True, "enter": True, "_crosswalk": False, "_crossed": True, "_stop": False}
        self.since = {k: None for k in self.hits}   # Per event: time the current state started (used for hold_sec).
        self.miss = {k: 0 for k in self.hits}      # Per event: consecutive frames it was not seen. Used to arm during blind_sec.
        self.turn_clear = 0     # Consecutive frames during a turn with no horizontal line. Used for the exit decision.
        self.count = 0          # nth=: how many of this action's until events have passed so far.
        self.turned = None      # Turn with deg=: px the view has swung in the turn direction (YawMeter). Driver fills it every frame.
        self.paused = False     # Standing still for the LEGO figure. Driver fills it every frame.
        self.ruler = EndRuler()     # cm=: where the end line is.
        self.junc_used = False      # The last action ended on a junction line, so that line may still be in front of the car.
        self.gap_end = None         # gap=: after passing one, the next is not looked at until this time (set again by begin()).
        self.blind_top_y = None     # blind_sec: highest point (farthest) of the junction line seen during this action's blind period.
        self.end_hits = 0           # cm=: frames in a row the lane's line end has been found (LANEEND_FRAMES).
        self.line_gap = None        # cm=: the line across minus the lane end, when both were seen this action (LANEEND_LINE_GAP).
        self.run_out = False        # cm=: the end line was found and the car is covering the last distance.
        self.run_start = None       # cm=: when the end line was reached.
        self.moving = True          # cm=: whether the motor is driving the car forward. Driver fills it every frame.
        self.junc_latch = False     # until=junction: the cross line was seen low enough (JUNCTION_LATCH_FRAMES) to remember.
        self.junc_seen_y = None     # Last y of that line. It has to have reached JUNCTION_LOST_Y before a loss counts.
        self.junc_gone = 0          # Frames in a row it has been absent since, with the lane still visible.
        self.junc_hold = 0          # Frames in a row the low cross line has been seen. A gap starts this over.
        self.entered = False        # next(): Enter was pressed during this action.

    def waiting_enter(self):
        step = self.step()
        return step is not None and step["until"] == "enter"

    def press_enter(self):
        """Enter on the page. Counts only at a next(), so one pressed earlier cannot skip it."""
        self.entered = self.waiting_enter()

    def step(self):
        step = self.course[self.idx] if self.idx < len(self.course) else None
        if step is not None and step["angle"] is None:     # right()/left() default: the largest angle the car can steer.
            step = dict(step, angle=AUTO_STEER_MAX if step["do"] == "right" else -AUTO_STEER_MAX)
        return step

    @property
    def speed(self):
        step = self.step()
        return None if step is None else step["speed"]

    def begin(self, now):
        self.step_start = now
        self.clear_junction_latch()
        self.blind_top_y = None
        self.entered = False    # next(): Enter counts only once this action is running.
        self.count = 0          # nth=: how many of this action's until events have passed so far.
        self.gap_end = None     # gap=: after passing one, the next is not looked at until this time.
        self.stop_whole = False     # go_to(): the stop point found this frame is a road area (see STOP_WHOLE_FRAMES).
        self.hits["lane"] = 0
        self.turn_clear = 0
        self.hold = None
        self.red_hits, self.red_seen, self.red_box = 0, False, None
        self.red_boxes, self.local_light = [], None
        self.ruler, self.run_out, self.run_start = EndRuler(), False, None
        self.line_gap = None
        self.end_hits = 0
        step = self.step()
        if step is None:
            self.done = True
            print("\n[course end] stopped")
            return
        if step["do"] in ("straight", "left", "right") and step["until"] == "lane":
            # Straight through a gap, or a turn: the lane visible at the start (before the gap, before the turn) must disappear once and be found again, so the action does not end immediately.
            self.armed["lane"] = False
        if step.get("blind") and step["until"]:
            # Even after the blind period, do not end immediately on the line visible then. Count it again only after it has disappeared once.
            self.armed[step["until"]] = False
        if step["until"] is not None:
            # This action's end event starts counting fresh inside this action, so detections piled up during the previous action do not end it at once.
            self.hits[step["until"]] = 0
            self.since[step["until"]] = None
            self.miss[step["until"]] = 0
        if step["sec"] is not None:
            how = f"{step['sec']}s"
        else:
            how = f"until {self.EVENT_NAMES[step['until']]}" if step["until"] else ""
            if step.get("at") is not None:
                how += f" (at screen {step['at']})"
            if step.get("cm") is not None:
                how += f" +{step['cm']:g}cm"
            if step.get("hold"):
                how += f" held {step['hold']}s"
            if step["min"]:
                how += f" at least {step['min']}s"
            if step.get("blind"):
                how += f" ignore first {step['blind']:g}s"
            if step["max"] is not None:
                how += f" at most {step['max']}s"
        if step["do"] in ("left", "right", "curve"):
            angle = f" {step['angle']:+.0f} deg"
        elif step["do"] == "straight" and step["angle"] != 0:
            angle = " hold start angle" if step["angle"] == "hold" else f" {step['angle']:+.0f} deg"
        else:
            angle = ""
        speed = f" speed {step['speed']}" if step["speed"] is not None else ""
        print(f"\n[{self.idx + 1}/{len(self.course)}] {step['do']}{angle} {how}{speed}")

    def shift_clock(self, dt):
        """Standing for the LEGO figure does not use up this action's time, or the time a condition must hold."""
        if self.step_start is not None:
            self.step_start += dt
        self.since = {k: None if v is None else v + dt for k, v in self.since.items()}
        if self.run_start is not None:
            self.run_start += dt
        if self.gap_end is not None:    # gap=: the wait after a passed line is for driving on, not for standing.
            self.gap_end += dt

    def near_red(self, box):
        """Whether the green lamp is near where the red lamp was, on the same traffic light. Distance scales with lamp size."""
        if self.red_box is None or box is None:
            return False
        rx, ry, rw, rh = self.red_box
        gx, gy, gw, gh = box
        dist = ((rx + rw / 2 - gx - gw / 2) ** 2 + (ry + rh / 2 - gy - gh / 2) ** 2) ** 0.5
        return dist <= LIGHT_NEAR * max(rw, rh)

    def watch(self, junction, crosswalk, det, ys=None, light=None, lamps=None, obstacle=None, now=0.0):
        """Update the consecutive-detection count for each event.

        ys = {"junction": y, "crosswalk": y, "gap": y}: the lowest y where that line was seen.
        end is the line where the lane stops ahead, at the same place as gap. It is None when a junction horizontal line is present.
        If the current action has at=, that event counts only once the line has come down to the at= height.
        light is the traffic-light color ("red", "green", or None).
        Green counts only after red has been seen for JUNCTION_FRAMES in a row during this action. Red must change to green before the car starts.
        A green object that was visible from the start, or a green light that was already on before the car arrived, does not start the car.
        lamps is the per-color lamp boxes (TrafficLight.lamps): green must be near where the red lamp was (LIGHT_NEAR).
        If lamps is omitted, position is not compared.
        obstacle is whether the front infrared sensor sees an object. None if there is no sensor, which always means clear.
        """
        ys = dict(ys or {})
        dead_end = bool((self.step() or {}).get("dead_end"))     # go_end(): only a line across the lane ends it, never a gap.
        ends = [y for y in (ys.get("junction") if junction and det is not None else None, None if dead_end else ys.get("gap"))
                if y is not None]
        ys["end"] = max(ends) if ends else None     # end: the nearer of a junction line (seen with the lane) and a lane gap.
        self.red_hits = self.red_hits + 1 if light == "red" else 0
        if self.red_hits >= JUNCTION_FRAMES:
            self.red_seen = True
            if lamps and lamps.get("red"):
                self.red_box = lamps["red"]
            # Spots that were red are kept, not cleared: when the lamp turns green that red box disappears,
            # The spot must still be re-read (the frame can stay 'red' because of some other red object).
            for box in (lamps or {}).get("red_all", []):
                if len(self.red_boxes) < 6 and not any(
                        abs(box[0] - b[0]) < max(b[2], 4) and abs(box[1] - b[1]) < max(b[3], 4) for b in self.red_boxes):
                    self.red_boxes.append(box)
        green_here = lamps is None or self.near_red(lamps.get("green"))
        step = self.step()
        # During a turn, a new lane needs a line just ahead plus position and heading (TURN_END_*). Corner tape can match position alone.
        # Its heading is near ±1, so it is rejected here, while the exit straight has heading near 0 and is accepted.
        turning = step is not None and step["do"] in ("left", "right")
        self.turn_clear = self.turn_clear + 1 if turning and not junction else 0
        # While a corner horizontal line is still present, heading is judged strictly. Once it is gone, a slanted exit line is accepted.
        heading_limit = TURN_EXIT_HEADING if self.turn_clear >= TURN_EXIT_CLEAR else TURN_END_HEADING
        # A confirmed lane end is the junction, wherever the line across is: a + has none, and in an L it may lie 20 cm beyond.
        early = step is not None and self.uses_laneend(step) and self.ruler.kind == "laneend"
        # A cross line counts with the lane. Once this action has already seen it, a frame where the lane center
        # drops (the right-line point gone, or only a far one) still counts. A line that never had a lane is not one.
        junction_seen = ((junction or early) and det is not None) or (junction and self.hits["junction"] > 0)
        seen = {"junction": junction_seen, "crosswalk": crosswalk,
                # New lane: a line is visible just in front of the car and the lane roughly matches. No line just ahead, as when sitting over a gap, counts as gone.
                # The lane beyond a gap may stay visible higher up, but straight(until="lane") ends only after the line just ahead disappears and comes back.
                # A cross line already in the stop zone is the junction even if the right-line point (the red dot) is missing or only far, so the lane center drops.
                # A + junction has no cross line: that one still needs the lane. A line with no lane at all is handled below.
                # Or the car is already pointed along a lane but sits off to one side (TURN_END_*_WIDE): alignment fixes the offset,
                # while turning on would swing past the exit (run 20261103-200649: heading 0.1, offset 0.5-0.6 for 0.4 s, turn ran to max).
                "lane": (det is not None and ys.get("near", True)
                         and ((abs(det[0]) < TURN_END_OFFSET and abs(det[1]) < heading_limit)
                              or (abs(det[0]) < TURN_END_OFFSET_WIDE and abs(det[1]) < TURN_END_HEADING_WIDE))) if turning else
                        (not junction and det is not None and abs(det[0]) < 0.6 and abs(det[1]) < 0.5
                         and ys.get("near", True)),
                "gap": ys.get("gap") is not None,
                "end": ys["end"] is not None,
                # crossed: no crosswalk anywhere in view, with the lane found. The crosswalk has gone under the car.
                "crossed": ys.get("crosswalk") is None and det is not None,
                # The registered crosswalk (aicar crosswalk): in the stop zone, and gone from view with the lane found.
                "_crosswalk": ys.get("_crosswalk") is not None and ys["_crosswalk"] >= JUNCTION_Y * FRAME_H,
                "_crossed": ys.get("_crosswalk") is None and det is not None,
                # go_to("A"): the registered shape has come down to where it was when the car stood at the stop.
                "_stop": ys.get("_stop") is not None and ys.get("_stop_target") is not None and ys["_stop"] >= ys["_stop_target"],
                # Green: a green light near the red light anywhere in the frame, or the spot that was red has changed to green (cyan included).
                "green": self.red_seen and ((light == "green" and green_here) or self.local_light == "green"),
                "obstacle": bool(obstacle), "clear": not obstacle,
                "lost": det is None,
                "aligned": det is not None and abs(det[0]) < ALIGN_OFFSET and abs(det[1]) < ALIGN_HEADING}
        if step is not None and step["until"] in self.AT_EVENTS and (step.get("at") is not None or step["until"] in ("gap", "end")):
            # gap / end with no at= end at JUNCTION_Y like junction and crosswalk, not on a gap seen anywhere in the frame.
            y = ys.get(step["until"])
            near_enough = y is not None and y >= (step.get("at") or JUNCTION_Y) * FRAME_H - 1     # -1 absorbs float error such as 0.55*360 = 198.00000000000003.
            if step["until"] == "junction":
                # at= must keep a + junction, which has no line across the lane. early is that lane end.
                # A cross line that has reached at= counts even when the lane center drops after it was already seen.
                seen["junction"] = (near_enough and det is not None) or (early and det is not None) or (
                    near_enough and self.hits["junction"] > 0)
            else:
                seen[step["until"]] = near_enough
        until = step["until"] if step is not None else None
        # blind_sec: for this long after the action starts, the end condition is not counted and the event is not armed.
        # A line that flickers during that time does not end the action the moment the blind period finishes.
        blind = (until is not None and self.step_start is not None
                 and (now - self.step_start < (step.get("blind") or 0)
                      or (self.gap_end is not None and now < self.gap_end)))
        for k, v in seen.items():
            self.miss[k] = 0 if v else self.miss[k] + 1
            if blind and k == until:
                self.hits[k] = 0
                self.since[k] = None
                # Gone for BLIND_REARM_FRAMES in a row during blind still counts as "disappeared once". A one- or two-frame flicker does not.
                # Otherwise a line that left and came back inside the blind period (the old lane during a quick turn) never arms.
                if self.miss[k] >= BLIND_REARM_FRAMES:
                    self.armed[k] = True
                continue
            self.hits[k] = self.hits[k] + 1 if v else 0
            self.since[k] = (self.since[k] if self.since[k] is not None else now) if v else None
            if not v:
                self.armed[k] = True
            elif k == "junction" and until == "junction" and self.count == 0 and not self.junc_used:
                # The line in front is the one this action is driving to. It often stays in view the whole way,
                # so waiting for it to vanish once meant it never counted and the car drove through it.
                # nth= has already passed a line when count > 0: that line must still disappear before the next one counts.
                # So must the line the last action ended on (go(); go() with no blind_sec ended the second one on it at once).
                self.armed[k] = True
        self.note_junction(ys.get("junction"), det, blind)

    def blind_watch(self, line_y, det):
        """blind_sec: whether the junction line is followed during blind, so that a near junction that goes under the car
        before blind ends still counts. Only a line first seen well ahead of the car (BLIND_NEW_LINE_Y), with the lane in view,
        when no earlier action ended on a junction still in front (go(); go() passes it), and before any nth= line was passed.
        """
        if line_y is not None and det is not None:
            self.blind_top_y = line_y if self.blind_top_y is None else min(self.blind_top_y, line_y)
        return (not self.junc_used and self.count == 0 and self.gap_end is None
                and self.blind_top_y is not None and self.blind_top_y < BLIND_NEW_LINE_Y * FRAME_H - 1)

    def clear_junction_latch(self):
        """Forget a junction line remembered because it reached the bottom of the frame."""
        self.junc_latch, self.junc_seen_y, self.junc_gone, self.junc_hold = False, None, 0, 0

    def note_junction(self, line_y, det, blind):
        """Remember a cross line that has come down the frame, and count the frames it has since been gone.

        Holding the line for JUNCTION_FRAMES still ends the action on its own. A line that then slips under the car
        often misses that count, because one gap resets it. Seen low for JUNCTION_LATCH_FRAMES, then gone for
        JUNCTION_GONE_FRAMES, is the same junction. The lane has to be visible: losing the lane is not the line leaving.
        A line that disappears before JUNCTION_LOST_Y was a false line, so the memory is dropped. blind_sec remembers nothing.
        A + junction has no cross line; its lane end does not come through here.
        """
        step = self.step()
        if step is None or step["until"] != "junction":
            return
        if blind and not self.blind_watch(line_y, det):
            self.junc_hold = 0
            self.junc_gone = 0
            return
        if det is None:
            return
        at = step.get("at")
        mark = (JUNCTION_Y if at is None else at) * FRAME_H - 1
        low = line_y is not None and line_y >= mark
        if line_y is not None and self.junc_latch:
            self.junc_seen_y = line_y       # Where it was last, including after it has climbed back up the frame.
        if low and (self.armed["junction"] or blind):
            self.junc_hold += 1
            self.junc_gone = 0
            if self.junc_hold >= JUNCTION_LATCH_FRAMES:
                self.junc_latch = True
                self.junc_seen_y = line_y
                self.armed["junction"] = True   # Only reached during blind for a line that came down the frame (blind_watch).
            return
        self.junc_hold = 0
        if not self.junc_latch or line_y is not None:
            self.junc_gone = 0
            return
        self.junc_gone += 1
        if self.junc_gone >= JUNCTION_GONE_FRAMES and (self.junc_seen_y or 0) < JUNCTION_LOST_Y * FRAME_H - 1:
            self.clear_junction_latch()

    def track_end(self, ys, now):
        """cm=: measure the end line every frame, before it is reached too, so the speed is known by the time it leaves the frame."""
        step = self.step()
        if step is None or step.get("cm") is None:
            return
        hline, tip, lane_end = ys.get("hline_y"), ys.get("tip_y"), ys.get("end_y")
        if self.uses_laneend(step):
            if lane_end is not None and hline is not None:      # Both in view: how far beyond the lane end the line across lies here.
                gap = floor_cm(hline) - floor_cm(lane_end)
                if 5.0 < gap < 40.0:
                    self.line_gap = gap if self.line_gap is None else 0.8 * self.line_gap + 0.2 * gap
            self.end_hits = self.end_hits + 1 if lane_end is not None else 0
            if self.ruler.kind == "laneend" or self.end_hits >= LANEEND_FRAMES:
                self.ruler.feed("laneend", lane_end, now, locked=self.run_out, moving=self.moving)
                return
            # No line end yet: the line across the lane is measured as before, and stands in if no end is ever found.
        if self.run_out:
            self.ruler.feed(self.ruler.kind, {"line": hline, "tip": tip}.get(self.ruler.kind), now, locked=True, moving=self.moving)
        elif hline is not None:
            self.ruler.feed("line", hline, now)
        elif step["until"] == "end" and not step.get("dead_end") and ys.get("gap") is not None and tip is not None:
            self.ruler.feed("tip", tip, now)

    @staticmethod
    def uses_laneend(step):
        """to_junction(cm=) measured to where the lane's line ends (JUNCTION_REF)."""
        return JUNCTION_REF == "laneend" and step["until"] == "junction" and step.get("cm") is not None

    @staticmethod
    def laneend_target(step):
        """Distance of the line end where the action ends: in view, so it is read, not guessed."""
        return float(np.clip(step["cm"], LANEEND_MIN_CM, LANEEND_MAX_CM))

    def laneend_dist(self, step):
        """Lane-end distance from the car front in cm (as shown on screen), or None when unknown."""
        if not self.uses_laneend(step) or self.ruler.z is None or self.ruler.guess:
            return None
        if self.ruler.kind == "laneend":
            return self.ruler.z + LANEEND_OFFSET_CM
        if self.ruler.kind == "line":
            return self.ruler.z + LANEEND_OFFSET_CM - self.lane_gap()
        return None

    def start_run_out(self, step, ys, now):
        """cm=: the end line was reached. Make sure the ruler is on that line, not on what it was reading before."""
        self.run_out, self.run_start = True, now
        ruler = self.ruler
        if self.uses_laneend(step) and ruler.kind == "laneend":
            return      # Already on the line end: track_end fed it this frame.
        kind = "line" if step["until"] == "junction" or step.get("dead_end") or ys.get("junction") is not None else "tip"
        y = ys.get("hline_y") if kind == "line" else ys.get("tip_y")
        if kind != ruler.kind or ruler.jump or ruler.read_t is None or now - ruler.read_t > END_FRESH_SEC:
            ruler.reset(kind)
            if y is not None:
                ruler.feed(kind, y, now, locked=True)
            else:       # Only the band test sees it. Its band center is coarse, but much better than no distance at all.
                band = ys.get("junction") if kind == "line" else ys.get("gap")
                if band is not None:
                    ruler.z, ruler.t, ruler.guess = floor_cm(band), now, True

    def end_reached(self, step):
        """cm=: the end line has come to its place. A line never measured ends at once, since nothing can be said about it."""
        if self.ruler.z is None:
            return True
        speed = self.ruler.speed if self.moving and self.ruler.speed else 0.0
        if self.ruler.kind == "laneend":
            return self.ruler.z + LANEEND_OFFSET_CM - speed * END_LEAD_SEC <= self.laneend_target(step)
        if step.get("dead_end"):        # go_end(): the line closing the lane, cm ahead of the car front.
            return self.ruler.z + LANEEND_OFFSET_CM - speed * END_LEAD_SEC <= step["cm"]
        if self.uses_laneend(step):     # No line end was found: where it would be, before the line across by the gap.
            return self.ruler.z + LANEEND_OFFSET_CM - self.lane_gap() - speed * END_LEAD_SEC <= self.laneend_target(step)
        return self.ruler.z - speed * END_LEAD_SEC <= END_BASE_CM - step["cm"]

    def lane_gap(self):
        """How far beyond the lane end the line across lies: as measured this action, or LANEEND_LINE_GAP."""
        return LANEEND_LINE_GAP if self.line_gap is None else self.line_gap

    def happened(self, event, now=0.0, hold=0.0):
        step = self.step()
        if event == "lane" and step is not None and step["do"] in ("left", "right"):
            need = TURN_END_FRAMES
        else:
            need = {"lane": TURN_CLEAR_FRAMES, "aligned": ALIGN_FRAMES}.get(event, JUNCTION_FRAMES)
            if event == "_stop" and self.stop_whole:
                need = STOP_WHOLE_FRAMES        # A road area matches only close to its place: fewer frames, or a quick car would pass it.
        held = hold <= 0 or (self.since[event] is not None and now - self.since[event] >= hold)
        return self.armed[event] and self.hits[event] >= need and held

    def junction_lost(self):
        """The cross line was seen at the bottom of the frame, then left it. Same end as holding it for JUNCTION_FRAMES."""
        step = self.step()
        if (step is None or step["until"] != "junction" or not self.armed["junction"]
                or not self.junc_latch or self.junc_gone < JUNCTION_GONE_FRAMES):
            return False
        return (self.junc_seen_y or 0) >= JUNCTION_LOST_Y * FRAME_H - 1

    def update(self, junction, now, det=None, crosswalk=False, ys=None, light=None, lamps=None, obstacle=None):
        if self.step_start is None:
            self.begin(now)
        self.watch(junction, crosswalk, det, ys, light, lamps, obstacle, now)
        self.track_end(ys or {}, now)
        while (step := self.step()) is not None:
            if step["do"] == "finish":
                self.done = True
                return
            elapsed = now - self.step_start
            if step["until"] == "enter":
                reason = "Enter" if self.entered else None
            elif step["sec"] is not None and step["until"] is None:
                reason = "time" if elapsed >= step["sec"] else None
            elif step["sec"] is not None and elapsed >= step["sec"]:
                reason = "time"     # wait() keeps both, and ends on whichever comes first.
            elif step["max"] is not None and elapsed >= step["max"] and not (self.needs_light() and self.red_seen):
                # wait_green: max_sec is for a red light that was never found. Once red is seen, the car waits for green, however long.
                reason = "max time"
            elif (step.get("deg") and self.turned is not None and elapsed >= step["min"]
                  and self.turned >= step["deg"] / 90 * TURN_YAW_PX):
                reason = f"turned {step['deg']} deg"
            elif self.run_out:
                # Checked before the safety cap. The last cm= has its own limit, and the safety cap was already
                # spent getting to the line. Cutting it here stops short of the measured spot.
                if self.end_reached(step):
                    reason = self.end_reason(step)
                else:
                    limit = LANEEND_MAX_SEC if self.uses_laneend(step) else END_MAX_SEC
                    reason = (f"{self.EVENT_NAMES[step['until']]} +{step['cm']:g}cm not measured, {limit:g}s passed"
                              if now - self.run_start >= limit else None)
            elif step["max"] is None and step["until"] is not None and step["do"] != "wait" and elapsed >= STEP_MAX_SEC:
                reason = f"safety time {STEP_MAX_SEC}s (the condition never came, set max_sec to change)"
            elif (step["until"] is not None and elapsed >= step["min"]
                  and (self.gap_end is None or now >= self.gap_end)     # gap=: still in the wait after a passed one.
                  and (self.happened(step["until"], now, step.get("hold", 0.0)) or self.junction_lost())
                  and not (self.uses_laneend(step) and self.ruler.kind != "laneend" and 0 < self.end_hits < LANEEND_FRAMES)):    # A lane end being confirmed: wait the few frames for it.
                lost = self.junction_lost()     # Cleared below, so read it first. A held line and a lost line never both match one frame.
                reason = self.EVENT_NAMES[step["until"]]
                self.clear_junction_latch()     # This line is used up. The next one has to be seen on its own.
                if step["until"] not in self.STATE_EVENTS:
                    self.armed[step["until"]] = False   # Do not count the same line again in the next action.
                self.count += 1
                if self.count < (step.get("nth") or 1):
                    # nth=: pass this one (a bend that looks like a junction) and keep going. It must vanish before the next counts.
                    print(f"\n[{self.idx + 1}] {reason} #{self.count} passed (ends at #{step['nth']})")
                    self.ruler.reset(None)      # The next line is a new one.
                    self.end_hits, self.line_gap = 0, None
                    if step.get("gap"):
                        wait = step["gap"][min(self.count, len(step["gap"])) - 1]
                        self.gap_end = now + wait
                        print(f"[{self.idx + 1}] next one not looked at for {wait:g}s")
                    reason = None
                elif step.get("cm") is not None:
                    self.start_run_out(step, ys or {}, now)
                    reason = self.end_reason(step) if self.end_reached(step) else None
                elif lost:
                    reason = f"{reason} (lost)"
            else:
                reason = None
            if reason is None:
                return
            print(f"\n[{self.idx + 1} done] {reason} ({elapsed:.1f}s)")
            if step["until"] == "junction" and (self.run_out or self.count >= (step.get("nth") or 1)):
                self.junc_used = True       # Ended on the line: it is still in front of (or under) the car.
            elif step["do"] in ("follow", "straight", "left", "right"):
                self.junc_used = False      # Drove on without ending on a junction: any such line is behind.
            self.idx += 1
            self.begin(now)

    def end_reason(self, step):
        z = self.ruler.z
        if z is None:
            return f"{self.EVENT_NAMES[step['until']]} (line not measured, ended without +{step['cm']:g}cm)"
        how = "guessed from speed" if self.ruler.guess else "measured on the line"
        if step.get("dead_end"):
            return f"dead end: end line {z + LANEEND_OFFSET_CM:.1f}cm ahead of the car front ({how}, target {step['cm']:g}cm)"
        if self.ruler.kind == "laneend":
            return f"{self.EVENT_NAMES[step['until']]} lane end at {z + LANEEND_OFFSET_CM:.1f}cm ({how}, target {self.laneend_target(step):g}cm)"
        if self.uses_laneend(step):
            gap = self.lane_gap()
            return (f"{self.EVENT_NAMES[step['until']]} lane end at about {z + LANEEND_OFFSET_CM - gap:.1f}cm ({how}, no lane end seen: "
                    f"the line across minus {gap:.0f}cm, target {self.laneend_target(step):g}cm)")
        ref = ""
        return f"{self.EVENT_NAMES[step['until']]} +{END_BASE_CM - z:.1f}cm ({how}{ref}, target {step['cm']:g}cm)"

    def lane_side(self, now):
        step = self.step()
        return (step or {}).get("side") or "center"

    def final_approach(self):
        """Heading for the line this action ends on: not still before (or passing) an nth= line it skips."""
        step = self.step()
        if step is None or step["until"] is None:
            return False
        if self.run_out:
            return True
        nth = step.get("nth") or 1
        return self.count >= nth - 1 and (self.count == 0 or self.armed[step["until"]])

    def opt(self, key, default=None):
        """Option on the current action (shift, steer, style, ...). default if the course has ended or the option was not set."""
        step = self.step()
        value = (step or {}).get(key)
        return default if value is None else value

    def turning(self, now):
        """True during a turn action, before control has handed over to lane following."""
        step = self.step()
        if step is None or step["do"] not in ("left", "right"):
            return False
        if self.step_start is None:     # First frame, before update() has started the action (driving with --start at a turn).
            return True
        return now - self.step_start < step["min"] or self.hits["lane"] < TURN_CLEAR_FRAMES

    def following(self):
        step = self.step()
        return step is None or step["do"] == "follow"

    def target(self, lane, now, current=0.0):
        """Steering target in degrees for the current action. lane is the lane-follow steering, or None if the lane is not visible.

        current is the actual steering angle now. straight(angle="hold") uses it to remember the angle at the start.
        There is no fixed angle: every action mixes in lane-follow steering to the extent the lane is visible.
        """
        step = self.step()
        if step is None or step["do"] in ("follow", "stop", "wait", "finish"):
            return lane
        angle = step["angle"]
        if step["do"] == "back":
            return angle        # Backing up: the wheels as set, no lane steering (it is for driving forward).
        if step["do"] == "straight":
            if angle == "hold":
                if self.hold is None:
                    self.hold = current
                angle = self.hold
            if lane is None:
                return angle
            return angle + float(np.clip(lane - angle, -STRAIGHT_CORR, STRAIGHT_CORR))
        if lane is None:
            return angle
        if step["do"] == "curve":
            return angle + CURVE_ASSIST * (lane - angle)
        # Turn: hold the set angle for the minimum time, then hand over to lane-follow steering as the new lane stays found.
        if now - self.step_start < step["min"]:
            return angle
        w = max(0.0, 1 - self.hits["lane"] / TURN_CLEAR_FRAMES)
        return w * angle + (1 - w) * lane

    def stopped(self, now):
        step = self.step()
        return step is not None and step["do"] in ("stop", "wait")

    def needs_light(self):
        """Whether this action needs traffic-light detection (wait_green). Otherwise the light computation is skipped."""
        step = self.step()
        return step is not None and step["do"] == "wait" and step["until"] == "green"

    def label(self):
        return f"{min(self.idx + 1, len(self.course))}/{len(self.course)}"

    def status(self, now):
        step = self.step()
        if self.paused:
            return "PERSON STOP"
        if step is None:
            return "END"
        if step["do"] in ("left", "right", "curve"):
            return f"{step['do'].upper()} {step['angle']:+.0f}"
        if step["do"] == "wait":
            if step["until"] == "green":
                return "WAIT GREEN" if self.red_seen else "WAIT RED"
            return {"obstacle": "WAIT OBSTACLE", "clear": "WAIT CLEAR", "enter": "WAIT ENTER"}.get(step["until"], "WAIT")
        if step["do"] == "straight" and step["angle"] == "hold":
            return "STRAIGHT HOLD"
        return step["do"].upper()


def speed_for(angle, base):
    """Slow down as steering grows."""
    if base <= MIN_SPEED:   # An action that set a very slow speed keeps that speed.
        return base
    ratio = min(abs(angle) / STEER_LIMIT, 1.0)
    return base - (base - MIN_SPEED) * ratio


WEB_KEY_FILE =VISION_FILE.with_name("web_key.json")


def web_key():
    """Secret for sending keys from the web page, made once and kept on the Pi. aicar puts it in the address it opens."""
    try:
        return json.loads(WEB_KEY_FILE.read_text())
    except (OSError, ValueError):
        key = secrets.token_urlsafe(16)
        write_json(WEB_KEY_FILE, key, backup=False, mode=0o600)
        return key


class KeyInbox:
    """Keys typed on the web page, waiting for the running program. Keys only: the page can run nothing else on the car."""

    def __init__(self):
        self.pending = collections.deque(maxlen=64)     # A program that reads none keeps only the newest.

    def add(self, text):
        self.pending.extend(c for c in text if " " <= c <= "~" or c in "\n\r\x03")

    def take(self):
        """Keys typed since the last call. Ctrl+C acts as it does in a terminal, so each program's existing clean exit runs."""
        keys = ""
        while self.pending:
            keys += self.pending.popleft()
        if "\x03" in keys:
            raise KeyboardInterrupt
        return keys


class Stream:
    """Send the debug view to a browser, and take keys from it for the running program (self.keys).

    The page fetches the latest view one at a time (/frame.bin): the camera picture as JPEG plus the drawing commands
    (overlay.Canvas), which the browser paints itself. The Pi draws nothing and encodes only the plain picture.
    If Wi-Fi drops briefly or the program restarts, the page reconnects on its own, and a slow client always gets the latest view with none queued up.
    Encoding runs on the browser request thread, not the main loop, and only while someone is watching (not while the page is paused or hidden).
    """

    PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
:root{--bg:#000;--panel:#161616;--line:#2a2a2a;--g:#50e650;--txt:#f0f0f0;--dim:#969696;--off:#373737;--red:#ff4040;--amb:#ffc83c;
 --ok:#50e650;--bad:#ff4040;--warn:#ffb020}
*{box-sizing:border-box} html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--txt);font:14px -apple-system,"Apple SD Gothic Neo",sans-serif;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:10px;padding:8px 14px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.logo{color:#fff;font-weight:700;font-size:15px}
#st{display:flex;align-items:center;gap:6px;color:#aaa;font-size:13px;margin-left:auto}
#dot{width:10px;height:10px;border-radius:50%;background:#666}
.tools{display:flex;gap:6px}
button{background:var(--panel);color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:5px 10px;font:inherit;font-size:13px;cursor:pointer}
button:hover{border-color:#555} button.on{border-color:var(--warn);color:var(--warn)}
kbd{font:11px ui-monospace,monospace;color:var(--dim);margin-left:4px}
main{flex:1;min-height:0;display:flex}
.wrap{position:relative;flex:1;min-width:0;min-height:0;display:flex;align-items:center;justify-content:center;background:#000}
#v{display:block;width:100%;height:100%;object-fit:contain}
#msg{position:absolute;inset:0;display:none;align-items:center;justify-content:center;z-index:3;
 background:rgba(0,0,0,.55);color:#fff;font-size:18px;font-weight:600;text-align:center;padding:16px}
#paused{position:absolute;top:10px;left:10px;display:none;background:rgba(0,0,0,.7);color:var(--warn);z-index:2;
 border:1px solid var(--warn);border-radius:6px;padding:4px 10px;font-weight:600}
#alert{position:absolute;top:10px;left:50%;transform:translateX(-50%);display:none;z-index:2;max-width:92%;color:#fff;
 padding:7px 16px;border-radius:4px;font-weight:700;font-size:17px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:.92}
#drag{position:absolute;display:none;border:2px solid var(--ok);background:rgba(60,220,60,.15);pointer-events:none}
.wrap.drag{cursor:crosshair;touch-action:none} .wrap.drag #v{user-select:none;-webkit-user-select:none}
#toast{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.75);color:#fff;z-index:3;
 border-radius:6px;padding:6px 12px;opacity:0;transition:opacity .3s;pointer-events:none}
aside{display:none;width:290px;flex:0 0 290px;flex-direction:column;background:var(--panel);overflow:auto}
body.dash aside{display:flex}
.sec{padding:10px 14px;border-bottom:1px solid var(--line)}
.cap{display:flex;justify-content:space-between;color:var(--dim);font-size:12px;margin-bottom:4px}
#title{color:#fff;font-size:18px;font-weight:700;margin:4px 0 2px}
#until{color:var(--amb);font-size:12px}
.two{display:flex;gap:14px} .two>div{flex:1;min-width:0}
.big{font-size:18px;font-weight:600}
.bar{position:relative;height:8px;background:#464646;margin-top:5px}
.bar i{position:absolute;top:0;bottom:0;left:0;background:var(--g)}
.bar.thin{height:3px;margin-top:8px} .bar.thin i{background:var(--dim)}
.lanebar{position:relative;height:14px;margin:4px 0 2px;border-left:3px solid #288cff;border-right:3px solid #ff3c3c;background:#1f1f1f}
.lanebar::before{content:"";position:absolute;left:50%;top:0;bottom:0;width:2px;margin-left:-1px;background:var(--g)}
.lanebar b{position:absolute;top:2px;width:10px;height:10px;margin-left:-5px;border-radius:50%;background:#fff;transition:left .1s linear}
.bar.mid i{background:#50c8ff} .bar.mid::before{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:#fff;z-index:1}
#flags{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.flag{background:var(--off);color:var(--dim);font-size:11px;padding:3px 6px;border-radius:2px;display:flex;justify-content:space-between;white-space:nowrap;overflow:hidden}
.flag.on{color:#fff} .flag.dark{color:#141414}
.mini canvas{display:block;width:100%;aspect-ratio:4/3;background:#000;border:1px solid var(--line)}
.mini .tag{display:block;margin-top:5px;font-weight:700;font-size:13px}
.scr{position:relative;max-width:220px} .scr.none::after{content:"none";position:absolute;inset:0;display:flex;align-items:center;
 justify-content:center;color:#555;font-size:12px}
#keys{display:flex;align-items:center;gap:8px;padding:8px 14px;border-top:1px solid var(--line);flex-wrap:wrap}
#keys label{color:#fff;font-weight:600;font-size:13px} #kstate{color:var(--ok);font-size:12px} #keys small{color:var(--dim);flex:1 1 200px}
@media (max-width:900px){main{flex-direction:column;overflow:auto} .wrap{flex:none;aspect-ratio:16/9} aside{width:100%;flex:none;overflow:visible}}
@media (max-width:600px){kbd{display:none} .tools{width:100%} button{flex:1} #keys small{flex-basis:100%}}
</style></head><body>
<header><span class="logo">__TITLE__</span>
<span id="st"><i id="dot"></i><span id="stt">connecting…</span></span>
<span class="tools">
 <button id="bPause" title="pause / resume the view">Pause<kbd>Space</kbd></button>
 <button id="bSave" title="save the view as a photo">Save<kbd>S</kbd></button>
 <button id="bFull" title="full screen (double click too)">Full<kbd>F</kbd></button>
</span></header>
<main>
<div class="wrap" id="wrap"><canvas id="v" width="0" height="0"></canvas><div id="alert"></div><div id="drag"></div><div id="paused">Paused</div>
<div id="msg">Disconnected · reconnecting…</div><div id="toast"></div></div>
<aside id="side">
 <div class="sec"><div class="cap"><span id="stepn">-</span><span id="el"></span></div><div id="title">-</div><div id="until"></div><div class="bar thin" id="limb"><i id="lim"></i></div></div>
 <div class="sec two"><div><div class="cap">SPEED</div><div class="big" id="spd">0.00</div><div class="bar"><i id="spdb"></i></div></div>
  <div><div class="cap">STEER</div><div class="big" id="str">0°</div><div class="bar mid"><i id="strb"></i></div></div></div>
 <div class="sec"><div class="cap"><span>POSITION</span><span id="posv">-</span></div><div class="lanebar"><b id="pos"></b></div></div>
 <div class="sec"><div id="flags"></div></div>
 <div class="sec two">
  <div class="mini"><div class="cap">LIGHT</div><div class="scr"><canvas id="mLight" width="160" height="120"></canvas></div><span class="tag" id="tLight">-</span></div>
  <div class="mini" id="cPerson"><div class="cap">PERSON</div><div class="scr"><canvas id="mPerson" width="160" height="120"></canvas></div><span class="tag" id="tPerson">-</span></div>
 </div>
</aside>
</main>
<div id="keys"><label id="klabel">Terminal keys</label>
<button id="bEnter">Enter</button><button id="bStop">Ctrl+C</button><span id="kstate"></span><small>__KEYS_HELP__</small></div>
<script>
// Fetch the latest view one at a time: the camera picture (JPEG) and the drawing commands, painted here, not on the Pi.
// On failure wait a moment and keep retrying (resumes by itself when the program restarts).
const $ = id => document.getElementById(id);
const cv = $("v"), ctx = cv.getContext("2d"), msg = $("msg"), dot = $("dot"), stt = $("stt"), wrap = $("wrap");
let seq = -1, fails = 0, shown = [], paused = false, pausedAt = 0, drawn = false, wake = null;
function status(ok, text) { dot.style.background = ok ? "var(--ok)" : "var(--bad)"; stt.textContent = text; }
function toast(t) { const e = $("toast"); e.textContent = t; e.style.opacity = 1; clearTimeout(e.t); e.t = setTimeout(() => e.style.opacity = 0, 1500); }
function timeout(ms) { const c = new AbortController(); setTimeout(() => c.abort(), ms); return c.signal; }   // AbortSignal.timeout is missing on older Safari.
function setPaused(p) {
  paused = p; pausedAt = Date.now(); $("paused").style.display = p ? "block" : "none";
  $("bPause").classList.toggle("on", p); $("bPause").firstChild.textContent = p ? "Resume" : "Pause";
  if (!p && wake) wake();
}
function save() {
  if (!drawn) return toast("No frame received yet");
  const d = new Date(), z = n => String(n).padStart(2, "0");
  const name = `aicar-${d.getFullYear()}${z(d.getMonth() + 1)}${z(d.getDate())}-${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}.png`;
  cv.toBlob(b => {
    if (!b) return toast("Could not save the photo");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b); a.download = name;
    a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); toast("Photo saved: " + name);
  }, "image/png");
}
// --- Painting the commands (overlay.py Canvas.payload). Coordinates are camera pixels, as the Pi drew them. ---
const tint = document.createElement("canvas"), tctx = tint.getContext("2d");
let tintData = null;
const FONT = "Arial, Helvetica, sans-serif";
function rgb(c) { return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)]; }
function seg(x0, y0, x1, y1, c, t) {
  const o = t % 2 ? 0.5 : 0;
  ctx.strokeStyle = c; ctx.lineWidth = t; ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(x0 + o, y0 + o); ctx.lineTo(x1 + o, y1 + o); ctx.stroke();
}
function paint(ops, blobs) {
  for (const o of ops) {
    switch (o[0]) {
      case "l": seg(o[1], o[2], o[3], o[4], o[5], o[6]); break;
      case "r": {
        const [, x0, y0, x1, y1, c, t] = o;
        if (t < 0) { ctx.fillStyle = c; ctx.fillRect(x0, y0, x1 - x0 + 1, y1 - y0 + 1); }
        else { const f = t % 2 ? 0.5 : 0; ctx.strokeStyle = c; ctx.lineWidth = t; ctx.lineJoin = "miter"; ctx.strokeRect(x0 + f, y0 + f, x1 - x0, y1 - y0); }
        break;
      }
      case "c": {
        const [, x, y, r, c, t] = o;
        ctx.beginPath(); ctx.arc(x + 0.5, y + 0.5, t < 0 ? r + 0.5 : r, 0, 2 * Math.PI);
        if (t < 0) { ctx.fillStyle = c; ctx.fill(); } else { ctx.strokeStyle = c; ctx.lineWidth = t; ctx.stroke(); }
        break;
      }
      case "t": {     // Fitted to the box OpenCV gave this text, so labels sit in their chips and right-aligned values line up.
        const [, x, y, s, c, t, tw, th] = o;
        ctx.font = `${t > 1 ? 700 : 500} ${Math.max(th / 0.716, 6)}px ${FONT}`;
        const m = ctx.measureText(s).width;
        if (!m) break;
        ctx.save(); ctx.translate(x, y); ctx.scale(tw / m, 1); ctx.fillStyle = c; ctx.fillText(s, 0, 0); ctx.restore();
        break;
      }
      case "a": {     // Same tips as cv2.arrowedLine.
        const [, x0, y0, x1, y1, c, t, tip] = o;
        seg(x0, y0, x1, y1, c, t);
        const len = Math.hypot(x0 - x1, y0 - y1) * tip, ang = Math.atan2(y0 - y1, x0 - x1);
        for (const k of [1, -1]) seg(x1, y1, Math.round(x1 + len * Math.cos(ang + k * Math.PI / 4)), Math.round(y1 + len * Math.sin(ang + k * Math.PI / 4)), c, t);
        break;
      }
      case "p": {     // An open line through the points [x0, y0, x1, y1, ...].
        const [, pts, c, t] = o;
        if (pts.length < 4) break;
        const f = t % 2 ? 0.5 : 0;
        ctx.strokeStyle = c; ctx.lineWidth = t; ctx.lineCap = "round"; ctx.lineJoin = "round";
        ctx.beginPath(); ctx.moveTo(pts[0] + f, pts[1] + f);
        for (let i = 2; i < pts.length; i += 2) ctx.lineTo(pts[i] + f, pts[i + 1] + f);
        ctx.stroke();
        break;
      }
      case "d": { const [, x0, y, x1, c, dash, gap, t] = o; for (let x = x0; x < x1; x += dash + gap) seg(x, y, Math.min(x + dash, x1), y, c, t); break; }
      case "m": {     // Line mask (np.packbits rows): its pixels in the color at alpha, the rest untouched.
        const [, top, k, c, a, w, h] = o, bits = blobs[k];
        if (!bits) break;     // A browser without DecompressionStream shows the view without the line tint.
        if (tint.width !== w || tint.height !== h) { tint.width = w; tint.height = h; }
        if (!tintData || tintData.width !== w || tintData.height !== h) tintData = tctx.createImageData(w, h);
        else tintData.data.fill(0);       // Reused every frame instead of a new half-megabyte buffer.
        const im = tintData, d = im.data, [r, g, b] = rgb(c), row = (w + 7) >> 3;
        for (let y = 0, i = 0; y < h; y++) {
          const base = y * row;
          for (let x = 0; x < w; x++, i += 4) {
            if ((bits[base + (x >> 3)] >> (7 - (x & 7))) & 1) { d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = 255; }
          }
        }
        tctx.putImageData(im, 0, 0);
        ctx.globalAlpha = a; ctx.drawImage(tint, 0, top); ctx.globalAlpha = 1;
        break;
      }
      case "f": {     // A filled polygon [x0, y0, x1, y1, ...] at alpha.
        const [, pts, c, a] = o;
        if (pts.length < 6) break;
        ctx.globalAlpha = a; ctx.fillStyle = c; ctx.beginPath(); ctx.moveTo(pts[0], pts[1]);
        for (let i = 2; i < pts.length; i += 2) ctx.lineTo(pts[i], pts[i + 1]);
        ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
        break;
      }
      case "s": { const [, x0, y0, x1, y1, c, a] = o; ctx.globalAlpha = a; ctx.fillStyle = c; ctx.fillRect(x0, y0, x1 - x0 + 1, y1 - y0 + 1); ctx.globalAlpha = 1; break; }
    }
  }
}
async function inflate(bytes) {
  if (!window.DecompressionStream) return null;
  return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
}
async function show(buf) {
  const u8 = new Uint8Array(buf);
  if (u8.length < 8 || String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== "AIC1") throw new Error("bad view");
  const n = new DataView(buf).getUint32(4), head = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n)));
  let at = 8 + n;
  const raw = head.blobs.map(len => { const b = u8.subarray(at, at + len); at += len; return b; });
  const jpegs = new Set(head.pics.filter(p => typeof p[2] === "number").map(p => p[2]));
  for (const k of Object.values(head.minis || {})) jpegs.add(k);
  const blobs = await Promise.all(raw.map((b, k) => jpegs.has(k) ? createImageBitmap(new Blob([b], {type: "image/jpeg"})) : inflate(b)));
  if (cv.width !== head.w || cv.height !== head.h) { cv.width = head.w; cv.height = head.h; }
  for (const [x, w, src] of head.pics) {      // The camera pictures, and plain panels as their color.
    if (typeof src === "number") ctx.drawImage(blobs[src], x, 0);
    else { ctx.fillStyle = src; ctx.fillRect(x, 0, w, head.h); }
  }
  for (const [x, w, ops] of head.parts) {     // Each piece of a side-by-side view is clipped to its own area, as on the Pi.
    ctx.save(); ctx.beginPath(); ctx.rect(x, 0, w, head.h); ctx.clip(); ctx.translate(x, 0);
    paint(ops, blobs);
    ctx.restore();
  }
  if (head.state) {
    const m = {};
    for (const [name, k] of Object.entries(head.minis || {})) m[name] = blobs[k];
    dashboard(head.state, m);
  }
  for (const k of jpegs) blobs[k].close();
  drawn = true;
}
// --- The drive dashboard: the Pi sends its values (head.state) and close-ups (head.minis), the page lays them out. ---
function esc(t) { return String(t).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]); }
const miniSeen = {};      // Last time each close-up had a picture: a lamp or a figure missed for a frame or two keeps its picture.
function mini(id, bmp) {
  const c = $(id), x = c.getContext("2d"), now = performance.now();
  if (!bmp) {
    if (now - (miniSeen[id] || -1e9) < 700) return;      // Keep the last one a moment, so it does not flicker.
    c.parentElement.classList.add("none");
    x.fillStyle = "#000"; x.fillRect(0, 0, c.width, c.height);
    return;
  }
  miniSeen[id] = now;
  c.parentElement.classList.remove("none");
  x.fillStyle = "#000"; x.fillRect(0, 0, c.width, c.height);
  // Fills the box (cover): a close-up cut at the frame edge is not 4:3, and black bands would make it look smaller.
  const k = Math.max(c.width / bmp.width, c.height / bmp.height), w = bmp.width * k, h = bmp.height * k;
  x.drawImage(bmp, (c.width - w) / 2, (c.height - h) / 2, w, h);
}
let piFps = null;
function dashboard(S, minis) {
  document.body.classList.add("dash");
  piFps = S.fps;
  const st = S.step;
  $("stepn").textContent = `${st.i} / ${st.n}`;
  $("el").textContent = st.el.toFixed(1) + "s" + (st.lim ? ` / ${st.lim}s` : "");
  $("title").textContent = st.title;
  $("limb").style.display = st.lim ? "" : "none";      // How much of the step's time limit is used.
  if (st.lim) { $("lim").style.width = Math.min(100, 100 * st.el / st.lim).toFixed(1) + "%"; $("lim").style.background = st.el / st.lim > 0.85 ? "var(--amb)" : "var(--dim)"; }
  $("until").textContent = st.lines.find(l => l.startsWith("until")) || st.lines[0] || "";
  $("spd").textContent = S.speed.toFixed(2);
  $("spd").style.color = S.speed > 0 ? "var(--txt)" : "var(--red)";
  $("spdb").style.width = Math.min(100, 100 * S.speed / 0.6).toFixed(1) + "%";     // Same scale as the drawn panel.
  $("spdb").style.background = S.speed > 0 ? "var(--g)" : "var(--red)";
  const a = S.steer, f = Math.max(-1, Math.min(1, a / S.steer_max));
  $("str").textContent = (a < -0.5 ? "L " : a > 0.5 ? "R " : "") + Math.abs(a).toFixed(0) + " deg";
  Object.assign($("strb").style, {left: (50 + Math.min(0, f) * 50) + "%", width: Math.abs(f) * 50 + "%"});
  // The car in its lane: offset -1 / +1 is the left / right edge of the position range (the steering unit).
  const pos = $("pos");
  if (S.lane) {
    pos.style.display = ""; pos.style.left = (50 + 50 * Math.max(-1, Math.min(1, S.lane[0]))).toFixed(1) + "%";
    pos.style.background = Math.abs(S.lane[0]) > 0.5 ? "var(--amb)" : "#fff";
    $("posv").textContent = Math.abs(S.lane[0]) < 0.05 ? "center" : (S.lane[0] < 0 ? "left " : "right ") + Math.abs(S.lane[0]).toFixed(2);
  } else { pos.style.display = "none"; $("posv").textContent = "lane lost"; }
  // The status marks as in the drawn panel: colored only when on.
  $("flags").innerHTML = S.flags.map(([n, on, c, v]) => {
    const dark = on && parseInt(c.slice(1, 3), 16) + parseInt(c.slice(3, 5), 16) + parseInt(c.slice(5, 7), 16) > 380;
    return `<div class="flag${on ? " on" : ""}${dark ? " dark" : ""}"${on ? ` style="background:${c}"` : ""}><span>${esc(n)}</span><span>${v == null ? "" : esc(v)}</span></div>`;
  }).join("");
  const al = $("alert");
  if (S.banner) { al.textContent = S.banner[0]; al.style.background = S.banner[1]; al.style.display = "block"; }
  else al.style.display = "none";
  const L = S.light, lt = $("tLight");
  lt.textContent = L.state ? L.state.toUpperCase() : "-";
  lt.style.color = L.state === "red" ? "#ff2828" : L.state === "green" ? "#3cdc3c" : "var(--dim)";
  mini("mLight", minis.light);
  const P = S.person, pt = $("tPerson");
  $("cPerson").style.display = P ? "" : "none";      // Figure detection off (PERSON_ON = False): nothing to show.
  pt.textContent = P ? (P.state === "CLEAR" ? "-" : P.state) : "off";
  pt.style.color = !P || P.state === "CLEAR" ? "var(--dim)" : P.state === "SEEN" ? "var(--amb)" : "var(--red)";
  mini("mPerson", minis.person);
}
function full() { document.fullscreenElement ? document.exitFullscreen() : wrap.requestFullscreen?.(); }
$("bPause").onclick = () => setPaused(!paused); $("bSave").onclick = save; $("bFull").onclick = full; wrap.ondblclick = full;
// For a program that reads keys, keys pressed in this window go straight to it with no click. Otherwise only Ctrl+C goes, and the rest are view shortcuts.
// Read by physical key (code), so w is w even with a Korean input method on.
const KEY = "__KEY__", PROGRAM = "__PROGRAM__", CONFIRM = __CONFIRM__;      // An empty KEY is view only.
const KEYS_TO_PROGRAM = __KEYS_TO_PROGRAM__ && !!KEY;
const KEY_CODES = {Minus: ["-", "_"], Equal: ["=", "+"]};
let keyQueue = Promise.resolve(), armed = null, armedAt = 0;
function postKeys(text, ms) {
  return fetch("/key", {method: "POST", body: text, signal: timeout(ms),
                        headers: {"X-AICAR-Key": KEY, "X-AICAR-Program": PROGRAM}});
}
function sendKeys(text) {
  if (!text) return;
  keyQueue = keyQueue.then(() => postKeys(text, 1500)).then(r => {
    if (r.status === 409) { location.reload(); return; }      // Another program started meanwhile: keys must not go to it.
    if (!r.ok) throw new Error(r.status);
  }).catch(() => toast("Could not send the key · the program may have stopped"));
}
async function sendStop() {     // Sent at once, without waiting for an earlier key, a few times until it gets through.
  for (let i = 0; i < 5; i++) {
    try {
      if ((await postKeys("\\x03", 1000)).ok) { toast("Ctrl+C sent"); return; }
    } catch (e) {}
    await new Promise(ok => setTimeout(ok, 200));
  }
  toast("Could not send Ctrl+C · run aicar stop on the Mac");
}
function pressKey(text, repeat) {   // Keys that cannot be undone, such as save and quit, must be pressed again within 2 s.
  if (!CONFIRM.includes(text.toLowerCase())) { armed = null; sendKeys(text); return; }
  if (repeat) return;
  if (armed === text && Date.now() - armedAt < 2000) { armed = null; sendKeys(text); return; }
  armed = text; armedAt = Date.now();
  toast((text === "\\n" ? "Enter" : text) + " again to confirm");
}
function keyText(e) {
  if (e.code === "Enter") return "\\n";
  if (e.code === "Space") return " ";
  const letter = /^Key([A-Z])$/.exec(e.code);
  if (letter) return e.shiftKey ? letter[1] : letter[1].toLowerCase();
  const digit = /^Digit([0-9])$/.exec(e.code);
  if (digit && !e.shiftKey) return digit[1];
  if (KEY_CODES[e.code]) return KEY_CODES[e.code][e.shiftKey ? 1 : 0];
  return e.key.length === 1 && e.key <= "~" ? e.key : "";
}
addEventListener("keydown", e => {
  if (e.metaKey || e.altKey) return;
  if (e.ctrlKey) {
    if (KEY && !e.repeat && (e.code === "KeyC" || e.key === "c" || e.key === "C")) { e.preventDefault(); sendStop(); }
    return;
  }
  if (KEYS_TO_PROGRAM) {
    const text = keyText(e);
    if (text) { e.preventDefault(); pressKey(text, e.repeat); }     // Also keeps the button just clicked from being pressed again by Space or Enter.
  } else if (e.code === "Space") { e.preventDefault(); if (!e.repeat) setPaused(!paused); }
  else if (e.repeat) return;
  else if (e.code === "KeyS") save();
  else if (e.code === "KeyF") full();
});
function showFocus() {
  if (!KEY) return;
  const on = document.hasFocus();
  $("kstate").textContent = on ? "taking keys" : "click the window once to send keys"; $("kstate").style.color = on ? "var(--ok)" : "var(--warn)";
}
addEventListener("focus", showFocus); addEventListener("blur", showFocus); showFocus();
if (KEYS_TO_PROGRAM) document.querySelectorAll("kbd").forEach(k => k.remove());
if (!KEYS_TO_PROGRAM) $("bEnter").style.display = "none";
if (!KEY) {
  $("bStop").style.display = "none";
  $("klabel").textContent = "View only"; $("kstate").textContent = "keys can be sent only from the address aicar opened"; $("kstate").style.color = "var(--dim)";
}
$("bEnter").onclick = () => pressKey("\\n", false); $("bStop").onclick = sendStop;
// A box dragged on the left camera view goes to the program, as fractions (0-1) of the camera view.
const DRAG = __DRAG__;
if (DRAG && KEY) {
  const mark = $("drag");
  let from = null, area = null;
  const clip = v => Math.min(Math.max(v, 0), 1);
  const spot = e => [(e.clientX - area.x) / area.w, (e.clientY - area.y) / area.h];
  function picture() {      // The picture keeps its aspect and is centered, so it can be smaller than the img box.
    const r = cv.getBoundingClientRect(), s = Math.min(r.width / cv.width, r.height / cv.height);
    const w = cv.width * s, h = cv.height * s;
    return {x: r.left + (r.width - w) / 2, y: r.top + (r.height - h) / 2, w: w * DRAG, h};
  }
  function square(e) {
    const [x, y] = spot(e).map(clip);
    return [Math.min(from[0], x), Math.min(from[1], y), Math.max(from[0], x), Math.max(from[1], y)];
  }
  wrap.classList.add("drag");
  wrap.addEventListener("pointerdown", e => {
    if (e.button !== 0 || !drawn) return;
    area = picture();
    const [x, y] = spot(e);
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    if (paused) return toast("Cannot mark while paused");
    e.preventDefault(); from = [x, y]; wrap.setPointerCapture(e.pointerId);
  });
  wrap.addEventListener("pointermove", e => {
    if (!from) return;
    const [x0, y0, x1, y1] = square(e), w = wrap.getBoundingClientRect();
    Object.assign(mark.style, {display: "block", left: area.x - w.left + x0 * area.w + "px", top: area.y - w.top + y0 * area.h + "px",
                               width: (x1 - x0) * area.w + "px", height: (y1 - y0) * area.h + "px"});
  });
  wrap.addEventListener("pointercancel", () => { from = null; mark.style.display = "none"; });
  wrap.addEventListener("pointerup", e => {
    if (!from) return;
    const [x0, y0, x1, y1] = square(e);
    from = null; mark.style.display = "none";
    if (x1 - x0 < 0.015 || y1 - y0 < 0.02) return;      // A click, not a drag.
    fetch("/box", {method: "POST", body: [x0, y0, x1, y1].map(v => v.toFixed(4)).join(","), signal: timeout(1500),
                   headers: {"X-AICAR-Key": KEY, "X-AICAR-Program": PROGRAM}}).then(r => {
      if (r.status === 409) { location.reload(); return; }
      if (!r.ok) throw new Error(r.status);
      toast("Marked · views are taken in this box");
    }).catch(() => toast("Could not send · the program may have stopped"));
  });
  wrap.ondblclick = null;       // So the end of a drag does not go full screen. Full screen is on the button.
}
// Paused or in a hidden tab: no views are fetched, so the Pi draws and encodes nothing. Paused still checks the connection.
const idle = ms => new Promise(ok => { wake = ok; setTimeout(ok, ms); });
document.addEventListener("visibilitychange", () => { if (!document.hidden && wake) wake(); });
async function loop() {
  for (;;) {
    try {
      if (document.hidden) { await idle(60000); continue; }
      const r = await fetch(paused ? "/ping" : "/frame.bin?after=" + seq, {cache: "no-store", signal: timeout(3000)});
      const program = r.headers.get("X-AICAR-Program");
      if (program && program !== PROGRAM) { location.reload(); return; }     // Another program started: reopen with its keys and help.
      if (paused) {
        if (!r.ok) throw new Error(r.status);
        fails = 0; msg.style.display = "none";
        status(true, "paused · view from " + Math.round((Date.now() - pausedAt) / 1000) + " s ago");
        await idle(1000); continue;
      }
      if (r.status === 204) {     // No new view for 1 s (the program is running)
        fails = 0; msg.style.display = "none"; status(true, "connected · no new frame"); continue;
      }
      if (!r.ok) throw new Error(r.status);
      const next = +r.headers.get("X-Seq"), buf = await r.arrayBuffer();
      if (paused) continue;       // Paused while this one was on its way: keep the view as it is.
      await show(buf);
      seq = next; fails = 0; msg.style.display = "none";
      const now = performance.now(); shown.push(now); shown = shown.filter(x => now - x < 2000);
      status(true, "connected · " + (shown.length / 2).toFixed(0) + " fps" + (piFps ? ` · Pi ${piFps} fps` : ""));
      if (piFps && piFps < 15) { stt.style.color = "var(--amb)"; } else stt.style.color = "";      // A slow Pi steers late.
    } catch (e) {
      fails++;
      if (fails >= 2) { msg.style.display = "flex"; status(false, "disconnected · reconnecting"); }
      await new Promise(ok => setTimeout(ok, Math.min(300 * fails, 2000)));
    }
  }
}
loop();
</script></body></html>"""

    def __init__(self, port, title="AICAR", keys_help=None, confirm="", key=None, serve=True, drag_share=0):
        """keys_help: the keys this program reads, shown under the view. With it, keys pressed on the page go straight to the program, with no click.

        confirm: of those, the ones that save or quit. The page asks for them twice, so a stray key cannot.
        A browser sends keys only with the secret in the address aicar opens (web_key). Without it the page only shows the view.

        drag_share: how much of the picture's width, from the left, is the camera view a box can be dragged on (read_box). 0 turns dragging off.
        """
        import logging
        import flask.cli
        from flask import Flask, Response, request
        from werkzeug.serving import make_server
        logging.getLogger("werkzeug").setLevel(logging.ERROR)   # Turn off the request log printed for every frame.
        flask.cli.show_server_banner = lambda *_: None          # And the startup lines.
        self.img, self.seq = None, 0            # Latest view (Canvas or ndarray) and its number, written by the main loop.
        self.jpg, self.jpg_seq = None, -1       # Last painted JPEG (legacy /frame.jpg, /video) and the view number it came from.
        self.bin, self.bin_seq = None, -1       # Last page payload (/frame.bin) and the view number it came from.
        self.lock = threading.Lock()
        self.new_frame = threading.Condition()  # Wakes the requests waiting for a newer view.
        self.last_request = -1e9                # Time a browser last fetched a frame (used for watching).
        self.keys = KeyInbox()
        self.dragged = None                     # Last box dragged on the page, not yet read.
        self.key = web_key() if key is None else key
        self.program = secrets.token_hex(4)     # Tells this program from whichever one a page was opened for.
        app = Flask(__name__)
        page = (self.PAGE.replace("__TITLE__", title).replace("__KEYS_TO_PROGRAM__", "true" if keys_help else "false")
                .replace("__KEYS_HELP__", keys_help or "").replace("__PROGRAM__", self.program)
                .replace("__CONFIRM__", json.dumps(confirm)).replace("__DRAG__", json.dumps(drag_share)))

        @app.route("/")
        def index():
            allowed = hmac.compare_digest(request.args.get("key", ""), self.key)
            return Response(page.replace("__KEY__", self.key if allowed else ""), mimetype="text/html", headers={"Cache-Control": "no-store"})

        def latest(get, mimetype):
            # Wait up to 1 second for a view newer than after, then send only the latest one.
            self.last_request = time.monotonic()
            after = request.args.get("after", -1, type=int)
            if after > self.seq:        # The program restarted and the number went back to the start, so just send the latest view.
                after = -1
            with self.new_frame:
                self.new_frame.wait_for(lambda: self.img is not None and self.seq > after, timeout=1.0)
            data, seq = get()
            if data is None or seq <= after:
                return Response(status=204, headers={"X-AICAR-Program": self.program})
            return Response(data, mimetype=mimetype,
                            headers={"X-Seq": str(seq), "X-AICAR-Program": self.program, "Cache-Control": "no-store"})

        @app.route("/frame.bin")
        def frame_bin():
            """The page's view: picture plus drawing commands (overlay.Canvas.payload)."""
            return latest(self.payload, "application/octet-stream")

        @app.route("/frame.jpg")
        def frame():
            """The view painted on the Pi, for other programs that open the URL directly."""
            return latest(self.encoded, "image/jpeg")

        @app.route("/ping")
        def ping():
            """Connection check while the page is paused. Does not count as watching, so nothing is drawn or encoded."""
            return Response(status=204, headers={"X-AICAR-Program": self.program, "Cache-Control": "no-store"})

        @app.route("/key", methods=["POST"])
        def key():
            if not hmac.compare_digest(request.headers.get("X-AICAR-Key", ""), self.key):
                return Response(status=403)
            if (request.content_length or 0) > 32:
                return Response(status=413)
            text = request.get_data(as_text=True)
            # A page opened for an earlier program must not press this one's keys. It can still stop it.
            if request.headers.get("X-AICAR-Program") != self.program and text != "\x03":
                return Response(status=409)
            self.keys.add(text)
            return Response(status=204)

        @app.route("/box", methods=["POST"])
        def box():
            if not drag_share:
                return Response(status=404)
            if not hmac.compare_digest(request.headers.get("X-AICAR-Key", ""), self.key):
                return Response(status=403)
            if request.headers.get("X-AICAR-Program") != self.program:
                return Response(status=409)
            if (request.content_length or 0) > 64:
                return Response(status=413)
            try:
                x0, y0, x1, y1 = (float(v) for v in request.get_data(as_text=True).split(","))
            except ValueError:
                return Response(status=400)
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):       # Also refuses nan.
                return Response(status=400)
            self.dragged = (x0, y0, x1, y1)
            return Response(status=204)

        @app.route("/video")
        def video():
            """Legacy MJPEG stream, for other programs that open the URL directly."""
            def gen():
                last = -1
                while True:
                    time.sleep(0.05)
                    self.last_request = time.monotonic()
                    jpg, seq = self.encoded()
                    if jpg is not None and seq != last:
                        last = seq
                        yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

        self.app = app
        if serve:
            try:
                server = make_server("0.0.0.0", port, app, threaded=True)
            except (OSError, SystemExit) as e:   # Otherwise the page and its keys would still reach whichever program holds the port.
                raise SystemExit(f"port {port} is in use by another program. Run aicar stop first ({e})")
            threading.Thread(target=server.serve_forever, daemon=True).start()

    @property
    def watching(self):
        """Whether a browser fetched a frame in the last 2 seconds. If nobody is watching, the view does not need to be drawn."""
        return time.monotonic() - self.last_request < 2.0

    def read_keys(self, terminal=True):
        """Keys pressed since the last call: on the web page, then in the terminal the program was started from."""
        keys = self.keys.take()
        if terminal and select.select([sys.stdin], [], [], 0)[0]:
            keys += sys.stdin.read(1)
        return keys

    def read_box(self):
        """The box dragged on the page since the last call, as (left, top, right, bottom) fractions of the camera view, or None."""
        box, self.dragged = self.dragged, None
        return box

    def update(self, view):
        """Store the view without drawing or encoding it: an overlay.Canvas (commands the browser paints) or a plain ndarray.

        Neither the view nor its base picture may be modified afterward.
        """
        with self.new_frame:
            self.img = view
            self.seq += 1
            self.new_frame.notify_all()

    def payload(self):
        """Page payload of the latest view. Built only when the view is new, once even if several browsers are connected."""
        with self.lock:
            view, seq = self.img, self.seq
            if view is not None and seq != self.bin_seq:
                self.bin = (view if isinstance(view, Canvas) else Canvas(view)).payload()
                self.bin_seq = seq
            return self.bin, self.bin_seq

    def encoded(self):
        """JPEG of the latest view painted on the Pi (legacy URLs only). Encodes only when the view is new."""
        with self.lock:
            view, seq = self.img, self.seq
            if view is not None and seq != self.jpg_seq:
                self.jpg = cv2.imencode(".jpg", np.asarray(view), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                self.jpg_seq = seq
            return self.jpg, self.jpg_seq


class Recorder:
    # Settings a replay needs, besides the calibration and the course.
    """Save raw frames and a log under runs/<time>/. Replay them on a Mac with replay.py.

    JPEGs are written on a separate thread so the drive loop does not wait. close() finishes whatever is left.
    """

    def __init__(self, base_speed=None, settings=()):
        self.dir = Path(__file__).with_name("runs") / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True)
        if PERSON_ON and PERSON_FILE.exists():      # Replaying this run later needs the figure registered on the day.
            shutil.copy(PERSON_FILE, self.dir / PERSON_FILE.name)
            if PERSON_FILE.with_name(PHOTO_DIR).is_dir():
                shutil.copytree(PERSON_FILE.with_name(PHOTO_DIR), self.dir / PHOTO_DIR)
        if VISION_FILE.exists():                    # And the calibration of the day: the Mac has none of its own.
            shutil.copy(VISION_FILE, self.dir / VISION_FILE.name)
        for f in LANDMARK_FILES:        # And the registered light, crosswalk, lines and stop points.
            if f.exists():
                shutil.copy(f, self.dir / f.name)
        shutil.copy(Path(__file__).with_name("course.py"), self.dir / "course.py")      # And the course as it was.
        (self.dir / "run.json").write_text(json.dumps({
            "base_speed": base_speed, "set": list(settings), "lane_follow_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest()}))
        self.log = open(self.dir / "log.csv", "w")
        self.log.write("frame,t,offset,heading,junction,steer,speed,step,status,cmd\n")
        self.n = 0
        self.queue = queue.Queue(maxsize=300)   # If saving falls behind (about 5 seconds), the loop waits briefly. Frames are not dropped.
        self.writer = threading.Thread(target=self._save_frames, daemon=True)
        self.writer.start()
        print(f"Recording: {self.dir}")

    def _save_frames(self):
        while (item := self.queue.get()) is not None:
            path, frame = item
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def write(self, frame, t, det, junction, steer, speed, course, now, cmd=0.0):
        self.queue.put((str(self.dir / f"{self.n:05d}.jpg"), frame))
        off, head = det if det else ("", "")
        self.log.write(f"{self.n},{t:.3f},{off},{head},{int(junction)},{steer:.2f},{speed:.2f},"
                       f"{course.idx},{course.status(now)},{cmd:.2f}\n")
        self.n += 1

    def close(self):
        self.queue.put(None)
        self.writer.join()
        self.log.close()


LOG_KEEP = 30       # DriveLog: this many past drives are kept in logs/. Older ones are deleted when a drive starts.


class DriveLog:
    """Every drive leaves a small record in logs/<time>/, whether or not it is recorded with --record. No pictures, so it costs
    next to nothing, but it tells afterwards what the car decided and why an action ended (or never did):
      out.log     everything the drive printed (each action's start and end, measurements, warnings, errors)
      log.csv     one row per frame: lane position, steering, motor command, action, and what was in view
      course.py   the course as it was;  run.json  the options;  ends.csv / stops.csv  like a recording, when there are any
    Fetch one to the Mac with `aicar fetch-log`.
    """
    HEADER = "t,offset,heading,junction,steer,speed,cmd,step,status,light,obstacle,junction_y,gap_y,crosswalk_y,end_y,person\n"

    class Tee:
        """Writes to the real stream and to the log file. A fault in the file must never stop the drive."""

        def __init__(self, stream, file):
            self.stream, self.file = stream, file

        def write(self, text):
            try:
                self.file.write(text)
            except (OSError, ValueError):
                pass
            return self.stream.write(text)

        def flush(self):
            try:
                self.file.flush()
            except (OSError, ValueError):
                pass
            self.stream.flush()

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def __init__(self, options=None, root=None, keep=LOG_KEEP):
        root = Path(__file__).with_name("logs") if root is None else Path(root)
        root.mkdir(parents=True, exist_ok=True)
        old = sorted(d for d in root.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[8] == "-")
        for d in old[:max(len(old) - (keep - 1), 0)]:
            shutil.rmtree(d, ignore_errors=True)
        self.dir = root / time.strftime("%Y%m%d-%H%M%S")
        n = 1
        while self.dir.exists():        # Two drives started in the same second.
            n += 1
            self.dir = root / f"{time.strftime('%Y%m%d-%H%M%S')}-{n}"
        self.dir.mkdir()
        course_file = Path(__file__).with_name("course.py")
        if course_file.exists():
            shutil.copy(course_file, self.dir / "course.py")
        (self.dir / "run.json").write_text(json.dumps(dict(options or {}, lane_follow_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest())))
        self.out = open(self.dir / "out.log", "w", buffering=1)     # Line buffered: the last lines survive a crash or power loss.
        self.rows = open(self.dir / "log.csv", "w")
        self.rows.write(self.HEADER)
        self.n = 0
        self.streams = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = self.Tee(sys.stdout, self.out), self.Tee(sys.stderr, self.out)

    def write(self, t, driver, cmd, now):
        """One row for this frame. Never raises: the log is not worth a stopped car."""
        try:
            det, detect, course = driver.det, driver.detect, driver.course
            off, head = (f"{det[0]:.4f}", f"{det[1]:.4f}") if det else ("", "")
            y = lambda v: "" if v is None else f"{v:.0f}"
            self.rows.write(f"{t:.3f},{off},{head},{int(bool(driver.junction))},{driver.steering.angle:.2f},{driver.speed:.2f},{cmd:.2f},"
                            f"{course.idx + 1},{course.status(now)},{driver.light or ''},{int(bool(driver.obs and driver.obs.near))},"
                            f"{y(detect.junction_y)},{y(detect.gap_y)},{y(detect.crosswalk_y)},{y(detect.end_y)},"
                            f"{int(bool(driver.person and driver.person.seen))}\n")
            self.n += 1
            if self.n % 100 == 0:
                self.rows.flush()
        except Exception:
            pass

    def close(self, driver=None):
        if driver is not None:
            try:
                if driver.ends.rows:
                    driver.ends.save(self.dir / "ends.csv")
                driver.save_stop_log(self.dir / "stops.csv")
            except Exception:
                pass
        sys.stdout, sys.stderr = self.streams
        for f in (self.rows, self.out):
            try:
                f.close()
            except OSError:
                pass


class Driver:
    """One camera frame to (steering angle, speed). Real driving (main) and the simulator both use this."""

    def __init__(self, course, start=1, base_speed=BASE_SPEED, obstacle=None, person=None):
        self.detect, self.traffic, self.steering = LaneDetector(), TrafficLight(), Steering()
        self.traffic_view = TrafficLight()      # The light shown on the web page when not waiting for green: kept apart from
                                                # self.traffic, so watching the page cannot change what wait_green decides.
        self.course = CourseRunner(course, start)
        self.base_speed = base_speed
        self.obs = obstacle
        if person is not None:
            person.place = self.detect.in_lane if PERSON_IN_LANE else None
        self.person = None if person is None else PersonWatch(   # person is a PersonDetector. None drives without looking for the figure.
            person, FRAME_H, PERSON_AT, PERSON_X, PERSON_RESUME_DELAY, PERSON_MIN_STOP, PERSON_MAX_WAIT, PERSON_EVERY)
        self.ends = EndWatch()      # Measurement only: where each to_junction / to_end really ended.
        self.last_cmd = 0.0
        self.lost_since = None      # Time the lane was lost.
        self.yaw, self.yaw_idx = YawMeter(), -1     # Turn angle from the camera, reset at each turn action.
        self.approach = (0.0, -1.0, -1)    # slow=: (slowdown amount 0 to 1, time the line was last seen, action index).
        self.history = []           # Recent forward driving: (seconds, steering angle, speed). Replayed backwards when backing up.
        self.last_t = None
        self.back = None            # Back-up in progress: start, plan [(seconds, angle)], entries used, lost episode, action index.
        self.back_tries = {}        # Action index: back-ups used.
        self.back_lost = None       # lost_since of the episode that already backed up (one back-up per loss).
        self.cw_since = None
        self.speed = 0.0
        self.edge = False
        self.done = False
        self.det, self.junction, self.light = None, False, None
        self.light_tick = 0         # Frames since start, for LIGHT_VIEW_EVERY.
        self.hold_banner = None     # Start button: (text, color) shown instead of the usual banner while waiting or paused.
        self.red = None             # wait_green looking for red: action index, back-ups used, when this look began, back-up in progress.
        self.hline_t = None         # While aligning: last time heading said the horizontal line in view is a real one.
        self.aim = None             # STEER_MODE "pursuit": the target point on screen this frame (for the view).
        self.stop_found = None      # go_to("A"): where the registered shape was found this frame ({"x","y","w","h","score","scale"}), or None.
        self.stop_y = self.stop_target = None   # Its center, and where the center must come to (frame y).
        self.stop_idx, self.stop_last, self.stop_tick, self.stop_hist = -1, {}, 0, {}
        self.stop_log = []      # go_to(): one row per frame and point, saved with a recording as stops.csv (what was matched, how well, how far).

    def retrace_plan(self, seconds):
        """Recent forward driving played backwards: [(time it ends, steering angle)], up to this many seconds of reversing."""
        plan, t = [], 0.0
        for dt, angle, speed in reversed(self.history):     # Newest first: undo the last stretch first.
            t += dt * float(np.clip(speed / LOST_BACK_SPEED, 0.3, 2.0))
            plan.append((t, angle))
            if t >= seconds:
                break
        return plan or [(seconds, self.steering.angle)]

    def end_red_back(self, red, now):
        """That stretch of forward driving is behind the car now."""
        used = sum(1 for t_end, _ in red["back"] if t_end <= now - red["back_t"])
        del self.history[len(self.history) - used:]
        red["back"], red["look"] = None, now

    def search_red(self, now, hold):
        """wait_green with no red light found yet: back up a little and look again. Returns the steering angle while backing, otherwise None."""
        course, red = self.course, self.red
        if not RED_BACK_ON or not course.needs_light() or course.red_seen:
            if red is not None and red["back"] is not None:
                self.end_red_back(red, now)
            self.red = None
            return None
        if red is None or red["idx"] != course.idx:
            red = self.red = {"idx": course.idx, "tries": 0, "look": now, "back": None, "back_t": now}
        if hold or self.edge:       # The LEGO figure is in the way, or the mat ends: stand still and start the look over.
            if red["back"] is not None:
                self.end_red_back(red, now)
            red["look"] = now
            return None
        if red["back"] is not None:
            elapsed = now - red["back_t"]
            if elapsed < RED_BACK_SEC:
                return next((a for t_end, a in red["back"] if t_end > elapsed), red["back"][-1][1])
            self.end_red_back(red, now)
            return None
        if red["tries"] < RED_BACK_TRIES and now - red["look"] >= RED_LOOK_SEC:
            red["tries"] += 1
            red["back"], red["back_t"] = self.retrace_plan(RED_BACK_SEC), now
            print(f"\n[light] red not found -> back up {RED_BACK_SEC:g}s ({red['tries']}/{RED_BACK_TRIES})")
            return red["back"][0][1]
        return None

    def back_up(self, now, lost_long, steer):
        """Lost lane for LOST_HOLD: reverse along the recent path (steering history played backwards) until the lane comes back.

        Returns the steering to send (the retrace angle while backing, otherwise steer unchanged). self.back is set while backing.
        """
        course, back = self.course, self.back
        if back is not None:
            elapsed = now - back["start"]
            found = self.lost_since is None
            if found or elapsed >= back["len"] or course.idx != back["idx"] or course.stopped(now) or self.edge:
                used = sum(1 for t_end, _ in back["plan"] if t_end <= elapsed) if not found else back["used_at"]
                del self.history[len(self.history) - used:]   # That stretch is behind the car now.
                print(f"\n[back-up done] {'lane found again' if found else 'not found'} ({elapsed:.1f}s)")
                self.back = None
                return steer
            back["used_at"] = sum(1 for t_end, _ in back["plan"] if t_end <= elapsed)
            angle = next((a for t_end, a in back["plan"] if t_end > elapsed), back["plan"][-1][1])
            return self.steering.output(angle, now, 0.0)
        if (not lost_long or LOST_BACK_SEC <= 0 or course.stopped(now) or self.edge or self.lost_since == self.back_lost
                or self.back_tries.get(course.idx, 0) >= LOST_BACK_TRIES):
            return steer
        plan = self.retrace_plan(LOST_BACK_SEC)
        self.back = {"start": now, "plan": plan, "len": min(plan[-1][0], LOST_BACK_SEC), "used_at": 0, "idx": course.idx}
        self.back_lost = self.lost_since
        self.back_tries[course.idx] = self.back_tries.get(course.idx, 0) + 1
        print(f"\n[back-up] lane lost -> backing up the way it came for {self.back['len']:.1f}s ({self.back_tries[course.idx]}/{LOST_BACK_TRIES})")
        return self.steering.output(plan[0][1], now, 0.0)

    def find_stop(self, step, dbg, now=None, frame=None):
        """go_to("A") or go_to("A|B"): find the registered shapes in this frame's line mask, and where their center must come to.

        With several names, the one nearest to its stop counts: the car stops at whichever comes first.
        """
        self.stop_found = self.stop_y = self.stop_target = None
        names = step.get("stop") if step is not None and step["until"] == "_stop" else None
        names = [n for n in ([names] if isinstance(names, str) else names or []) if STOP_MODEL is not None and n in STOP_MODEL.points]
        if not names:
            self.stop_idx = -1
            return
        c = self.course
        if now is not None and c.step_start is not None and (
                now - c.step_start < (step.get("blind") or 0) or (c.gap_end is not None and now < c.gap_end)):
            return      # blind_sec / gap=: the stop is not counted now, so it is not searched for either (saves fps).
        top = int(FRAME_H * ROI_TOP)
        if self.stop_idx != self.course.idx:
            self.stop_last, self.stop_hist = {}, {}
        self.stop_idx = self.course.idx
        self.stop_tick += 1
        scale = CAMERA_F * LANE_CM / LANE_WIDTH_PX * max(342 - HORIZON_Y, 1.0)
        best = None
        # Boxes drawn above the line area are matched on the picture's edges there (only then is that computed).
        top_line = top
        view, top = (landmarks.stop_view(frame, self.detect.mask, top, STOP_MODEL.top_y(names)) if frame is not None
                     else (self.detect.mask, top))
        for name in names:
            last = self.stop_last.get(name)
            # Every frame near each view's place (passing over it may take only 1-2 frames); the wide look from afar every other frame.
            found = STOP_MODEL.find(name, view, top, last, wide=self.stop_tick % 2 == 0, key=self.stop_tick)
            if found is None and last is not None:
                found = STOP_MODEL.find(name, view, top, wide=self.stop_tick % 2 == 0, key=self.stop_tick)     # Lost it near the last place: look again.
            self.stop_last[name] = found
            # offset= cm farther (or nearer) than where it was registered: the floor distance of that place, moved.
            # Each view comes to where its own box was (views of one point may be boxes on different things).
            home_y = STOP_MODEL.view_y(name, found["view"]) if found is not None and "view" in found else STOP_MODEL.stop_y(name)
            dist = max(floor_cm(home_y) - (step.get("stop_cm") or 0.0), 1.0)
            target = float(min(HORIZON_Y + scale / dist, FRAME_H - 2))
            y = None if found is None else found["y"] + found["h"] / 2      # Its center, like the registered one.
            near = STOP_MODEL.near_views(name, top_line)
            if y is not None and near and found.get("view") not in near:
                y = min(y, target - 1)      # Found by a far view: it shows the point is ahead, but a near view decides the stop.
            if y is not None and not landmarks.StopModel.close_enough(found):
                y = min(y, target - 1)      # Road area followed but not matched closely yet: still coming, not there yet.
            left = None if y is None else floor_cm(y) - floor_cm(target)
            hist = self.stop_hist.setdefault(name, [])
            if left is None or now is None:
                hist.clear()
            elif landmarks.StopModel.close_enough(found) and found["score"] >= STOP_LEAD_SCORE and (not near or found.get("view") in near):
                need = STOP_WHOLE_FRAMES if found.get("whole") else JUNCTION_FRAMES
                if left <= stop_lead(hist, now, left, need):
                    y = max(y, target)      # Close enough at this speed: counted now, so the car comes to rest on it.
            else:
                stop_lead(hist, now, left, 1)   # Still keeps the speed reading going.
            if now is not None and len(self.stop_log) < 100000:
                f = found or {}
                self.stop_log.append(f"{now:.3f},{self.course.idx},{name},{f.get('view', '')},{int(f.get('view') in near) if f else ''},"
                                     f"{f.get('score', 0):.3f},{f.get('scale', 0):.3f},{'' if y is None else round(y, 1)},{target:.1f},"
                                     f"{'' if left is None else round(left, 2)},{int(y is not None and y >= target)}")
            if best is None or (left is not None and (best[3] is None or left < best[3])):
                best = (name, found, target, left, y)
            if dbg is not None and found is not None:
                x, fy, w, h = found["x"], found["y"], found["w"], found["h"]
                dbg.rectangle((x, fy), (x + w, fy + h), C_AT, 2)
                chip(dbg, f"{name} {max(left, 0):.1f}cm", x, max(fy - 22, 0), C_AT, 0.4)
        name, found, self.stop_target, _, self.stop_y = best
        self.stop_found = found
        if dbg is not None:
            dashed_hline(dbg, int(self.stop_target), C_AT, dash=10, gap=6)

    def save_stop_log(self, path):
        """go_to() matches of this drive, as stops.csv. Nothing is written when there was no go_to()."""
        if self.stop_log:
            Path(path).write_text("t,step,name,view,near,score,scale,y,target,left_cm,reached\n" + "\n".join(self.stop_log) + "\n")

    def stand_for_person(self, now):
        """Stand still for the LEGO figure. The action in progress waits as it is, so it carries on where it left off."""
        dt = now - self.last_t if self.last_t is not None else 0.0
        self.last_t = now
        self.course.shift_clock(dt)
        steering = self.steering
        if steering.out_t is not None:
            steering.out_t += dt
        if steering.prev_t is not None:
            steering.prev_t += dt
        level, seen_t, idx = self.approach
        self.approach = (level, seen_t + dt, idx)
        self.back, self.lost_since, self.cw_since = None, None, None
        self.speed = 0.0

    def pursuit_angle(self, course, base):
        """LANE_MODEL "track" with STEER_MODE "pursuit": the arc steering from the tracked lane, or None to use KP / KH."""
        t = self.detect.tracker
        if LANE_MODEL != "track" or STEER_MODE != "pursuit" or t is None or t.p is None or t.conf < TRACK_CONF_MIN:
            self.aim = None
            return None
        look = (LOOK_LW + LOOK_SPEED_LW * max(base, 0.0)) * STYLE_LOOK.get(course.opt("style", "normal"), 1.0)
        out = t.pursuit(sys.modules[__name__], look, WHEELBASE_LW, CAM_AHEAD_LW,
                        course.opt("shift", 0.0) * LANE_PX_PER_CM / REF_LANE_PX)      # shift= in lane widths, as the band method takes it.
        if out is None:
            self.aim = None
            return None
        angle, self.aim = out
        return float(np.clip(angle, -AUTO_STEER_MAX, AUTO_STEER_MAX))

    def approach_speed(self, base, now):
        """slow=: ease from base down to the slow speed as the line approaches.

        When a junction, crosswalk, or gap line first appears near the top, start at base and reduce toward the slow speed until the line reaches the end height (at=, or JUNCTION_Y if at= is unset).
        """
        course = self.course
        step = course.step()
        slow = course.opt("slow")
        if not getattr(_course, "SLOWDOWN_ON", True):
            return base         # course.py SLOWDOWN_ON = False: slow= is ignored, the set speed is kept to the end.
        if slow is None or step["until"] not in CourseRunner.AT_EVENTS + ("_stop",) or not course.final_approach():
            return base
        slow = min(max(slow, MIN_SPEED, APPROACH_MIN), base)      # Do not go below the speed that keeps the motor turning, nor speed up to get there.
        if course.run_out:      # cm=: stay slow for the last distance, even once the line has gone from view.
            return slow
        dist = course.laneend_dist(step)
        if dist is not None:    # Lane-end stop: full speed beyond SLOW_START_CM, then ease to slow at the target.
            target = course.laneend_target(step)
            level = float(np.clip((SLOW_START_CM - dist) / max(SLOW_START_CM - target, 1.0), 0.0, 1.0))
            self.approach = (level, now, course.idx)
            return base - (base - slow) * level
        level, seen_t, idx = self.approach
        if idx != course.idx:       # New action: start over.
            level, seen_t = 0.0, -1.0
        ends = [v for v in (self.detect.junction_y, self.detect.gap_y) if v is not None]
        y = {"junction": self.detect.junction_y, "crosswalk": self.detect.crosswalk_y, "_crosswalk": self.detect.xwalk2_y,
             "_stop": self.stop_y, "gap": self.detect.gap_y, "end": max(ends) if ends else None}[step["until"]]
        far = step["until"] in ("junction", "end")
        if y is None and far:
            y = self.detect.far_y       # Horizontal line seen in the distance, if it has not entered the ROI yet.
        if y is not None:
            start = FRAME_H * (FAR_TOP if far else ROI_TOP)
            end = self.stop_target if step["until"] == "_stop" and self.stop_target else (course.opt("at") or JUNCTION_Y) * FRAME_H
            level = max(level, float(np.clip((y - start) / max(end - start, 1.0), 0.0, 1.0)))
            seen_t = now
        elif now - seen_t > APPROACH_HOLD:
            level = 0.0
        self.approach = (level, seen_t, course.idx)
        return base - (base - slow) * level

    # --- Detection View (Browser Live View / Replay) ---
    PANEL_W = 240
    EVENT_EN = {"junction": "JUNCTION", "crosswalk": "CROSSWALK", "lane": "NEW LANE", "gap": "LANE ENDS", "end": "END (JUNCTION/GAP)", "crossed": "CROSSWALK PASSED",
                "green": "GREEN LIGHT", "obstacle": "OBSTACLE", "clear": "CLEAR", "lost": "LANE LOST",
                "aligned": "ALIGNED", "enter": "ENTER", "_crosswalk": "_CROSSWALK", "_crossed": "_CROSSWALK PASSED", "_stop": "STOP POINT"}
    DO_EN = {"follow": "FOLLOW LANE", "straight": "STRAIGHT", "back": "BACK UP", "right": "TURN RIGHT", "left": "TURN LEFT",
             "curve": "CURVE", "stop": "STOP", "wait": "WAIT", "finish": "FINISH"}

    def step_text(self, step):
        """One action to (large text, lines of small text)."""
        if step is None:
            return "COURSE END", ["follow lane"]
        title = self.DO_EN[step["do"]]
        if step["do"] in ("left", "right", "curve"):
            # right()/left() with no angle= keep None until they run (then AUTO_STEER_MAX), so the "next" line shows MAX.
            title += " MAX" if step["angle"] is None else f" {abs(step['angle']):.0f}"
        if step["sec"]:
            cond = f"for {step['sec']:g}s"
        elif step["until"]:
            cond = "until " + ("DEAD END" if step.get("dead_end") else self.EVENT_EN[step["until"]]) + (f" @{step['at']:g}" if step.get("at") else "")
            if step.get("cm") is not None:
                cond += f" +{step['cm']:g}cm"
        else:
            cond = ""
        tags = []
        for key, fmt in (("speed", "spd {:g}"), ("slow", "slow->{:g}"), ("side", "{} line"),
                         ("shift", "shift {:+g}cm"), ("steer", "max {:g}deg"), ("style", "{}")):
            if step.get(key) is not None and not (key == "speed" and step["do"] in ("stop", "wait")):
                tags.append(fmt.format(step[key]))
        return title, [c for c in (cond, "  ".join(tags[:2]), "  ".join(tags[2:])) if c]

    def view(self, dbg, now, fps=None, web=False):
        """Camera detection view with the info panel on the right.

        web=True: for the web page. The camera view only, with the panel's values as data (dbg.extra: web_state and the
        light and figure close-ups), which the page lays out itself. The banner goes there too instead of onto the picture."""
        course, detect, steering = self.course, self.detect, self.steering
        step = course.step()
        (self.traffic if course.needs_light() else self.traffic_view).draw(dbg, regions=course.needs_light())
        if self.person is not None:
            draw_person(dbg, self.person.detector, self.person.at)

        # at= height: this action ends when the junction or crosswalk line comes down to this dashed line.
        if step is not None and step["until"] in CourseRunner.AT_EVENTS and step["sec"] is None:
            at = step.get("at") or JUNCTION_Y       # With no at=, it ends at JUNCTION_Y.
            ay = int(at * FRAME_H)
            dashed_hline(dbg, ay, C_AT, thick=2)
            chip(dbg, f"END @{at:g}", FRAME_W - 90, ay - 22, C_AT, 0.38)

        if self.aim is not None and -FRAME_W < self.aim[0] < 2 * FRAME_W:     # STEER_MODE "pursuit": where the steering aims.
            ax, ay = int(self.aim[0]), int(min(self.aim[1], FRAME_H - 4))
            dbg.circle((ax, ay), 7, (20, 20, 20), -1, cv2.LINE_AA)
            dbg.circle((ax, ay), 5, C_AT, -1, cv2.LINE_AA)
        # Car position (bottom center) and an arrow for the current steering direction.
        cx0, by = int(FRAME_W * CAMERA_X), FRAME_H - 6
        a = np.radians(steering.angle * 1.6)
        tip = (int(cx0 + 70 * np.sin(a)), int(by - 70 * np.cos(a)))
        dbg.arrowedLine((cx0, by), tip, (20, 20, 20), 5, cv2.LINE_AA, tipLength=0.25)
        dbg.arrowedLine((cx0, by), tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)

        # One large notice at the top center, only the most important.
        lost = self.det is None
        banner = None
        if step is None or step["do"] == "finish" or self.done:
            banner = ("FINISH", (60, 160, 60)) if step is not None else None
        elif self.person is not None and self.person.hold:
            gone = self.person.gone_since
            banner = ("PERSON - CHECK" if self.person.checking else "PERSON - STOP" if gone is None else f"PERSON GONE - GO IN {max(self.person.resume_delay - (now - gone), 0):.1f}s",
                      (40, 40, 220))
        elif self.edge:
            banner = ("MAT EDGE - STOP", (40, 40, 220))
        elif step["do"] == "wait" and step["until"] == "green":
            backing_red = self.red is not None and self.red["back"] is not None
            banner = (("RED LIGHT - WAIT", (40, 40, 220)) if course.red_seen else
                      (("NO RED - BACKING UP", (200, 120, 0)) if backing_red else ("FINDING RED LIGHT", (120, 100, 80))))
        elif step["do"] == "wait":
            banner = ({"clear": "WAIT: REMOVE OBSTACLE", "obstacle": "WAIT: OBSTACLE", "enter": "PRESS ENTER ON PAGE"}.get(step["until"], "WAIT"),
                      (160, 120, 40))
        elif self.back is not None:
            banner = (f"BACKING UP - FINDING LANE {now - self.back['start']:.1f}s", (200, 120, 0))
        elif self.speed == 0 and not course.stopped(now) and step is not None:
            banner = ("STOPPED (LANE LOST)" if lost else "STOPPED", (40, 40, 220))
        elif lost and self.lost_since is not None:
            banner = (f"LANE LOST {now - self.lost_since:.1f}s", (0, 140, 255))
        if self.hold_banner:
            banner = self.hold_banner
        if web:
            dbg.extra = {"state": self.web_state(now, fps, banner), "minis": self.close_ups(dbg.base_image())}
            return dbg
        if banner:
            (tw, _), _ = cv2.getTextSize(banner[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x = (FRAME_W - tw) // 2 - 12
            dbg.shade((x, 8), (x + tw + 24, 44), banner[1], 0.8)
            dbg.putText(banner[0], (x + 12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # --- Right Panel ---
        W = self.PANEL_W
        p = Canvas((FRAME_H, W), fill=32)     # Plain background: sent as part of the base picture, the text and bars as commands.
        white, grey = (240, 240, 240), (150, 150, 150)

        def text(t, x, y, scale=0.45, color=white, thick=1):
            p.putText(t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        def bar(y, label, frac, value, color, mark=None):
            text(label, 12, y)
            text(value, W - 12 - cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0], y)
            p.rectangle((12, y + 6), (W - 12, y + 16), (70, 70, 70), -1)
            p.rectangle((12, y + 6), (12 + int((W - 24) * float(np.clip(frac, 0, 1))), y + 16), color, -1)
            if mark is not None:
                mx = 12 + int((W - 24) * float(np.clip(mark, 0, 1)))
                p.line((mx, y + 3), (mx, y + 19), C_AT, 2)

        # Current action / next action.
        idx, n = min(course.idx + 1, len(course.course)), len(course.course)
        text(f"STEP {idx}/{n}", 12, 24, 0.5, grey)
        el = now - course.step_start if course.step_start is not None else 0.0
        lim = None if course.needs_light() and course.red_seen else step and (step["sec"] or step["max"])
        t = f"{el:.1f}s" + (f" / {lim:g}s" if lim else "")
        text(t, W - 12 - cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0], 24, 0.45, grey)
        p.rectangle((12, 31), (W - 12, 34), (70, 70, 70), -1)
        p.rectangle((12, 31), (12 + int((W - 24) * (course.idx / max(n, 1))), 34), C_PATH, -1)
        title, lines = self.step_text(step)
        text(title, 12, 60, 0.62, white, 2)
        y = 80
        for line in lines:
            text(line, 12, y, 0.42, C_AT if line.startswith("until") else grey)
            y += 18
        nxt = course.course[course.idx + 1] if course.idx + 1 < n else None
        text("next: " + (self.step_text(nxt)[0] if nxt else "-"), 12, 134, 0.42, grey)
        p.line((12, 146), (W - 12, 146), (70, 70, 70), 1)

        # Speed / steering.
        slow = course.opt("slow")
        bar(168, "SPEED", self.speed / 0.6, f"{self.speed:.2f}", C_PATH if self.speed > 0 else (40, 40, 220),
            None if slow is None else slow / 0.6)
        text("STEER", 12, 206)
        st = steering.angle
        t = f"{'L' if st < -0.5 else 'R' if st > 0.5 else ''} {abs(st):.0f} deg"
        text(t, W - 12 - cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0], 206)
        mid = W // 2
        p.rectangle((12, 212), (W - 12, 222), (70, 70, 70), -1)
        sx = mid + int((W // 2 - 12) * float(np.clip(st / AUTO_STEER_MAX, -1, 1)))
        p.rectangle((min(mid, sx), 212), (max(mid, sx), 222), (255, 200, 80), -1)
        p.line((mid, 209), (mid, 225), white, 1)
        p.line((12, 236), (W - 12, 236), (70, 70, 70), 1)

        # Status marks, colored only when on.
        items = self.status_items()
        for k, (name, on, color, val) in enumerate(items):
            x, yy = 12 + (k % 2) * ((W - 24) // 2 + 2), 242 + (k // 2) * 24
            w = (W - 24) // 2 - 2
            p.rectangle((x, yy), (x + w, yy + 20), color if on else (55, 55, 55), -1)
            label = name if val is None else f"{name} {val}"
            dark = on and sum(color) > 380
            p.putText(label, (x + 6, yy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (20, 20, 20) if dark else (white if on else grey), 1, cv2.LINE_AA)

        # Bottom: white-line threshold and fps.
        track = "white line" if LINE_COLOR == "white" else "black line"
        foot = f"{track}  thr {detect.thr:.0f} {'auto' if WHITE_THR is None else 'set'}"
        if fps:
            foot += f"   {fps:.0f} fps"
        text(foot, 12, FRAME_H - 12, 0.4, grey)
        return Canvas.hstack([dbg, p])

    def status_items(self):
        """The status marks: (name, on, BGR color, value or None)."""
        course, detect = self.course, self.detect
        lost = self.det is None
        on_light = course.local_light or self.light     # Re-reading the spot that was red is more accurate.
        held = detect.tracker is not None and detect.tracker.p is not None
        return [("LANE", True, (60, 200, 60) if not lost else (40, 40, 220), "OK" if not lost else "LOST"),
                ("JUNCTION", detect.junction_y is not None or detect.far_y is not None, C_JUNC, None),
                ("CROSSWALK", detect.crosswalk_any, C_XWALK, None),
                ("LANE ENDS", detect.gap_y is not None, C_GAP, None),
                ("LIGHT", on_light is not None,
                 (40, 40, 255) if on_light == "red" else (60, 220, 60), (on_light or "").upper() or None),
                ("OBSTACLE", bool(self.obs and self.obs.near), (0, 140, 255), None),
                ("PERSON", bool(self.person and self.person.seen), C_PERSON, None),
                ("CHECK", JUNCTION_TRACK_ASSIST and held, C_AT, f"{detect.tracker.conf:.0%}" if held else None)]

    def web_state(self, now, fps, banner):
        """The panel's values for the web page, which draws the panel itself (plain numbers and text, a few hundred bytes)."""
        course, detect, steering = self.course, self.detect, self.steering
        step = course.step()
        n = len(course.course)
        title, lines = self.step_text(step)
        nxt = course.course[course.idx + 1] if course.idx + 1 < n else None
        el = now - course.step_start if course.step_start is not None else 0.0
        lim = None if course.needs_light() and course.red_seen else step and (step["sec"] or step["max"])
        css = lambda bgr: "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        on_light = course.local_light or self.light
        person = self.person
        if person is None:
            who = None
        elif person.hold:
            gone = person.gone_since
            who = ("CHECKING" if person.checking else "STOP" if gone is None
                   else f"GO IN {max(person.resume_delay - (now - gone), 0):.1f}s")
        else:
            who = "SEEN" if person.seen else "CLEAR"
        det = self.det
        return {
            "step": {"i": min(course.idx + 1, n), "n": n, "done": course.idx / max(n, 1), "title": title, "lines": lines,
                     "el": round(el, 1), "lim": lim, "next": self.step_text(nxt)[0] if nxt else None},
            "speed": round(self.speed, 2), "slow": course.opt("slow"),
            "steer": round(steering.angle, 1), "steer_max": AUTO_STEER_MAX,
            "lane": None if det is None else [round(det[0], 3), round(det[1], 3)],
            "flags": [[name, bool(on), css(color), val] for name, on, color, val in self.status_items()],
            "banner": None if banner is None else [banner[0], css(banner[1])],
            "light": {"state": on_light, "wait": course.needs_light(), "red_seen": bool(course.red_seen)},
            "person": None if person is None else {"state": who},
            "line": LINE_COLOR, "thr": round(float(detect.thr)), "thr_auto": WHITE_THR is None, "fps": None if fps is None else round(fps),
            "method": LANE_POINTS if LANE_MODEL != "track" else "track",
        }

    MINI = (160, 120)   # Largest close-up sent to the page (w, h).
    MINI_HALF = 36      # A close-up is at least twice this tall (frame px): a lamp a few px wide still shows its pole and housing.

    def close_ups(self, frame):
        """Small pictures of the traffic light and the LEGO figure, for the page's side views: {name: BGR image}.
        Only what was found this frame, each a few kilobytes."""
        out = {}
        course = self.course
        light = self.traffic if course.needs_light() else self.traffic_view
        boxes = [(x, y, w, h) for x, y, w, h, _ in light.boxes] or [(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1, _ in light.local]
        if boxes:
            x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
            out["light"] = self._crop(frame, x, y, w, h, 2.5)
        if self.person is not None and self.person.detector.found:
            f = self.person.detector.found
            x, y, w, h = figure_box(f["x"], f["y"], f["w"], f["h"], f["feet"])
            out["person"] = self._crop(frame, x, y, w, h, 1.6)
        return {k: v for k, v in out.items() if v is not None}

    def _crop(self, frame, x, y, w, h, grow):
        """The box grown grow times around its center, inside the frame, scaled to fit MINI."""
        cx, cy, half = x + w / 2, y + h / 2, max(max(w, h) * grow / 2, self.MINI_HALF)
        x0, y0 = int(max(cx - half * 4 / 3, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half * 4 / 3, frame.shape[1])), int(min(cy + half, frame.shape[0]))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = frame[y0:y1, x0:x1]
        k = min(self.MINI[0] / crop.shape[1], self.MINI[1] / crop.shape[0])
        size = (max(int(crop.shape[1] * k), 1), max(int(crop.shape[0] * k), 1))
        return cv2.resize(crop, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)

    def tick(self, frame, now, show=False):
        """Return (steering angle, or None to hold, motor command, debug image)."""
        course, detect, steering = self.course, self.detect, self.steering
        if course.turning(now):
            # During a turn the car rotates quickly, so remembered line positions from the previous frame are wrong. Judge each frame fresh.
            detect.left, detect.right, detect.center = [None] * SLICES, [None] * SLICES, [None] * SLICES
            if detect.tracker is not None:      # Likewise the tracked lane: it is the old one. It starts again from the new lane.
                detect.tracker.reset()
        detect.now = now
        detect.assist_turn = course.turning(now)   # The junction check does not trust the tracked lane mid turn: it is the old one.
        detect.moving = self.speed > 0 and abs(steering.angle) < GLARE_STEER_MAX     # Driving forward: the motion test for reflections can run.
        det, junction, dbg = detect(frame, course.lane_side(now), draw=show)     # Skip drawing when the view is not being sent.
        step = course.step()
        detect.far_y = None
        if step is not None and step.get("slow") is not None and step["until"] in ("junction", "end"):
            detect.far_y = detect.far_hline(frame)      # Compute only when needed (saves fps).
            if detect.far_y is not None and show:
                dashed_hline(dbg, detect.far_y, C_JUNC, dash=6, gap=6)
                chip(dbg, "JUNCTION AHEAD", FRAME_W - 130, detect.far_y - 20, C_JUNC, 0.38)
        self.find_stop(step, dbg, now, frame)
        course.stop_whole = bool(self.stop_found and self.stop_found.get("whole"))
        # Traffic light: every frame while waiting for green. For the view only, every LIGHT_VIEW_EVERY frames (the last result is shown between).
        need_light = course.needs_light()
        signs = None if detect.signs is None else (int(FRAME_H * ROI_TOP), detect.signs)
        self.traffic.floor_signs = self.traffic_view.floor_signs = signs
        if need_light:
            self.traffic(frame)
        elif show and self.light_tick % LIGHT_VIEW_EVERY == 0:
            self.traffic_view(frame)        # For the view only.
        self.light_tick += 1
        light = self.traffic.state if need_light else (self.traffic_view.state if show else None)
        course.local_light = None
        if step is not None and step["do"] == "wait" and step["until"] == "green" and course.red_seen:
            # Whether any spot that was red has changed to green. The lamp spot still changes even if another red object is present.
            colors = [self.traffic.color_at(frame, box) for box in course.red_boxes]
            course.local_light = "green" if "green" in colors else ("red" if "red" in colors else None)
        # LEGO figure: only while following the lane (follow, to_junction, go, ...) or in straight. Turns, stop, and wait do not look for it.
        looks = course.following() or (step is not None and step["do"] == "straight")
        hold = self.person is not None and looks and self.person.update(frame, now)
        course.paused = hold and not course.stopped(now) and (step is None or step["do"] != "finish")
        if course.paused:
            self.stand_for_person(now)
            self.det, self.junction, self.light = det, junction, light
            return None, 0.0, dbg
        if step is not None and step["do"] in ("left", "right") and step.get("deg"):
            if self.yaw_idx != course.idx:
                self.yaw.reset()
                self.yaw_idx = course.idx
            course.turned = self.yaw(frame) * (-1 if step["do"] == "right" else 1)
        else:
            course.turned, self.yaw_idx = None, -1
        course.moving = self.speed > 0
        idx_before, ruler_before = course.idx, course.ruler      # Starting the next action replaces the ruler.
        course.update(junction, now, det, detect.crosswalk,
                      {"junction": detect.junction_y, "crosswalk": detect.crosswalk_y, "gap": detect.gap_y, "_crosswalk": detect.xwalk2_y,
                       "_stop": self.stop_y, "_stop_target": self.stop_target,
                       "near": detect.near_line,
                       **({"hline_y": detect.hline_edge(), "tip_y": detect.tip_y,
                           "end_y": detect.junction_point() if CourseRunner.uses_laneend(step) else None} if step is not None and step.get("cm") is not None else {})},
                      light if need_light else None, self.traffic.lamps, self.obs.near if self.obs else None)
        self.det, self.junction, self.light = det, junction, light
        if dbg is not None and detect.end_y is not None:
            ey = int(detect.end_y)
            dbg.line((0, ey), (FRAME_W - 1, ey), C_JUNC, 1, cv2.LINE_AA)
            chip(dbg, f"LANE END {floor_cm(detect.end_y) + LANEEND_OFFSET_CM:.1f}cm", 10 if detect.end_side == "left" else FRAME_W - 190, ey - 20, C_JUNC, 0.4)
        watching = course.idx != idx_before and step is not None and step["until"] in ("junction", "end")
        if watching or self.ends.open is not None:
            line_y = detect.hline_edge()
            if watching:
                following = course.step()
                self.ends.ended(idx_before, step, ruler_before, line_y, self.last_cmd, steering.angle, det, now,
                                following is not None and following["do"] in ("stop", "wait"))
            else:
                self.ends.watch(line_y, now)
        if course.done:
            self.done = True
            self.speed = 0.0
            return None, 0.0, dbg

        base = course.speed if course.speed is not None else self.base_speed
        set_base = base     # The speed the action set, before slow= takes anything off.
        base = self.approach_speed(base, now)
        kp, kh, smooth = STYLES[course.opt("style", "normal")]
        if det is not None:
            # shift=+3: follow a line 3 cm to the right of the lane center.
            # offset is how far the lane center is to the right of the camera, so the car must see the lane center on its left (negative) to sit on the right.
            # Add shift so that state reads as zero.
            det = (det[0] + course.opt("shift", 0.0) * LANE_PX_PER_CM / (FRAME_W / 2), det[1])
        # Crosswalk just in front of the car: the stripes shake lane detection, so hold the current steering briefly and drive through.
        if detect.crosswalk and not detect.stripes:
            self.cw_since = now if self.cw_since is None else self.cw_since    # 0.0 is a real start time, not "unset".
        else:
            self.cw_since = None
        if self.cw_since is not None and now - self.cw_since < CROSSWALK_HOLD_MAX:
            lane = steering.angle
        elif det is not None:
            # Clip heading error for steering only. The raw det[1] was already used for the align and turn-end tests.
            raw_h = det[1]
            h = float(np.clip(raw_h, -HEADING_STEER_MAX, HEADING_STEER_MAX))
            # While aligning, a large heading error such as a corner or hairpin is not steered into further. Only position is corrected a little.
            # A horizontal line in view makes the heading jump to +-1 (run 20261103-203911 steered to -26), so hold heading then.
            if detect.junction_y is not None and abs(raw_h) >= ALIGN_HLINE_HEADING:
                self.hline_t = now
            real_hline = detect.junction_y is not None and self.hline_t is not None and now - self.hline_t <= ALIGN_HLINE_HOLD
            hold_heading = ((course.step() or {}).get("until") == "aligned" and abs(raw_h) > TURN_END_HEADING
                            and (ALIGN_HOLD_HEADING or real_hline))
            if hold_heading:
                h = 0.0
            lane = steering.lane(det[0], h, now, kp, kh)
            aim = self.pursuit_angle(course, base)
            if aim is not None and not hold_heading:
                lane = aim
                if (course.step() or {}).get("until") == "aligned" and (real_hline or detect.junction_y is not None):
                    # The same care as the heading hold above: aligning with a line across in view, do not swing into it.
                    # (The tracked heading does not jump at a line across, so the line itself is the sign here.)
                    lane = float(np.clip(lane, -ALIGN_STEER_MAX, ALIGN_STEER_MAX))
            if hold_heading:
                lane = float(np.clip(lane, -ALIGN_STEER_MAX, ALIGN_STEER_MAX))
            # When a junction line is visible ahead, do not snap the steering toward a large position error.
            now_step = course.step()
            if (now_step and now_step["do"] == "follow" and now_step.get("until") in ("junction", "end") and course.final_approach()
                    and (detect.junction_y is not None or detect.far_y is not None)):
                lane = float(np.clip(lane, -APPROACH_STEER_MAX, APPROACH_STEER_MAX))
        else:
            lane = None
        target = course.target(lane, now, steering.angle)
        # Mat-edge stop is skipped while the lane is visible or during a turn, so a false hit does not stop the car.
        # go_end(): the mat ends just past a dead end. While its measured end line is still ahead, that edge is expected.
        dead_end = bool((course.step() or {}).get("dead_end")) and course.run_out and course.ruler.z is not None
        self.edge = detect.edge and det is None and not course.turning(now) and not dead_end
        red_angle = self.search_red(now, hold)
        # Lane lost during follow: drive on for up to lost_hold() seconds (LOST_HOLD, longer when slower). Standing still does not count: a lane
        # unseen through stop(6.5) on a crosswalk read as long lost the moment the car set off, and it backed up.
        if target is None and not course.stopped(now):
            self.lost_since = self.lost_since if self.lost_since is not None else now
        else:
            self.lost_since = None
        if target is None and self.lost_since is not None and course.following() and LOST_CENTER_RATE > 0:
            # Lane lost while following: ease the wheels back to straight rather than keep the last turn.
            dt = min(now - steering.out_t, 0.1) if steering.out_t is not None and now > steering.out_t else 1 / 30
            back = LOST_CENTER_RATE * dt
            target, smooth = steering.angle - float(np.clip(steering.angle, -back, back)), 0.0
        # One output() per frame. Calling it again in the same instant used to spend a phantom 1/30s of
        # steering rate, so the red-light reverse turned the wheels at fps/30 times STEER_RATE.
        if red_angle is not None:
            target, smooth = red_angle, 0.0
        elif target is not None and course.stopped(now):
            target = steering.angle     # Standing still: keep the wheels where they are, so the car does not set off at a full-lock angle.
        steer = None
        if target is not None:
            limit = course.opt("steer", AUTO_STEER_MAX)
            steer = steering.output(float(np.clip(target, -limit, limit)), now, smooth)
        lost_for = now - self.lost_since if self.lost_since is not None else 0.0
        # cm=: over the end line the lane lines may run out. That is expected, so the car holds its steering and does not back up.
        lost_long = self.lost_since is not None and lost_for > lost_hold(base) and not course.run_out     # The set speed, not the one now (0 once stopped).
        steer = self.back_up(now, lost_long, steer)
        backing = self.back is not None
        if red_angle is not None:
            speed = -LOST_BACK_SPEED
        elif course.stopped(now) or hold:    # hold here: a stop action just ended with the figure still there.
            speed = 0.0
        elif (course.step() or {}).get("do") == "back":
            speed = -(course.speed if course.speed is not None else LOST_BACK_SPEED)     # back(): reverse for its seconds.
        elif self.edge:
            speed = 0.0
        elif backing:
            speed = -LOST_BACK_SPEED
        elif lost_long:
            speed = 0.0
        elif self.lost_since is not None:
            speed = min(base, MIN_SPEED) if getattr(_course, "SLOWDOWN_ON", True) else base   # Creep while the lane is lost (SLOWDOWN_ON = False: keep the speed).
        elif course.following() and STEER_SLOWDOWN and not course.opt("keep_speed"):
            speed = speed_for(steering.angle, base)   # Slow for a large steer only while following the lane (off: STEER_SLOWDOWN).
        else:
            speed = base        # Turn, straight, and curve keep the speed that was set.
        floor = course.opt("min_speed")
        if floor is not None and speed > 0:     # min_speed=: slowing for steering, slow=, or a lost lane stops here (never above the set speed).
            speed = max(speed, min(floor, set_base))
        if speed > 0 and self.lost_since is None and abs(steering.angle) > HARD_STEER:
            # A large steer drags the wheels and slows the car, so add this back. Skipped while the lane is lost.
            # Never past full power: the motor cannot give more, and the retrace would count on speed that was not there.
            speed = min(speed + HARD_STEER_BOOST, max(speed, 1.0))
        self.speed = speed
        if speed > 0 and self.last_t is not None:     # Remember forward driving for a later retrace.
            # The speed the motor really gets, so a retrace covers the distance really driven.
            self.history.append((now - self.last_t, steering.angle, max(speed, motor_floor(steering.angle))))
            while sum(h[0] for h in self.history) > LOST_BACK_MEMORY:
                self.history.pop(0)
        self.last_t = now
        cmd = speed
        if cmd > 0:
            # On the floor a motor command that is too low leaves the car stopped, especially at a large steer or on a weak battery, so raise it to the minimum.
            cmd = max(cmd, motor_floor(steering.angle))
        self.last_cmd = cmd
        return steer, cmd, dbg


def open_camera(locked=True):
    """locked: use the fixed exposure and white balance saved by calibrate.py (CAMERA_LOCK). False leaves the camera automatic."""
    from libcamera import Transform, controls
    from picamera2 import Picamera2

    cam = Picamera2()
    start = {"FrameRate": CAMERA_FPS}
    lock = locked and CAMERA_LOCK
    if lock:
        # Lock the values measured under venue lighting so brightness and color do not swing with mat reflections or a person's shadow.
        # Given at configure time, so they hold from the very first frame after every power-on (not auto for the first frames).
        start.update({"AeEnable": False, "AwbEnable": False,
                      "ExposureTime": int(CAMERA_LOCK["ExposureTime"]),
                      "AnalogueGain": float(CAMERA_LOCK["AnalogueGain"]),
                      "ColourGains": tuple(CAMERA_LOCK["ColourGains"])})
    if LENS_POSITION is not None:
        start.update({"AfMode": controls.AfModeEnum.Manual, "LensPosition": LENS_POSITION})
    cam.configure(cam.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},   # OpenCV BGR order.
        sensor={"output_size": CAMERA_SENSOR},
        transform=Transform(hflip=HFLIP, vflip=VFLIP),
        controls=start))
    cam.start()
    if LENS_POSITION is None:
        cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    if lock:
        # Wait until the sensor reports the locked exposure and gain (at most about 1 s), so the first frames used match every run.
        want_exp, want_gain = int(CAMERA_LOCK["ExposureTime"]), float(CAMERA_LOCK["AnalogueGain"])
        for _ in range(30):
            md = cam.capture_metadata()
            if abs(md.get("ExposureTime", 0) - want_exp) <= want_exp * 0.05 and abs(md.get("AnalogueGain", 0) - want_gain) <= want_gain * 0.05:
                lux, was = md.get("Lux"), CAMERA_LOCK.get("Lux")
                if lux and was and not 1 / LUX_WARN_RATIO <= lux / was <= LUX_WARN_RATIO:
                    print(f"[warning] lighting is {'brighter' if lux > was else 'darker'} than at calibration (lux {lux:.0f}, was {was:.0f}). "
                          f"Lines and colors may read differently: run Calibrate again for this lighting")
                break
        else:
            print(f"[warning] camera lock not reached: exposure {md.get('ExposureTime')} us gain {md.get('AnalogueGain', 0):.2f} (wanted {want_exp} us, {want_gain:.2f})")
    return cam


def load_person(path=PERSON_FILE):
    """What finds the LEGO figure: the one registered on the day (aicar person), and any other if PERSON_ANY. None turns the check off, with a warning if that was not asked for."""
    if not PERSON_ON:
        print("LEGO figure detection off (course.py PERSON_ON)")
        return None
    detector = PersonDetector.load(HORIZON_Y, path)
    every = AnyFigure.load(HORIZON_Y, detector) if PERSON_ANY else None
    if every:
        print(f"LEGO figure detection on (unregistered figures too, {len(every.views)} registered views)")
        return every
    if detector and detector.photo_sigs is None:
        print("[warning] No registration photos, so no photo comparison. Register again to turn it on: aicar person")
    if detector and detector.embeds is None:
        print("[warning] The registered figure has no look data, so only color and outline are used. Register again to turn it on: aicar person")
    print(f"LEGO figure detection on ({len(detector.views)} views)" if detector
          else "[warning] No LEGO figure registered, so the car will not stop for one. Register: aicar person")
    return detector


class MotorWatchdog:
    """Stops the motor if the drive loop sends no command for WATCHDOG_SEC (a camera that hangs in capture_array, for example).

    Only acts when the loop has stalled, so a normal drive is unchanged.
    """

    def __init__(self, car, timeout):
        self.car, self.timeout = car, timeout
        self.lock = threading.Lock()
        self.last = time.monotonic()
        self.tripped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def drive(self, cmd):
        with self.lock:
            self.car.drive(cmd)
            self.last = time.monotonic()
            if self.tripped:
                print("\n[watchdog] loop running again")
                self.tripped = False

    def _run(self):
        while not self._stop.wait(0.1):
            with self.lock:
                if not self.tripped and time.monotonic() - self.last > self.timeout:
                    try:
                        self.car.drive(0)
                    except Exception:
                        pass
                    self.tripped = True
                    print(f"\n[watchdog] no command for {self.timeout:g}s (camera stalled?) -> motor stopped")

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)


WATCHDOG_SEC = 0.5      # Motor stops if the drive loop sends nothing for this long.
FRAME_ERRORS_MAX = 10   # Consecutive frames that raise an error before the drive ends.


def stop_on_signal(*_):
    raise KeyboardInterrupt


def main():
    from car import Car

    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", action="store_true", help="drive the motor (otherwise steering only)")
    ap.add_argument("--speed", type=float, default=BASE_SPEED)
    ap.add_argument("--start", type=int, default=1, help="COURSE step to start from")
    ap.add_argument("--record", action="store_true", help="save frames and the log in runs/")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="do not keep a drive log in logs/ (aicar drive-log off)")
    ap.add_argument("--set", nargs="*", default=[], help="override settings for this run with KEY=VALUE (for example KP=1.8)")
    ap.add_argument("--button", choices=("off", "on", "auto"), default="off",
                    help="start button on GPIO21: on waits for a press, auto follows button.json (aicar button)")
    args = ap.parse_args()

    for kv in args.set:
        key, val = kv.split("=", 1)
        if key not in globals() or not key.isupper():
            raise SystemExit(f"--set: unknown setting {key}")
        try:
            new = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            new = val       # A string such as "line".
        print(f"{key}: {globals()[key]} -> {new}")
        globals()[key] = new

    try:
        box = None if args.no_log else DriveLog({"drive": args.drive, "speed": args.speed, "start": args.start, "record": args.record, "set": args.set})
    except OSError as e:       # A full or read-only card: drive anyway.
        box = None
        print(f"[warning] drive log not kept: {e}")

    # Started in the background with &, SIGINT is ignored, so register it again. Ctrl+C and kill -INT can then stop the process.
    signal.signal(signal.SIGINT, stop_on_signal)
    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)   # Keep driving if ssh drops. Stop with Ctrl+C or aicar stop.

    # Checks that need no hardware come first: a refused course then never touches the motor or the camera.
    problem = course_problem(COURSE, args.start, no_stream=args.no_stream)
    if problem:
        if box:
            print(problem)      # Into the drive log too, so a refused start can be read back later.
            box.close()
        raise SystemExit(problem)

    car = cam = obs = None
    button = ctl = bset = None

    def refuse(message):
        """Stop before driving: release everything opened so far, and keep the reason in the drive log."""
        for name, fn in (("car", lambda: car and car.close()), ("camera", lambda: cam and cam.stop()),
                         ("obstacle", lambda: obs and obs.close()), ("button", lambda: button and button.close())):
            try:
                fn()
            except Exception as e:
                print(f"[error] {name}: {e}")
        if box:
            print(message)
            box.close()
        raise SystemExit(message)

    try:
        car = Car()
        try:
            from obstacle import Obstacle
            obs = Obstacle()
        except Exception as e:      # Driving still works if the sensor or module is missing.
            print(f"[warning] obstacle sensor not used: {e}")
            obs = None
            # go_dis() drives until the sensor sees the barrier: with no sensor it would drive straight into it. A wait for
            # "clear" (start_gate) would pass at once, with the barrier still down.
            blind_steps = [i + 1 for i, st in enumerate(COURSE) if i + 1 >= args.start and
                           (st.get("until") == "clear" or (st.get("until") == "obstacle" and st["do"] != "wait"))]
            if blind_steps and args.drive:
                st = COURSE[blind_steps[0] - 1]
                refuse(f"step {blind_steps[0]} ({st['do']} until {st['until']}) needs the distance sensor, "
                       f"which is not working ({e}). Not driving.")
        if AUTO_TRIM:
            car.load_bias()     # Continue with the center trim learned on a previous run.
        if args.button != "off":
            from start_button import ButtonReader, StartControl, load_settings
            bset = load_settings()
            if args.button == "on" or bset["enabled"]:
                try:
                    button = ButtonReader(bset["hold_sec"])
                except Exception as e:      # Driving without the button would start at once: not what was asked for.
                    refuse(f"the start button (GPIO21) is not working: {type(e).__name__}: {e}. Not driving. "
                           f"Check the wiring, or turn the button off: aicar button off")
                ctl = StartControl(bset["hold"])
        cam = open_camera()
        person_det = load_person()
        driver = Driver(COURSE, args.start, args.speed, obs, person_det)
        pauses = any(s["until"] == "enter" for s in driver.course.course)
        # Only a course with next() sends keys to the program, so other drives keep the page's own keys.
        stream = None if args.no_stream else Stream(args.port, "AICAR drive", "Enter next step (next)" if pauses else None)
        rec = Recorder(args.speed, args.set) if args.record else None
    except SystemExit as e:
        refuse(str(e))
    except (ValueError, OSError, RuntimeError) as e:    # A bad course value, the camera or the motor not there: say why, release the rest.
        refuse(f"cannot start: {type(e).__name__}: {e}")
    print(calibration_note())
    print("Traffic light:", f"registered ({', '.join(sorted(LIGHT_MODEL.colors()))})" if LIGHT_MODEL else "not registered (color thresholds)",
          "| crosswalk:", f"registered ({len(XWALK_MODEL.views)} views)" if XWALK_MODEL else "not registered",
          "| lines:", f"registered (reflection cut {LINE_MODEL.reflect_cut():.2f})" if LINE_MODEL else "not registered",
          "| stop points:", ", ".join(STOP_MODEL.names()) if STOP_MODEL else "none")
    print("Lane:", LANE_MODEL, "| junction check:", "on" if JUNCTION_TRACK_ASSIST else "off",
          "(false junctions and false lane ends only; steering stays the band method)" if JUNCTION_TRACK_ASSIST and LANE_MODEL != "track" else "")
    print("Motor", "ON" if args.drive else "OFF (steering only)",
          f"| course {len(COURSE)} steps, from {args.start} | Ctrl+C to stop")

    frames, t0, t_start = 0, time.monotonic(), time.monotonic()
    fps = None
    prev_now = t_start
    watchdog = None
    errors = view_errors = 0
    # Start button: the course clock (dnow) stands still while waiting or paused, so a timed step does not run out meanwhile.
    paused_total, paused_at = 0.0, (t_start if ctl else None)
    driver_used = False
    if ctl:
        from start_button import NAMES, RUN, WAIT, describe
        hold_banners = {WAIT: ("PRESS BUTTON TO START", (160, 120, 40)), "PAUSE": ("PAUSED - PRESS BUTTON", (160, 120, 40))}
        driver.hold_banner = hold_banners[WAIT]
        print(describe(bset))
        print("Waiting for the start button (GPIO21)")
    try:
        watchdog = MotorWatchdog(car, WATCHDOG_SEC)
        while True:
            frame = cam.capture_array()
            now = time.monotonic()
            # Send this frame to the browser only if someone is watching. The view is only drawing commands (the browser paints
            # them), so every frame can go. If nobody is watching, it costs nothing.
            show = stream is not None and stream.watching
            if ctl:
                for ev in button.take():
                    act = ctl.handle(ev)
                    if act is None:
                        continue
                    if act == "start" and driver_used:      # A fresh run of the course from the first step.
                        driver = Driver(COURSE, args.start, args.speed, obs, person_det)
                    if act in ("start", "resume"):
                        driver_used = True
                        if paused_at is not None:
                            paused_total += now - paused_at
                        paused_at = None
                        driver.hold_banner = None
                    else:
                        watchdog.drive(0)
                        if paused_at is None:
                            paused_at = now
                        driver.hold_banner = hold_banners[ctl.state]
                    print(f"\n[button] {ev} -> {act}: {NAMES[ctl.state]}", flush=True)
                if ctl.state != RUN:
                    watchdog.drive(0)
                    if ctl.state == WAIT and ctl.fresh:
                        car.steer(0)
                    dnow = paused_at - paused_total
                    if show:
                        try:
                            _, _, dbg = driver.detect(frame, driver.course.lane_side(dnow), draw=True)
                            stream.update(driver.view(dbg, dnow, fps, web=True))
                        except Exception as e:
                            view_errors += 1
                            if view_errors == 1 or view_errors % 100 == 0:
                                print(f"\n[view error {view_errors}] {type(e).__name__}: {e} (driving goes on)")
                    if stream:
                        stream.keys.take()      # Keys pressed while standing still do not count later.
                    frames += 1
                    if time.monotonic() - t0 >= 1:
                        fps = frames / (time.monotonic() - t0)
                        print(f"\r{fps:4.1f} fps  {NAMES[ctl.state]}    ", end="", flush=True)
                        frames, t0 = 0, time.monotonic()
                    continue
            dnow = now - paused_total
            try:
                steer, cmd, dbg = driver.tick(frame, dnow, show)
                errors = 0
            except Exception as e:      # One bad frame does not end the drive: stand still this frame and carry on.
                errors += 1
                watchdog.drive(0)
                print(f"\n[frame error {errors}/{FRAME_ERRORS_MAX}] {type(e).__name__}: {e}")
                if errors >= FRAME_ERRORS_MAX:
                    raise
                continue
            if driver.done:
                watchdog.drive(0)     # The last step ended: stop now, not only when the pins are released below.
                print("\nCourse done")
                if ctl:     # With the start button the program stays up: the next press runs the course again.
                    ctl.finished()
                    paused_at = now
                    driver.hold_banner = hold_banners[WAIT]
                    print("Waiting for the start button (the next press runs the course from the first step)")
                    continue
                break
            if steer is not None:
                car.steer(steer)
            if driver.edge:
                print("\r[mat edge] stopped    ", end="", flush=True)
            watchdog.drive(cmd if args.drive else 0)
            course, detect, steering = driver.course, driver.detect, driver.steering
            det, junction, light, speed = driver.det, driver.junction, driver.light, driver.speed

            # Auto center trim: if straight driving keeps steering to one side, the servo center is off.
            both = sum(l is not None and r is not None for l, r in zip(detect.left, detect.right))
            if (AUTO_TRIM and args.drive and speed > 0 and course.following() and det is not None
                    and both >= SLICES - 2 and abs(det[1]) < 0.05 and abs(steering.angle) < 8):
                car.bias += steering.angle * (now - prev_now) / AUTO_TRIM_TAU
            prev_now = now

            if rec:
                rec.write(frame, now - t_start, det, junction, steering.angle, speed, course, dnow, cmd)
            if box:
                box.write(now - t_start, driver, cmd, dnow)
            frames += 1
            if show:
                try:
                    stream.update(driver.view(dbg, dnow, fps, web=True))
                except Exception as e:      # The view is for the page only: a fault in it must never stop the drive.
                    view_errors += 1
                    if view_errors == 1 or view_errors % 100 == 0:
                        print(f"\n[view error {view_errors}] {type(e).__name__}: {e} (driving goes on)")
            if stream:
                if "\n" in stream.keys.take():      # Enter is read only for next(). Ctrl+C on the page stops the car, like in the terminal.
                    course.press_enter()
            if time.monotonic() - t0 >= 1:
                fps = frames / (time.monotonic() - t0)
                print(f"\r{fps:4.1f} fps  "
                      f"steer {steering.angle:+5.1f}  speed {speed:.2f}  "
                      f"step {course.label()} {course.status(dnow):9s} "
                      + (f"center {car.bias:+.1f} " if AUTO_TRIM else "")
                      + (f"light {light} " if light else "")
                      + ("GAP " if detect.gap_y is not None else "")
                      + f"{'LOST' if det is None else ''}    ", end="", flush=True)
                frames, t0 = 0, time.monotonic()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        def release():
            # Motor first: nothing below may keep the car moving if it fails.
            if watchdog:
                watchdog.close()
            try:
                car.close()
            except Exception as e:
                print(f"\n[error] car.close: {e}")
                try:
                    car.drive(0)
                except Exception:
                    pass
            for name, fn in (("save_bias", lambda: (car.save_bias(), print(f"\nAuto trim saved: {car.bias:+.1f} deg")) if AUTO_TRIM and args.drive else None),
                             ("camera", cam.stop),
                             ("obstacle", lambda: obs.close() if obs else None),
                             ("button", lambda: button.close() if button else None),
                             ("recording", lambda: rec and (rec.close(), driver.ends.save(rec.dir / "ends.csv"), driver.save_stop_log(rec.dir / "stops.csv"),
                                                             print(f"Recording saved: {rec.dir} ({rec.n} frames)"))),
                             ("drive log", lambda: box and (print(f"Drive log: {box.dir} ({box.n} frames)"), box.close(driver)))):
                try:
                    fn()
                except Exception as e:
                    print(f"\n[error] {name}: {e}")

        try:
            release()
        except KeyboardInterrupt:
            # A second Ctrl-C used to leave this block, so the motor could stay on.
            try:
                if car is not None:
                    car.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
