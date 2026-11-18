"""Debug-view drawing as a list of commands, so the Pi only records them and the browser paints them.

Canvas has the OpenCV drawing calls used by the views (line, rectangle, circle, putText, arrowedLine) with the same
arguments, minus the image. It keeps the commands instead of painting. Then:
  - render() paints them with OpenCV in the same order onto the base picture: the same pixels the old in-place drawing
    gave (replay video, saved calibration view, the legacy /video stream).
  - payload() is what the web page gets: the real pictures as JPEG (a plain panel is only its color), the commands
    as JSON, line masks as packed bits (zlib), and with extra set, values the page lays out itself (state) and small
    close-up pictures (minis, JPEG).
    The page paints the commands itself, so the Pi spends no time drawing or encoding the drawn picture.

ImageDraw has the same methods but paints straight onto an ndarray, so the helpers (chip, dashed_hline, ...) take either.
"""
import json
import struct
import zlib

import cv2
import numpy as np

MAGIC = b"AIC1"


def _css(bgr):
    b, g, r = (int(v) for v in bgr[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _paint(img, op):
    """One recorded command onto img, exactly as the original OpenCV call did it."""
    kind = op[0]
    if kind == "line":
        _, (x0, y0, x1, y1), color, thick, lt = op
        cv2.line(img, (x0, y0), (x1, y1), color, thick, lt)
    elif kind == "rect":
        _, (x0, y0, x1, y1), color, thick, lt = op
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thick, lt)
    elif kind == "circle":
        _, (x, y), r, color, thick, lt = op
        cv2.circle(img, (x, y), r, color, thick, lt)
    elif kind == "text":
        _, (x, y), text, face, scale, color, thick, lt, _tw, _th = op
        cv2.putText(img, text, (x, y), face, scale, color, thick, lt)
    elif kind == "arrow":
        _, (x0, y0, x1, y1), color, thick, lt, tip = op
        cv2.arrowedLine(img, (x0, y0), (x1, y1), color, thick, lt, tipLength=tip)
    elif kind == "poly":
        _, pts, color, thick, lt = op
        cv2.polylines(img, [pts], False, color, thick, lt)
    elif kind == "dash":
        _, (x0, y, x1), color, dash, gap, thick = op
        for x in range(x0, x1, dash + gap):
            cv2.line(img, (x, y), (min(x + dash, x1), y), color, thick, cv2.LINE_AA)
    elif kind == "tint":
        _, top, on, lut, _color, _alpha = op
        roi = img[top:top + on.shape[0]]
        cv2.copyTo(cv2.LUT(roi, lut), on.view(np.uint8), roi)
    elif kind == "fpoly":
        _, pts, color, alpha = op
        x0, y0 = np.maximum(pts.min(axis=0), 0)
        x1, y1 = np.minimum(pts.max(axis=0) + 1, (img.shape[1], img.shape[0]))
        if x1 > x0 and y1 > y0:       # Only the polygon's box inside the picture is blended (a negative end would wrap around).
            roi = img[y0:y1, x0:x1]
            over = roi.copy()
            cv2.fillPoly(over, [pts - (x0, y0)], color, cv2.LINE_AA)
            cv2.addWeighted(over, alpha, roi, round(1 - alpha, 9), 0, roi)
    elif kind == "shade":
        _, (x0, y0, x1, y1), color, alpha = op
        over = img.copy()
        cv2.rectangle(over, (x0, y0), (x1, y1), color, -1)
        cv2.addWeighted(over, alpha, img, round(1 - alpha, 9), 0, img)    # round: 1 - 0.8 is 0.19999999999999996, not the 0.2 the view always used.
    else:
        raise ValueError(f"unknown draw command {kind}")


def _ints(*v):
    return tuple(int(x) for x in v)


class _Draw:
    """Drawing calls shared by Canvas (records) and ImageDraw (paints at once). Arguments as in OpenCV, without the image."""

    def _add(self, op):
        raise NotImplementedError

    def line(self, p0, p1, color, thickness=1, lineType=cv2.LINE_8):
        self._add(("line", _ints(*p0, *p1), tuple(color), int(thickness), lineType))

    def rectangle(self, p0, p1, color, thickness=1, lineType=cv2.LINE_8):
        self._add(("rect", _ints(*p0, *p1), tuple(color), int(thickness), lineType))

    def circle(self, center, radius, color, thickness=1, lineType=cv2.LINE_8):
        self._add(("circle", _ints(*center), int(radius), tuple(color), int(thickness), lineType))

    def putText(self, text, org, fontFace, fontScale, color, thickness=1, lineType=cv2.LINE_8):
        (tw, th), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)    # So the page fits its own font to the same box.
        self._add(("text", _ints(*org), str(text), fontFace, float(fontScale), tuple(color), int(thickness), lineType, tw, th))

    def arrowedLine(self, p0, p1, color, thickness=1, lineType=cv2.LINE_8, tipLength=0.1):
        self._add(("arrow", _ints(*p0, *p1), tuple(color), int(thickness), lineType, float(tipLength)))

    def polyline(self, pts, color, thickness=1, lineType=cv2.LINE_8):
        """An open line through these points (an N x 2 array of screen pixels)."""
        pts = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 2:
            self._add(("poly", pts, tuple(color), int(thickness), lineType))

    def fill_poly(self, pts, color, alpha):
        """A filled polygon (N x 2 screen pixels) blended over the picture: alpha * color + (1 - alpha) * pixel."""
        pts = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 3:
            self._add(("fpoly", pts, tuple(color), float(alpha)))

    def dashed(self, x0, y, x1, color, dash, gap, thick):
        """Horizontal dashed line (one command instead of one per dash)."""
        self._add(("dash", _ints(x0, y, x1), tuple(color), int(dash), int(gap), int(thick)))

    def tint(self, top, on, lut, color, alpha):
        """Paint the True pixels of the bool mask on (rows from top) through lut: alpha * color + (1 - alpha) * pixel."""
        self._add(("tint", int(top), np.ascontiguousarray(on, dtype=bool), lut, tuple(color), float(alpha)))

    def shade(self, p0, p1, color, alpha):
        """Filled box blended over the picture: alpha * color + (1 - alpha) * pixel."""
        self._add(("shade", _ints(*p0, *p1), tuple(color), float(alpha)))


class ImageDraw(_Draw):
    """The drawing calls, painted straight onto an ndarray."""

    def __init__(self, img):
        self.img = img

    @property
    def shape(self):
        return self.img.shape

    def _add(self, op):
        _paint(self.img, op)


def as_draw(img):
    """A Canvas stays as it is. An ndarray gets ImageDraw, which paints on it at once."""
    return img if isinstance(img, _Draw) else ImageDraw(img)


class Canvas(_Draw):
    """A view: a base picture that is never changed, and the commands drawn over it.

    Pieces placed side by side (hstack) keep their own commands, clipped to their own area, as when each was drawn
    on its own picture. Drawing on the joined canvas covers the whole of it.
    """

    def __init__(self, base, fill=None):
        """base: the picture (ndarray), or a (height, width) size. fill: color of a base that is only a size."""
        if isinstance(base, np.ndarray):
            self.base, self.h, self.w = base, base.shape[0], base.shape[1]
        else:
            self.base, (self.h, self.w) = None, base
        self.fill = fill
        self.parts = [[0, self.w, []]]      # [x offset, width, commands]
        self.extra = None       # For the page only: {"state": JSON-able dict, "minis": {name: small BGR picture}}. Not painted.

    @property
    def shape(self):
        return (self.h, self.w, 3)

    def _add(self, op):
        part = self.parts[-1]
        if part[0] != 0 or part[1] != self.w:      # Joined canvas: a command over all of it.
            part = [0, self.w, []]
            self.parts.append(part)
        part[2].append(op)

    @staticmethod
    def hstack(pieces):
        """Pieces (Canvas or ndarray) side by side, like np.hstack. Each keeps its commands inside its own area."""
        pieces = [p if isinstance(p, Canvas) else Canvas(p) for p in pieces]
        h = pieces[0].h
        if any(p.h != h for p in pieces):
            raise ValueError("pieces must have the same height")
        out = Canvas((h, sum(p.w for p in pieces)))
        out.parts, out._pieces = [], []
        x = 0
        for p in pieces:
            out._pieces.append((x, p))
            for dx, w, ops in p.parts:
                out.parts.append([x + dx, w, ops])
            x += p.w
        return out

    def base_image(self):
        """The picture under the commands, as one ndarray (no drawing)."""
        pieces = getattr(self, "_pieces", None)
        if pieces is None:
            if self.base is not None:
                return self.base
            return np.full((self.h, self.w, 3), self.fill if self.fill is not None else 0, np.uint8)
        return np.hstack([p.base_image() for _, p in pieces])

    def image(self):
        """Base picture with every command painted by OpenCV: the old drawn view, pixel for pixel."""
        img = self.base_image().copy()
        for dx, w, ops in self.parts:
            area = img[:, dx:dx + w]
            for op in ops:
                _paint(area, op)
        return img

    # Lets np.hstack, cv2.imwrite and friends take a Canvas as the painted picture.
    def __array__(self, dtype=None, copy=None):
        img = self.image()
        return img if dtype is None else img.astype(dtype)

    def _bases(self, x=0):
        """(x, width, picture or None, fill) for each piece of the base picture, left to right."""
        pieces = getattr(self, "_pieces", None)
        if pieces is None:
            return [(x, self.w, self.base, self.fill if self.fill is not None else 0)]
        return [b for dx, p in pieces for b in p._bases(x + dx)]

    def payload(self, quality=70):
        """What the web page gets: MAGIC, header length, JSON header, then the blobs the header lists in order.

        Blobs: one JPEG per real picture (a size-only panel is just its fill color), then each line mask as
        np.packbits rows, zlib-compressed (the page inflates it with DecompressionStream).
        """
        blobs, pics, sent = [], [], {}
        for x, w, img, fill in self._bases():
            if img is None:
                pics.append([x, w, _css((fill,) * 3 if np.isscalar(fill) else fill)])
                continue
            if id(img) not in sent:     # The same picture twice (camera view: raw and detection side by side) goes once.
                blobs.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes())
                sent[id(img)] = len(blobs) - 1
            pics.append([x, w, sent[id(img)]])
        minis = {}
        for name, img in ((self.extra or {}).get("minis") or {}).items():     # Small close-ups the page shows beside the view.
            blobs.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            minis[name] = len(blobs) - 1
        masks, parts = [], []
        for dx, w, ops in self.parts:
            out = []
            for op in ops:
                kind = op[0]
                if kind == "line":
                    out.append(["l", *op[1], _css(op[2]), op[3]])
                elif kind == "rect":
                    out.append(["r", *op[1], _css(op[2]), op[3]])
                elif kind == "circle":
                    out.append(["c", *op[1], op[2], _css(op[3]), op[4]])
                elif kind == "text":
                    out.append(["t", *op[1], op[2], _css(op[5]), op[6], op[8], op[9]])
                elif kind == "arrow":
                    out.append(["a", *op[1], _css(op[2]), op[3], op[5]])
                elif kind == "dash":
                    out.append(["d", *op[1], _css(op[2]), op[3], op[4], op[5]])
                elif kind == "poly":
                    out.append(["p", op[1].ravel().tolist(), _css(op[2]), op[3]])
                elif kind == "tint":
                    on = op[2]
                    masks.append(zlib.compress(np.packbits(on, axis=1).tobytes(), 1))
                    out.append(["m", op[1], len(blobs) + len(masks) - 1, _css(op[4]), op[5], on.shape[1], on.shape[0]])
                elif kind == "shade":
                    out.append(["s", *op[1], _css(op[2]), op[3]])
                elif kind == "fpoly":
                    out.append(["f", op[1].ravel().tolist(), _css(op[2]), op[3]])
            parts.append([dx, w, out])
        blobs += masks
        head = {"w": self.w, "h": self.h, "pics": pics, "blobs": [len(b) for b in blobs], "parts": parts}
        if self.extra is not None:
            head["state"], head["minis"] = self.extra.get("state"), minis
        head = json.dumps(head, separators=(",", ":")).encode()
        return b"".join([MAGIC, struct.pack(">I", len(head)), head, *blobs])
