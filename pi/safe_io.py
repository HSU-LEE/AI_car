"""Settings files that survive a power cut.

The car runs on a battery and is switched off by pulling the plug. Writing a file in place leaves it cut short or empty
if the power goes during the write, and the calibration or registration in it is gone. So:
  write_json  writes a temporary file, forces it to the card, then swaps it in whole. The file is always either the old
              content or the new, never half of one. The previous content is kept beside it as <name>.bak.
  read_json   reads the file, and falls back to <name>.bak when the file is damaged (an older half-written file, a bad
              card). A file that is simply not there stays "nothing saved": deleting one by hand resets that setting.
"""
import json
import os
from pathlib import Path


def backup_of(path):
    path = Path(path)
    return path.with_name(path.name + ".bak")


def write_json(path, data, backup=True, mode=None):
    """Save data as JSON at path, whole or not at all. backup: keep the previous (readable) content as <name>.bak."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(data)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644 if mode is None else mode)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())        # On the card, not just in memory, before it replaces anything.
        if backup and path.exists():
            try:
                old = path.read_text()
                json.loads(old)         # Only a readable file is worth keeping: a damaged one must not push out a good backup.
                bak_tmp = path.with_name(path.name + ".bak.tmp")
                bak_tmp.write_text(old)
                os.replace(bak_tmp, backup_of(path))     # A copy, so the file itself is never missing, not even for a moment.
            except (OSError, ValueError):
                pass
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    try:        # The swap itself is on the card only once the folder is.
        d = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    except OSError:
        pass


def read_json(path):
    """The JSON in path. If the file is damaged, the one in <name>.bak, with a warning.

    Raises OSError (no such file) or ValueError (damaged, and no readable backup), as reading the file directly would.
    """
    path = Path(path)
    text = path.read_text()
    try:
        return json.loads(text)
    except ValueError as damaged:
        bak = backup_of(path)
        try:
            data = json.loads(bak.read_text())
        except (OSError, ValueError):
            raise damaged from None
        print(f"[warning] {path.name} is damaged (cut off while saving?): using the previous one ({bak.name})")
        return data
