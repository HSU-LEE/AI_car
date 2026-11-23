"""Run course.py text from stdin and report problems as JSON, for the VS Code helper.

Prints {"ok": bool, "steps": [...], "problems": [{"line", "col", "end", "msg", "level"}]}.
Lines are 1-based and point at the course entry that raised or printed the problem.
"""
import builtins
import difflib
import json
import sys
import traceback


def course_line(tb_frames, path):
    module_lines = [f.lineno for f in tb_frames if f.filename == path and f.name == "<module>"]
    return module_lines[-1] if module_lines else None


def main():
    path = sys.argv[1]
    src = sys.stdin.read()
    problems = []
    steps = []

    def report(line, msg, level, col=None, end=None):
        problems.append({"line": line or 1, "col": col, "end": end, "msg": msg, "level": level})

    try:
        code = compile(src, path, "exec")
    except SyntaxError as e:
        report(e.lineno, f"SyntaxError: {e.msg}", "error", (e.offset or 1) - 1, (e.end_offset or 0) - 1 if e.end_offset else None)
        print(json.dumps({"ok": False, "steps": [], "problems": problems}))
        return

    def course_print(*args, **kw):
        msg = " ".join(str(a) for a in args)
        line = course_line(traceback.extract_stack(), path)
        report(line, msg.replace("[warning] ", ""), "warning" if msg.startswith("[warning]") else "info")

    scope = {"__name__": "course_check", "__file__": path, "__builtins__": builtins, "print": course_print}
    try:
        exec(code, scope)
    except Exception as e:
        frames = traceback.extract_tb(e.__traceback__)
        msg = f"{type(e).__name__}: {e}"
        if isinstance(e, NameError) and getattr(e, "name", None):
            names = [k for k, v in scope.items() if callable(v) and not k.startswith("__")]
            near = difflib.get_close_matches(e.name, names, 1)
            if near and "did you mean" not in msg.lower():
                msg += f' -> did you mean "{near[0]}"?'
        report(course_line(frames, path), msg, "error")
    else:
        describe = scope.get("describe")
        for st in scope.get("COURSE") or []:
            try:
                steps.append(describe(st) if describe else str(st.get("do")))
            except Exception:
                steps.append(str(st.get("do")))
    print(json.dumps({"ok": not any(p["level"] == "error" for p in problems), "steps": steps, "problems": problems},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
