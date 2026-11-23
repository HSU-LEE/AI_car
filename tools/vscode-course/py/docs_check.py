"""Which course.py functions and options docs.json does not describe, for the VS Code helper and the tests.

    python -I tools/vscode-course/py/docs_check.py [path/to/course.py]

Prints {"missing_functions", "missing_options", "missing_params", "stale_functions"} as JSON.
Exit code 1 when a function or option is missing (params are reported only).
"""
import builtins
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs.json"
DEFAULT_COURSE = HERE.parents[2] / "pi" / "course.py"
ADVANCED = ("_to_crosswalk", "_pass_crosswalk")     # Course actions whose names start with "_".
NOT_ACTIONS = ("describe",)                         # Public names that are helpers, not course actions.
MODULE = "course_docs_check"


def course_api(path):
    """(functions {name: [param names]}, options) from running course.py without its prints."""
    import inspect
    scope = {"__name__": MODULE, "__file__": str(path), "__builtins__": builtins, "print": lambda *a, **k: None}
    exec(compile(Path(path).read_text(encoding="utf-8"), str(path), "exec"), scope)
    fns = {}
    for name, v in scope.items():
        if not inspect.isfunction(v) or name in NOT_ACTIONS:
            continue
        if name.startswith("_") and name not in ADVANCED:
            continue
        target = inspect.unwrap(v)
        if target.__module__ != MODULE and v.__module__ != MODULE:
            continue
        params = [p.name for p in inspect.signature(target).parameters.values()
                  if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
        fns[name] = params
    options = set(scope.get("OPTIONS", ())) | set(scope.get("COMMON", ())) | {"until"}
    return fns, sorted(options)


def check(course_path=DEFAULT_COURSE, docs_path=DOCS):
    docs = json.loads(Path(docs_path).read_text(encoding="utf-8"))
    fns, options = course_api(course_path)
    documented = docs["functions"]
    missing_params = {}
    for name, params in fns.items():
        own = documented.get(name, {}).get("params", {})
        gaps = [p for p in params if p not in own and p not in docs["options"]]
        if gaps:
            missing_params[name] = gaps
    return {
        "missing_functions": sorted(n for n in fns if n not in documented),
        "missing_options": [o for o in options if o not in docs["options"]],
        "missing_params": missing_params,
        "stale_functions": sorted(n for n in documented if n not in fns),
    }


def main():
    res = check(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COURSE)
    print(json.dumps(res, ensure_ascii=False))
    sys.exit(1 if res["missing_functions"] or res["missing_options"] else 0)


if __name__ == "__main__":
    main()
