// The Python tools the extension runs: check_course.py and docs_check.py.
const test = require("node:test");
const assert = require("node:assert");
const cp = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..", "..", "..");
const VENV = path.join(ROOT, ".venv", "bin", "python");
const PY = fs.existsSync(VENV) ? VENV : "python3";
const TOOLS = path.join(__dirname, "..", "py");
const SAMPLE = path.join(__dirname, "fixtures", "course_sample.py");

function run(script, args, input) {
  const r = cp.spawnSync(PY, ["-I", path.join(TOOLS, script), ...args], { input, encoding: "utf8", timeout: 20000 });
  return { code: r.status, out: JSON.parse(r.stdout) };
}

test("docs_check: the real course.py is fully described", () => {
  const { code, out } = run("docs_check.py", [path.join(ROOT, "pi", "course.py")]);
  assert.deepStrictEqual(out.missing_functions, []);
  assert.deepStrictEqual(out.missing_options, []);
  assert.strictEqual(code, 0);
});

test("docs_check: a new undocumented action is found, private helpers are not", () => {
  const { code, out } = run("docs_check.py", [SAMPLE]);
  assert.ok(out.missing_functions.includes("undocumented_action"));
  assert.ok(!out.missing_functions.includes("_helper"));
  assert.ok(out.stale_functions.includes("tl"), "described in docs.json but not in this file");
  assert.strictEqual(code, 1);
});

test("check_course: the sample runs and lists its steps", () => {
  const { out } = run("check_course.py", [SAMPLE], fs.readFileSync(SAMPLE, "utf8"));
  assert.strictEqual(out.ok, true);
  assert.strictEqual(out.steps.length, 4);
});

test("check_course: a typo is an error on its line, with a suggestion", () => {
  const src = fs.readFileSync(SAMPLE, "utf8").replace("    finish(),", "    finsh(),");
  const { out } = run("check_course.py", [SAMPLE], src);
  assert.strictEqual(out.ok, false);
  const p = out.problems[0];
  assert.match(p.msg, /finish/);
  assert.strictEqual(src.split("\n")[p.line - 1].trim(), "finsh(),");
});

test("check_course: a syntax error is reported, not a crash", () => {
  const { out } = run("check_course.py", [SAMPLE], "COURSE = [follow(\n");
  assert.strictEqual(out.ok, false);
  assert.match(out.problems[0].msg, /SyntaxError/);
});
