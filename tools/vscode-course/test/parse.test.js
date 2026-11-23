// Parser tests: node --test tools/vscode-course/test
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const path = require("path");
const { splitTop, tupleNames, parseCourse, callContext, paramAt } = require("../src/parse");
const { valueKey, valueList, signature, DOCS } = require("../src/model");

const SAMPLE = fs.readFileSync(path.join(__dirname, "fixtures", "course_sample.py"), "utf8");
const REAL = path.join(__dirname, "..", "..", "..", "pi", "course.py");

// Context at the "|" mark, which is removed from the text.
function at(src) {
  const i = src.indexOf("|");
  return callContext(src.slice(0, i) + src.slice(i + 1), i);
}

test("splitTop: commas and brackets inside strings and calls stay together", () => {
  assert.deepStrictEqual(splitTop(`a, b=(1, 2), c="x,y)", d='(,'`), ["a", "b=(1, 2)", 'c="x,y)"', "d='(,'"]);
  assert.deepStrictEqual(splitTop(`s="a\\",b", t=[1, {2: 3}]`), ['s="a\\",b"', "t=[1, {2: 3}]"]);
  assert.deepStrictEqual(splitTop(""), []);
});

test("tupleNames reads quoted names over several lines", () => {
  assert.deepStrictEqual(tupleNames(`EVENTS = ("a",\n  "b", None)\n`, "EVENTS"), ["a", "b"]);
  assert.deepStrictEqual(tupleNames("X = 1\n", "EVENTS"), []);
});

test("parseCourse: signatures, defaults, options from COMMON and **opt", () => {
  const info = parseCourse(SAMPLE);
  assert.ok(info.fns.follow && info.fns.go && info.fns.turn && info.fns.finish);
  assert.strictEqual(info.fns._helper, undefined, "private helpers are not actions");
  assert.deepStrictEqual(info.fns.go.params.map((p) => [p.name, p.def]), [["at", "None"], ["speed", "None"], ["nth", "1"]]);
  assert.deepStrictEqual(info.fns.follow.options.sort(), ["at", "max_sec", "min_sec", "side"]);
  assert.deepStrictEqual(info.fns.go.options, [], "no **opt and not wrapped: no extra options");
  assert.deepStrictEqual(info.events, ["junction", "end", "clear"]);
  assert.strictEqual(signature(info.fns.go), "go(at=None, speed=None, nth=1)");
});

test("callContext: positional, keyword, and the string being typed", () => {
  let c = at(`COURSE = [go(0.8, |`);
  assert.strictEqual(c.fn, "go");
  assert.strictEqual(c.argIndex, 1);
  assert.strictEqual(c.positional, 1);
  c = at(`follow(until="ju|`);
  assert.strictEqual(c.kw, "until");
  assert.strictEqual(c.inString, '"');
  c = at(`follow(sec=1, speed=|`);
  assert.strictEqual(c.kw, "speed");
  assert.deepStrictEqual([...c.used].sort(), ["sec", "speed"]);
});

test("callContext: parentheses and commas inside strings do not count", () => {
  const c = at(`follow(until="a,(b", |`);
  assert.strictEqual(c.fn, "follow");
  assert.strictEqual(c.argIndex, 1);
});

test("callContext: calls over several lines and nested calls", () => {
  let c = at(`COURSE = _flat([\n    follow(\n        until="end",\n        |`);
  assert.strictEqual(c.fn, "follow");
  assert.ok(c.used.has("until"));
  c = at(`go(at=max(0.5, |`);
  assert.strictEqual(c.fn, "max");
  c = at(`go(at=max(0.5, 0.6), |`);
  assert.strictEqual(c.fn, "go");
  assert.strictEqual(c.argIndex, 1);
});

test("callContext: comments", () => {
  assert.strictEqual(at(`# go(|`), null, "inside a comment");
  const c = at(`go(  # the first junction\n    at=|`);
  assert.strictEqual(c.fn, "go");
  assert.strictEqual(c.kw, "at");
  assert.strictEqual(at(`x = "#"; go(|`).fn, "go", "a # in a string is not a comment");
});

test("callContext: half-typed code does not throw", () => {
  for (const src of [`go(|`, `go(at=|`, `follow(until="|`, `COURSE = [go(at=0.8), tl(|`, `def f(|`, `|`, `go(""")|`]) {
    assert.doesNotThrow(() => at(src), src);
  }
  assert.strictEqual(at(`def f(|`), null, "a def is not a call");
  assert.strictEqual(at(`x = (|`), null, "a bare bracket is not a call");
  assert.strictEqual(at(`go(at=0.8)|`), null, "after the call closed");
});

test("paramAt and value lists", () => {
  const info = parseCourse(SAMPLE);
  assert.strictEqual(paramAt(info.fns.go, at(`go(0.8, |`)), "speed");
  assert.strictEqual(paramAt(info.fns.go, at(`go(at=0.8, |`)), null, "after a keyword: no positional slot");
  assert.strictEqual(valueKey("turn", "direction"), "direction");
  assert.strictEqual(valueKey("go", "direction"), null);
  assert.deepStrictEqual(valueList("until", info).map((v) => v[0]), ["junction", "end", "clear"]);
});

test("the real course.py: every action docs.json describes is parsed", () => {
  const info = parseCourse(fs.readFileSync(REAL, "utf8"));
  const missing = Object.keys(DOCS.functions).filter((n) => !info.fns[n]);
  assert.deepStrictEqual(missing, []);
  assert.ok(info.events.includes("junction"));
});

test("speed: parsing the real course.py and a context lookup stay fast", () => {
  const text = fs.readFileSync(REAL, "utf8") + "\n";
  const t0 = process.hrtime.bigint();
  for (let k = 0; k < 20; k++) parseCourse(text + "#".repeat(k));
  callContext(text, text.length);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / 20;
  assert.ok(ms < 50, `${ms.toFixed(1)} ms`);
});
