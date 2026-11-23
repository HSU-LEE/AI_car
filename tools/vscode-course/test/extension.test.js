// Loads the extension with a stand-in for the VS Code API and checks that it registers and answers.
const test = require("node:test");
const assert = require("node:assert");
const Module = require("module");
const fs = require("fs");
const path = require("path");

const REAL = path.join(__dirname, "..", "..", "..", "pi", "course.py");
const registered = {};

class Pos { constructor(line, character) { this.line = line; this.character = character; } }
class Range { constructor(a, b, c, d) { this.start = c === undefined ? a : new Pos(a, b); this.end = c === undefined ? b : new Pos(c, d); } }
class Markdown { constructor(v = "") { this.value = v; } appendMarkdown(s) { this.value += s; return this; } appendCodeblock(s) { this.value += s; return this; } }
const disposable = { dispose() {} };

const fakeVscode = {
  Range, Position: Pos, MarkdownString: Markdown,
  Hover: class { constructor(c, r) { this.contents = c; this.range = r; } },
  SignatureInformation: class { constructor(l, d) { this.label = l; this.documentation = d; this.parameters = []; } },
  ParameterInformation: class { constructor(l, d) { this.label = l; this.documentation = d; } },
  SignatureHelp: class {},
  SnippetString: class { constructor(v) { this.value = v; } },
  CompletionItem: class { constructor(l, k) { this.label = l; this.kind = k; } },
  CompletionItemKind: { EnumMember: 1, Property: 2 },
  DiagnosticSeverity: { Error: 0, Warning: 1, Information: 2 },
  StatusBarAlignment: { Left: 1 },
  languages: {
    createDiagnosticCollection: () => ({ set() {}, delete() {}, dispose() {} }),
    registerHoverProvider: (s, p) => { registered.hover = p; return disposable; },
    registerSignatureHelpProvider: (s, p) => { registered.sig = p; return disposable; },
    registerCompletionItemProvider: (s, p) => { registered.complete = p; return disposable; },
  },
  window: {
    createStatusBarItem: () => ({ show() {}, hide() {}, dispose() {} }),
    createOutputChannel: () => ({ clear() {}, appendLine() {}, show() {}, dispose() {} }),
    onDidChangeActiveTextEditor: () => disposable,
  },
  workspace: {
    textDocuments: [], getWorkspaceFolder: () => null,
    onDidOpenTextDocument: () => disposable, onDidChangeTextDocument: () => disposable, onDidCloseTextDocument: () => disposable,
  },
  commands: { registerCommand: () => disposable },
};

const load = Module._load;
Module._load = function (req, ...rest) { return req === "vscode" ? fakeVscode : load.call(this, req, ...rest); };

// A text document over a string, as the providers use it.
function doc(text) {
  const starts = [0];
  for (let i = 0; i < text.length; i++) if (text[i] === "\n") starts.push(i + 1);
  return {
    languageId: "python", fileName: "/x/pi/course.py", getText: (r) => (r ? text.slice(off(r.start), off(r.end)) : text),
    offsetAt: off, positionAt: (o) => { let l = 0; while (l + 1 < starts.length && starts[l + 1] <= o) l++; return new Pos(l, o - starts[l]); },
    lineAt: (l) => ({ text: text.slice(starts[l], (starts[l + 1] || text.length + 1) - 1) }),
    getWordRangeAtPosition(p, re) {
      const line = this.lineAt(p.line).text;
      for (const m of line.matchAll(new RegExp(re.source, "g"))) {
        if (m.index <= p.character && p.character <= m.index + m[0].length) return new Range(new Pos(p.line, m.index), new Pos(p.line, m.index + m[0].length));
      }
      return undefined;
    },
  };
  function off(p) { return starts[p.line] + p.character; }
}

test("activate registers hover, parameter hints and completion", () => {
  const ext = require("../extension");
  ext.activate({ subscriptions: [] });
  assert.ok(registered.hover && registered.sig && registered.complete);
});

test("providers answer on the real course.py text", () => {
  const base = fs.readFileSync(REAL, "utf8") + "\nX = [go(";
  const d = doc(base);
  const end = d.positionAt(base.length);
  const items = registered.complete.provideCompletionItems(d, end);
  assert.ok(items.some((i) => i.label === "at="), "go( offers at=");
  const help = registered.sig.provideSignatureHelp(d, end);
  assert.match(help.signatures[0].label, /^go\(/);
  const hoverAt = d.positionAt(base.length - 3);
  const h = registered.hover.provideHover(d, hoverAt);
  assert.ok(h && h.contents.value.includes("go("), "hover on go shows its signature");
});

test("a value list inside until=\"...\"", () => {
  const text = fs.readFileSync(REAL, "utf8") + '\nX = [follow(until="';
  const d = doc(text);
  const items = registered.complete.provideCompletionItems(d, d.positionAt(text.length));
  assert.ok(items.some((i) => i.label === "junction"));
});
