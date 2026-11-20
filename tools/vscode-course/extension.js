// AICAR course.py helper: hover help, parameter hints, option and value completion,
// and a live check of the course while you type. Descriptions live in docs.json, so
// course.py itself needs no comments.
const vscode = require("vscode");
const cp = require("child_process");
const fs = require("fs");
const path = require("path");
const DOCS = require("./docs.json");

function isCourse(doc) {
  return doc && doc.languageId === "python" && path.basename(doc.fileName) === "course.py";
}

// ---------- reading course.py ----------

function splitTop(s) {
  const out = [];
  let depth = 0, cur = "", quote = null;
  for (const ch of s) {
    if (quote) {
      cur += ch;
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") quote = ch;
    else if ("([{".includes(ch)) depth++;
    else if (")]}".includes(ch)) depth--;
    if (ch === "," && depth === 0) {
      out.push(cur.trim());
      cur = "";
    } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

function tupleNames(text, name) {
  const m = text.match(new RegExp("^" + name + "\\s*=\\s*\\(([\\s\\S]*?)\\)\\s*$", "m"));
  return m ? [...m[1].matchAll(/"(\w+)"/g)].map((x) => x[1]) : [];
}

const parseCache = new Map();

function parseCourse(text) {
  if (parseCache.has(text)) return parseCache.get(text);
  const fns = {};
  for (const m of text.matchAll(/^def ([A-Za-z]\w*|_(?:to|pass)_crosswalk)\((.*)\):/gm)) {
    const params = [];
    let kwargs = false;
    for (const p of splitTop(m[2])) {
      if (p.startsWith("**")) { kwargs = true; continue; }
      const eq = p.indexOf("=");
      params.push(eq < 0 ? { name: p, def: null } : { name: p.slice(0, eq).trim(), def: p.slice(eq + 1).trim() });
    }
    fns[m[1]] = { name: m[1], params, kwargs, line: text.slice(0, m.index).split("\n").length - 1 };
  }
  const loop = text.match(/^for _name in \(([\s\S]*?)\):/m);
  const wrapped = new Set(loop ? [...loop[1].matchAll(/"(\w+)"/g)].map((x) => x[1]) : []);
  const common = tupleNames(text, "COMMON");
  const options = tupleNames(text, "OPTIONS");
  for (const f of Object.values(fns)) {
    const extra = new Set();
    if (wrapped.has(f.name)) common.forEach((o) => extra.add(o));
    if (f.kwargs) options.forEach((o) => extra.add(o));
    f.params.forEach((p) => extra.delete(p.name));
    f.options = [...extra];
  }
  const result = { fns, events: tupleNames(text, "EVENTS") };
  if (parseCache.size > 4) parseCache.clear();
  parseCache.set(text, result);
  return result;
}

// Walks the text up to `offset` and returns the innermost open call:
// { fn, argIndex, kw, used, argStart, inString, strStart }.
function callContext(text, offset) {
  const stack = [];
  let i = 0, inString = null, strStart = -1;
  while (i < offset) {
    const ch = text[i];
    if (inString) {
      if (ch === "\\") { i += 2; continue; }
      if (text.startsWith(inString, i)) { i += inString.length; inString = null; continue; }
      i++;
      continue;
    }
    if (ch === "#") {
      const nl = text.indexOf("\n", i);
      if (nl < 0 || nl >= offset) return null;
      i = nl;
      continue;
    }
    if (ch === '"' || ch === "'") {
      const triple = text.startsWith(ch.repeat(3), i);
      inString = triple ? ch.repeat(3) : ch;
      i += inString.length;
      strStart = i;
      continue;
    }
    const top = stack[stack.length - 1];
    if (ch === "(" || ch === "[" || ch === "{") {
      let fn = null;
      if (ch === "(") {
        const m = text.slice(Math.max(0, i - 40), i).match(/([A-Za-z_]\w*)\s*$/);
        if (m && !/^(def|if|elif|while|for|return|and|or|not|in)$/.test(m[1])) fn = m[1];
      }
      stack.push({ fn, call: ch === "(", argIndex: 0, positional: 0, kw: null, used: new Set(), argStart: i + 1 });
    } else if (ch === ")" || ch === "]" || ch === "}") {
      stack.pop();
    } else if (top && ch === ",") {
      if (!top.kw && text.slice(top.argStart, i).trim()) top.positional++;
      top.argIndex++;
      top.kw = null;
      top.argStart = i + 1;
    } else if (top && top.call && ch === "=" && text[i + 1] !== "=" && !"<>!=".includes(text[i - 1])) {
      const m = text.slice(top.argStart, i).match(/^\s*([A-Za-z_]\w*)\s*$/);
      if (m) { top.kw = m[1]; top.used.add(m[1]); }
    }
    i++;
  }
  const top = stack[stack.length - 1];
  if (!top || !top.call || !top.fn) return null;
  return { ...top, inString: inString && inString.length === 1 ? inString : null, strStart };
}

function paramAt(fn, ctx) {
  if (ctx.kw) return ctx.kw;
  if (ctx.used.size) return null;
  const p = fn.params[ctx.argIndex];
  return p ? p.name : null;
}

// ---------- text ----------

function paramDoc(fnName, param) {
  const f = DOCS.functions[fnName];
  return (f && f.params[param]) || DOCS.options[param] || "";
}

function signature(fn) {
  return `${fn.name}(${fn.params.map((p) => (p.def === null ? p.name : `${p.name}=${p.def}`)).join(", ")}${fn.options.length ? ", ..." : ""})`;
}

function fnMarkdown(fn) {
  const d = DOCS.functions[fn.name] || { doc: "", params: {}, examples: [] };
  const md = new vscode.MarkdownString();
  md.appendCodeblock(signature(fn), "python");
  if (d.doc) md.appendMarkdown(d.doc + "\n\n");
  const rows = fn.params.filter((p) => paramDoc(fn.name, p.name));
  if (rows.length) md.appendMarkdown(rows.map((p) => `\`${p.name}\` ${paramDoc(fn.name, p.name)}`).join("  \n") + "\n\n");
  if (fn.options.length) md.appendMarkdown(`공통 옵션: ${fn.options.map((o) => `\`${o}=\``).join(" ")}\n\n`);
  if (d.examples && d.examples.length) md.appendCodeblock(d.examples.join("\n"), "python");
  return md;
}

function valueKey(fnName, param) {
  if (param === "direction") return ["turn", "corner", "uturn"].includes(fnName) ? "direction" : null;
  if (param === "name") return fnName === "go_to" ? "name" : null;
  if (param === "line") return fnName === "cross" ? "line" : null;
  if (param === "angle") return ["straight", "push"].includes(fnName) ? "angle" : null;
  return ["until", "side", "style"].includes(param) ? param : null;
}

function valueList(key, info) {
  const docs = DOCS.values[key] || {};
  if (key === "until") {
    const evs = info.events.filter((e) => e);
    return (evs.length ? evs : Object.keys(docs)).map((v) => [v, docs[v] || ""]);
  }
  return Object.entries(docs);
}

// ---------- providers ----------

const selector = { language: "python", pattern: "**/course.py" };

function hover(doc, pos) {
  if (!isCourse(doc)) return null;
  const text = doc.getText();
  const info = parseCourse(text);
  const range = doc.getWordRangeAtPosition(pos, /[A-Za-z_]\w*/);
  if (!range) return null;
  const word = doc.getText(range);
  const after = doc.lineAt(pos.line).text.slice(range.end.character);
  const ctx = callContext(text, doc.offsetAt(range.start));

  if (ctx && ctx.inString) {
    const fn = info.fns[ctx.fn];
    const key = fn && valueKey(fn.name, paramAt(fn, ctx));
    const d = key && DOCS.values[key] && DOCS.values[key][word];
    return d ? new vscode.Hover(`\`"${word}"\` ${d}`, range) : null;
  }
  if (/^\s*\(/.test(after) && info.fns[word] && DOCS.functions[word]) return new vscode.Hover(fnMarkdown(info.fns[word]), range);
  if (/^\s*=(?!=)/.test(after) && ctx && info.fns[ctx.fn]) {
    const d = paramDoc(ctx.fn, word);
    return d ? new vscode.Hover(`\`${word}=\` ${d}`, range) : null;
  }
  if (range.start.character === 0 && DOCS.settings[word]) return new vscode.Hover(`\`${word}\` ${DOCS.settings[word]}`, range);
  return null;
}

function signatureHelp(doc, pos) {
  if (!isCourse(doc)) return null;
  const text = doc.getText();
  const info = parseCourse(text);
  const ctx = callContext(text, doc.offsetAt(pos));
  const fn = ctx && info.fns[ctx.fn];
  if (!fn || !DOCS.functions[fn.name]) return null;
  const label = signature(fn);
  const sig = new vscode.SignatureInformation(label, new vscode.MarkdownString(DOCS.functions[fn.name].doc));
  let at = fn.name.length + 1;
  for (const p of fn.params) {
    const part = p.def === null ? p.name : `${p.name}=${p.def}`;
    sig.parameters.push(new vscode.ParameterInformation([at, at + part.length], paramDoc(fn.name, p.name)));
    at += part.length + 2;
  }
  const help = new vscode.SignatureHelp();
  help.signatures = [sig];
  help.activeSignature = 0;
  const name = paramAt(fn, ctx);
  const idx = fn.params.findIndex((p) => p.name === name);
  help.activeParameter = idx < 0 ? fn.params.length : idx;
  if (idx < 0 && name && fn.options.includes(name)) {
    sig.documentation = new vscode.MarkdownString(`\`${name}=\` ${paramDoc(fn.name, name)}`);
  }
  return help;
}

function completions(doc, pos) {
  if (!isCourse(doc)) return null;
  const text = doc.getText();
  const info = parseCourse(text);
  const offset = doc.offsetAt(pos);
  const ctx = callContext(text, offset);
  const fn = ctx && info.fns[ctx.fn];
  if (!fn || !DOCS.functions[fn.name]) return null;
  const param = paramAt(fn, ctx);

  if (ctx.inString) {
    const key = valueKey(fn.name, param);
    if (!key) return null;
    const range = new vscode.Range(doc.positionAt(ctx.strStart), pos);
    return valueList(key, info).map(([v, d], i) => {
      const item = new vscode.CompletionItem(v, vscode.CompletionItemKind.EnumMember);
      item.detail = d;
      item.range = range;
      item.sortText = String(i).padStart(3, "0");
      return item;
    });
  }

  const typed = text.slice(ctx.argStart, offset);
  const bareValue = typed.match(/^\s*([A-Za-z_]\w*)\s*=\s*$/);
  if (bareValue) {
    const key = valueKey(fn.name, bareValue[1]);
    if (!key) return null;
    return valueList(key, info).map(([v, d], i) => {
      const item = new vscode.CompletionItem(`"${v}"`, vscode.CompletionItemKind.EnumMember);
      item.detail = d;
      item.sortText = String(i).padStart(3, "0");
      return item;
    });
  }
  if (!/^\s*[A-Za-z_]*$/.test(typed)) return null;
  const items = [];
  const add = (name, rank, def) => {
    if (ctx.used.has(name)) return;
    const item = new vscode.CompletionItem(`${name}=`, vscode.CompletionItemKind.Property);
    item.insertText = new vscode.SnippetString(`${name}=$0`);
    item.filterText = name;
    item.detail = def === undefined || def === null ? "" : `기본 ${def}`;
    item.documentation = new vscode.MarkdownString(paramDoc(fn.name, name));
    item.sortText = `0${rank}${String(items.length).padStart(3, "0")}`;
    if (valueKey(fn.name, name)) item.command = { command: "editor.action.triggerSuggest", title: "" };
    items.push(item);
  };
  fn.params.forEach((p, i) => { if (i >= ctx.positional) add(p.name, 0, p.def); });
  fn.options.forEach((o) => add(o, 1, null));
  return items;
}

// ---------- live check ----------

function pythonFor(doc) {
  const folder = vscode.workspace.getWorkspaceFolder(doc.uri);
  const roots = [folder && folder.uri.fsPath, path.dirname(path.dirname(doc.fileName))].filter(Boolean);
  for (const r of roots) {
    const p = path.join(r, ".venv", "bin", "python");
    if (fs.existsSync(p)) return p;
  }
  return "python3";
}

function activate(context) {
  const diags = vscode.languages.createDiagnosticCollection("aicar-course");
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  const output = vscode.window.createOutputChannel("AICAR course");
  const timers = new Map();
  const checker = path.join(__dirname, "check_course.py");
  let lastSteps = [];

  status.command = "aicar.showSteps";
  context.subscriptions.push(diags, status, output,
    vscode.commands.registerCommand("aicar.showSteps", () => {
      output.clear();
      lastSteps.forEach((s, i) => output.appendLine(`${String(i + 1).padStart(2)}. ${s}`));
      output.show(true);
    }));

  function check(doc) {
    if (!isCourse(doc)) return;
    const child = cp.execFile(pythonFor(doc), ["-I", checker, doc.fileName],
      { cwd: path.dirname(doc.fileName), timeout: 8000, maxBuffer: 4 << 20 },
      (err, stdout) => {
        let res;
        try { res = JSON.parse(stdout); } catch (e) {
          status.text = "$(warning) course.py 검사 실패";
          status.tooltip = String(err || e);
          status.show();
          return;
        }
        const list = res.problems.map((p) => {
          const line = Math.min(Math.max(p.line - 1, 0), doc.lineCount - 1);
          const t = doc.lineAt(line);
          const start = p.col !== null && p.col >= 0 ? p.col : t.firstNonWhitespaceCharacterIndex;
          const end = p.end !== null && p.end > start ? p.end : t.text.trimEnd().length;
          const sev = p.level === "error" ? vscode.DiagnosticSeverity.Error
            : p.level === "warning" ? vscode.DiagnosticSeverity.Warning : vscode.DiagnosticSeverity.Information;
          const d = new vscode.Diagnostic(new vscode.Range(line, start, line, Math.max(end, start + 1)), p.msg, sev);
          d.source = "course";
          return d;
        });
        diags.set(doc.uri, list);
        lastSteps = res.steps;
        status.text = res.ok ? `$(check) 코스 ${res.steps.length}단계` : "$(error) 코스 오류";
        status.tooltip = res.ok ? "눌러서 단계 목록 보기" : res.problems.filter((p) => p.level === "error").map((p) => p.msg).join("\n");
        status.show();
      });
    child.stdin.end(doc.getText());
  }

  function schedule(doc, ms) {
    if (!isCourse(doc)) return;
    clearTimeout(timers.get(doc.uri.toString()));
    timers.set(doc.uri.toString(), setTimeout(() => check(doc), ms));
  }

  context.subscriptions.push(
    vscode.languages.registerHoverProvider(selector, { provideHover: hover }),
    vscode.languages.registerSignatureHelpProvider(selector, { provideSignatureHelp: signatureHelp }, "(", ",", "="),
    vscode.languages.registerCompletionItemProvider(selector, { provideCompletionItems: completions }, "(", ",", " ", '"', "'", "="),
    vscode.workspace.onDidOpenTextDocument((d) => schedule(d, 100)),
    vscode.workspace.onDidChangeTextDocument((e) => schedule(e.document, 600)),
    vscode.workspace.onDidCloseTextDocument((d) => { diags.delete(d.uri); }),
    vscode.window.onDidChangeActiveTextEditor((ed) => {
      if (ed && isCourse(ed.document)) { status.show(); schedule(ed.document, 100); } else status.hide();
    }));
  vscode.workspace.textDocuments.forEach((d) => schedule(d, 100));
}

function deactivate() {}

module.exports = { activate, deactivate, _test: { parseCourse, callContext, splitTop, paramAt, valueKey } };
