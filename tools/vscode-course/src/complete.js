// Hover, parameter hints and completion for course.py.
const vscode = require("vscode");
const { parseCourse, callContext, paramAt } = require("./parse");
const { DOCS, isCourseFile, paramDoc, signature, valueKey, valueList } = require("./model");

function isCourse(doc) {
  return doc && doc.languageId === "python" && isCourseFile(doc.fileName);
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

module.exports = { isCourse, hover, signatureHelp, completions };
