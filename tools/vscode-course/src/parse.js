// Reading course.py text: function signatures, value tuples, and the call around the cursor.
// No VS Code API here, so Node tests can load it directly.

// Split "a, b=(1, 2), c='x,y'" at top-level commas.
function splitTop(s) {
  const out = [];
  let depth = 0, cur = "", quote = null;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (quote) {
      cur += ch;
      if (ch === "\\" && i + 1 < s.length) { cur += s[++i]; continue; }
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

// The quoted names in `NAME = ("a", "b", ...)`.
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
// { fn, argIndex, positional, kw, used, argStart, inString, strStart }, or null outside a call or in a comment.
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
        const m = text.slice(Math.max(0, i - 48), i).match(/(?:\b(def|class)\s+)?([A-Za-z_]\w*)\s*$/);
        // "def f(" and "class C(" are definitions, not calls.
        if (m && !m[1] && !/^(def|class|if|elif|while|for|return|and|or|not|in|lambda)$/.test(m[2])) fn = m[2];
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
      const m = text.slice(top.argStart, i).replace(/#[^\n]*/g, "").match(/^\s*([A-Za-z_]\w*)\s*$/);
      if (m) { top.kw = m[1]; top.used.add(m[1]); }
    }
    i++;
  }
  const top = stack[stack.length - 1];
  if (!top || !top.call || !top.fn) return null;
  return { ...top, inString: inString && inString.length === 1 ? inString : null, strStart };
}

// The parameter the cursor is on: the keyword being typed, or the positional slot.
function paramAt(fn, ctx) {
  if (ctx.kw) return ctx.kw;
  if (ctx.used.size) return null;
  const p = fn.params[ctx.argIndex];
  return p ? p.name : null;
}

module.exports = { splitTop, tupleNames, parseCourse, callContext, paramAt };
