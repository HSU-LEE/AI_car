// Help text and value lists from docs.json. No VS Code API here.
const path = require("path");
const DOCS = require("../docs.json");

function isCourseFile(fileName) {
  return path.basename(fileName) === "course.py";
}

function paramDoc(fnName, param) {
  const f = DOCS.functions[fnName];
  return (f && f.params[param]) || DOCS.options[param] || "";
}

function signature(fn) {
  return `${fn.name}(${fn.params.map((p) => (p.def === null ? p.name : `${p.name}=${p.def}`)).join(", ")}${fn.options.length ? ", ..." : ""})`;
}

// Which docs.json value list a parameter takes, or null.
function valueKey(fnName, param) {
  if (param === "direction") return ["turn", "corner", "uturn"].includes(fnName) ? "direction" : null;
  if (param === "name") return fnName === "go_to" ? "name" : null;
  if (param === "line") return fnName === "cross" ? "line" : null;
  if (param === "angle") return ["straight", "push"].includes(fnName) ? "angle" : null;
  return ["until", "side", "style"].includes(param) ? param : null;
}

// [[value, description], ...]. until= follows EVENTS in course.py when it is there.
function valueList(key, info) {
  const docs = DOCS.values[key] || {};
  if (key === "until") {
    const evs = info.events.filter((e) => e);
    return (evs.length ? evs : Object.keys(docs)).map((v) => [v, docs[v] || ""]);
  }
  return Object.entries(docs);
}

module.exports = { DOCS, isCourseFile, paramDoc, signature, valueKey, valueList };
