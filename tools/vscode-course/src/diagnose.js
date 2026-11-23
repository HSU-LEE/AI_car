// Live check: runs py/check_course.py on the open text and shows its problems and the step count.
const vscode = require("vscode");
const cp = require("child_process");
const fs = require("fs");
const path = require("path");
const { isCourse } = require("./complete");

function pythonFor(doc) {
  const folder = vscode.workspace.getWorkspaceFolder(doc.uri);
  const roots = [folder && folder.uri.fsPath, path.dirname(path.dirname(doc.fileName))].filter(Boolean);
  for (const r of roots) {
    const p = path.join(r, ".venv", "bin", "python");
    if (fs.existsSync(p)) return p;
  }
  return "python3";
}

function toDiagnostic(doc, p) {
  const line = Math.min(Math.max(p.line - 1, 0), doc.lineCount - 1);
  const t = doc.lineAt(line);
  const start = p.col !== null && p.col >= 0 ? p.col : t.firstNonWhitespaceCharacterIndex;
  const end = p.end !== null && p.end > start ? p.end : t.text.trimEnd().length;
  const sev = p.level === "error" ? vscode.DiagnosticSeverity.Error
    : p.level === "warning" ? vscode.DiagnosticSeverity.Warning : vscode.DiagnosticSeverity.Information;
  const d = new vscode.Diagnostic(new vscode.Range(line, start, line, Math.max(end, start + 1)), p.msg, sev);
  d.source = "course";
  return d;
}

// Sets up the checker. Returns schedule(doc, ms) for the change and open events.
function startChecker(context) {
  const diags = vscode.languages.createDiagnosticCollection("aicar-course");
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  const output = vscode.window.createOutputChannel("AICAR course");
  const timers = new Map();
  const checker = path.join(__dirname, "..", "py", "check_course.py");
  let lastSteps = [];

  status.command = "aicar.showSteps";
  context.subscriptions.push(diags, status, output,
    vscode.commands.registerCommand("aicar.showSteps", () => {
      output.clear();
      lastSteps.forEach((s, i) => output.appendLine(`${String(i + 1).padStart(2)}. ${s}`));
      output.show(true);
    }),
    vscode.workspace.onDidCloseTextDocument((d) => { diags.delete(d.uri); }));

  function check(doc) {
    if (!isCourse(doc)) return;
    const version = doc.version;
    const showStatus = () => {
      const ed = vscode.window.activeTextEditor;
      if (ed && ed.document === doc) status.show();
    };
    const child = cp.execFile(pythonFor(doc), ["-I", checker, doc.fileName],
      { cwd: path.dirname(doc.fileName), timeout: 8000, maxBuffer: 4 << 20 },
      (err, stdout) => {
        if (doc.isClosed || doc.version !== version) return;
        let res;
        try { res = JSON.parse(stdout); } catch (e) {
          status.text = "$(warning) course.py 검사 실패";
          status.tooltip = String(err || e);
          showStatus();
          return;
        }
        diags.set(doc.uri, res.problems.map((p) => toDiagnostic(doc, p)));
        lastSteps = res.steps;
        status.text = res.ok ? `$(check) 코스 ${res.steps.length}단계` : "$(error) 코스 오류";
        status.tooltip = res.ok ? "눌러서 단계 목록 보기" : res.problems.filter((p) => p.level === "error").map((p) => p.msg).join("\n");
        showStatus();
      });
    child.stdin.on("error", () => {});
    child.stdin.end(doc.getText());
  }

  function schedule(doc, ms) {
    if (!isCourse(doc)) return;
    clearTimeout(timers.get(doc.uri.toString()));
    timers.set(doc.uri.toString(), setTimeout(() => check(doc), ms));
  }

  return { schedule, status };
}

module.exports = { startChecker, pythonFor };
