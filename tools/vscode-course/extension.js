// AICAR course.py helper: hover help, parameter hints, option and value completion,
// and a live check of the course while you type. Descriptions live in docs.json, so
// course.py itself needs no comments. This file only registers the parts in src/.
const vscode = require("vscode");
const { isCourse, hover, signatureHelp, completions } = require("./src/complete");
const { startChecker } = require("./src/diagnose");

const selector = { language: "python", pattern: "**/course.py" };

function activate(context) {
  const { schedule, status } = startChecker(context);
  context.subscriptions.push(
    vscode.languages.registerHoverProvider(selector, { provideHover: hover }),
    vscode.languages.registerSignatureHelpProvider(selector, { provideSignatureHelp: signatureHelp }, "(", ",", "="),
    vscode.languages.registerCompletionItemProvider(selector, { provideCompletionItems: completions }, "(", ",", " ", '"', "'", "="),
    vscode.workspace.onDidOpenTextDocument((d) => schedule(d, 100)),
    vscode.workspace.onDidChangeTextDocument((e) => schedule(e.document, 600)),
    vscode.window.onDidChangeActiveTextEditor((ed) => {
      if (ed && isCourse(ed.document)) { status.show(); schedule(ed.document, 100); } else status.hide();
    }));
  vscode.workspace.textDocuments.forEach((d) => schedule(d, 100));
}

function deactivate() {}

module.exports = { activate, deactivate };
