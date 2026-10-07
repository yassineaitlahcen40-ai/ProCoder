import * as childProcess from "node:child_process";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";

import {
  isSafeGeneratedRelativePath,
  isSafeRunId,
  parseBridgeEvent,
  transition,
  type BridgeEvent,
  type WorkflowState,
} from "./protocol";

const MAX_REQUEST_LENGTH = 12_000;

class ProcoderViewProvider implements vscode.WebviewViewProvider, vscode.Disposable {
  private view?: vscode.WebviewView;
  private backend?: childProcess.ChildProcessWithoutNullStreams;
  private lineBuffer = "";
  private droppingOversizedLine = false;
  private state: WorkflowState = { status: "idle", stage: "Idle" };
  private readonly disposables: vscode.Disposable[] = [];

  constructor(private readonly context: vscode.ExtensionContext) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true };
    view.webview.html = this.renderHtml(view.webview);
    view.webview.onDidReceiveMessage(
      (message: unknown) => void this.onMessage(message),
      undefined,
      this.disposables,
    );
    view.onDidDispose(() => {
      this.view = undefined;
    });
    this.post({ type: "state", state: this.state });
  }

  dispose(): void {
    this.disposables.forEach((item) => item.dispose());
    if (this.backend && this.backend.exitCode === null) {
      this.send({ type: "shutdown" });
      this.backend.stdin.end();
    }
  }

  private async onMessage(value: unknown): Promise<void> {
    if (typeof value !== "object" || value === null || !("type" in value)) {
      return;
    }
    const message = value as Record<string, unknown>;
    switch (message.type) {
      case "run": {
        const request = message.request;
        if (typeof request !== "string" || !request.trim() || request.length > MAX_REQUEST_LENGTH) {
          this.showLocalError("Enter a request (maximum 12,000 characters).");
          return;
        }
        if (this.state.status === "awaiting_confirmation") {
          this.showLocalError("Approve or reject the voice transcript first.");
          return;
        }
        if (!(await this.ensureBackend())) {
          return;
        }
        this.send({ type: "run", request });
        return;
      }
      case "voiceStart":
        if (await this.ensureBackend()) {
          this.send({ type: "voice_start" });
        }
        return;
      case "voiceStop":
        this.send({ type: "voice_stop" });
        return;
      case "cancel":
        this.send({ type: "cancel" });
        return;
      case "transcriptDecision": {
        if (
          typeof message.approved !== "boolean" ||
          (message.transcript !== undefined && typeof message.transcript !== "string")
        ) {
          return;
        }
        this.send({
          type: "transcript_decision",
          approved: message.approved,
          transcript: message.transcript,
        });
        return;
      }
      case "openFile":
        await this.openGeneratedFile(message.runId, message.file);
        return;
      default:
        return;
    }
  }

  private async ensureBackend(): Promise<boolean> {
    if (this.backend && this.backend.exitCode === null) {
      return true;
    }
    const root = getBackendRoot();
    if (!root) {
      this.showLocalError("Open the ProCoder project folder in VS Code first.");
      return false;
    }
    const python = vscode.workspace
      .getConfiguration("procoder")
      .get<string>("pythonPath", "python");
    const env: NodeJS.ProcessEnv = {};
    for (const key of [
      "PATH",
      "SYSTEMROOT",
      "WINDIR",
      "TEMP",
      "TMP",
      "USERPROFILE",
      "APPDATA",
      "LOCALAPPDATA",
      "HOME",
    ]) {
      const entry = process.env[key];
      if (entry !== undefined) {
        env[key] = entry;
      }
    }
    env.PYTHONUTF8 = "1";
    env.PYTHONIOENCODING = "utf-8";
    try {
      const backend = childProcess.spawn(python, ["-m", "bridge"], {
        cwd: root,
        env,
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true,
      });
      this.backend = backend;
      backend.stdout.setEncoding("utf8");
      backend.stdout.on("data", (chunk: string) => this.onStdout(chunk));
      backend.stderr.on("data", () => {
        // Backend diagnostics are kept out of the UI and protocol channel.
      });
      backend.on("error", () => {
        this.showLocalError("Could not start the Python backend. Check procoder.pythonPath.");
        this.backend = undefined;
      });
      backend.stdin.on("error", () => {
        this.showLocalError("The Python backend is not accepting commands.");
      });
      backend.on("exit", (code) => {
        this.backend = undefined;
        this.lineBuffer = "";
        if (code !== 0 && this.state.status !== "success") {
          this.showLocalError("The ProCoder backend stopped unexpectedly.");
        }
      });
      return true;
    } catch {
      this.showLocalError("Could not start the Python backend.");
      return false;
    }
  }

  private onStdout(chunk: string): void {
    let remainder = chunk;
    if (this.droppingOversizedLine) {
      const nextLine = remainder.indexOf("\n");
      if (nextLine < 0) {
        return;
      }
      remainder = remainder.slice(nextLine + 1);
      this.droppingOversizedLine = false;
    }
    this.lineBuffer += remainder;
    let newline = this.lineBuffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.lineBuffer.slice(0, newline);
      this.lineBuffer = this.lineBuffer.slice(newline + 1);
      if (Buffer.byteLength(line, "utf8") > 1_000_000) {
        this.showLocalError("The backend sent an invalid protocol event.");
      } else {
        const event = parseBridgeEvent(line);
        if (!event) {
          this.showLocalError("The backend sent an invalid protocol event.");
        } else {
          this.onBridgeEvent(event);
        }
      }
      newline = this.lineBuffer.indexOf("\n");
    }
    if (Buffer.byteLength(this.lineBuffer, "utf8") > 1_000_000) {
      this.lineBuffer = "";
      this.droppingOversizedLine = true;
      this.showLocalError("The backend sent an oversized protocol message.");
    }
  }

  private onBridgeEvent(event: BridgeEvent): void {
    this.state = transition(this.state, event);
    this.post({ type: "state", state: this.state });
    this.post({ type: "event", event });
    if (event.type === "workflow_completed") {
      this.view?.webview.postMessage({
        type: "result",
        result: event,
      });
    }
  }

  private send(message: Record<string, unknown>): void {
    if (!this.backend || this.backend.exitCode !== null) {
      this.showLocalError("The ProCoder backend is not running.");
      return;
    }
    try {
      this.backend.stdin.write(
        `${JSON.stringify({ ...message, protocol_version: 1 })}\n`,
      );
    } catch {
      this.showLocalError("The Python backend is not accepting commands.");
    }
  }

  private async openGeneratedFile(runIdValue: unknown, fileValue: unknown): Promise<void> {
    if (
      typeof runIdValue !== "string" ||
      !isSafeRunId(runIdValue) ||
      typeof fileValue !== "string" ||
      !isSafeGeneratedRelativePath(fileValue)
    ) {
      this.showLocalError("That generated file path is not allowed.");
      return;
    }
    const root = getBackendRoot();
    if (!root) {
      this.showLocalError("Open the ProCoder project folder first.");
      return;
    }
    const generatedRoot = path.resolve(root, "workspace", "generated_code", runIdValue);
    const target = path.resolve(generatedRoot, ...fileValue.split("/"));
    const relative = path.relative(generatedRoot, target);
    if (relative.startsWith("..") || path.isAbsolute(relative)) {
      this.showLocalError("That generated file path is not allowed.");
      return;
    }
    try {
      const workspacePath = path.join(root, "workspace");
      const generatedPath = path.join(workspacePath, "generated_code");
      for (const directory of [workspacePath, generatedPath]) {
        const info = await fs.promises.lstat(directory);
        if (!info.isDirectory() || info.isSymbolicLink()) {
          throw new Error("unsafe generated workspace");
        }
      }
      const rootInfo = await fs.promises.lstat(generatedRoot);
      if (!rootInfo.isDirectory() || rootInfo.isSymbolicLink()) {
        throw new Error("unsafe root");
      }
      const canonicalRoot = await fs.promises.realpath(generatedRoot);
      const canonicalTarget = await fs.promises.realpath(target);
      const canonicalRelative = path.relative(canonicalRoot, canonicalTarget);
      if (canonicalRelative.startsWith("..") || path.isAbsolute(canonicalRelative)) {
        throw new Error("outside generated workspace");
      }
      let current = generatedRoot;
      for (const part of fileValue.split("/")) {
        current = path.join(current, part);
        const info = await fs.promises.lstat(current);
        if (info.isSymbolicLink()) {
          throw new Error("unsafe link");
        }
      }
      const info = await fs.promises.lstat(target);
      if (!info.isFile() || info.nlink > 1) {
        throw new Error("not a regular generated file");
      }
      const document = await vscode.workspace.openTextDocument(vscode.Uri.file(target));
      await vscode.window.showTextDocument(document, { preview: false });
    } catch {
      this.showLocalError("The generated file is unavailable or unsafe to open.");
    }
  }

  private showLocalError(message: string): void {
    this.state = { ...this.state, status: "failed", stage: "Failed", error: message };
    this.post({ type: "state", state: this.state });
  }

  private post(message: Record<string, unknown>): void {
    void this.view?.webview.postMessage(message);
  }

  private renderHtml(webview: vscode.Webview): string {
    const nonce = getNonce();
    const csp = `default-src 'none'; style-src 'nonce-${nonce}' ${webview.cspSource}; script-src 'nonce-${nonce}';`;
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style nonce="${nonce}">
    :root { color-scheme: light dark; }
    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 12px; }
    h1 { font-size: 18px; margin: 0 0 4px; }
    .muted { color: var(--vscode-descriptionForeground); font-size: 12px; }
    textarea { box-sizing: border-box; width: 100%; min-height: 116px; resize: vertical; margin: 12px 0 8px; padding: 8px; color: var(--vscode-input-foreground); background: var(--vscode-input-background); border: 1px solid var(--vscode-input-border, transparent); }
    button { border: 0; border-radius: 3px; padding: 7px 10px; margin: 0 5px 6px 0; color: var(--vscode-button-foreground); background: var(--vscode-button-background); cursor: pointer; }
    button.secondary { color: var(--vscode-button-secondaryForeground); background: var(--vscode-button-secondaryBackground); }
    button:disabled { opacity: .55; cursor: default; }
    .card { margin-top: 12px; padding: 10px; border: 1px solid var(--vscode-panel-border); border-radius: 5px; }
    #status { font-weight: 600; }
    #error { color: var(--vscode-errorForeground); white-space: pre-wrap; }
    #files button { display: block; width: 100%; text-align: left; overflow-wrap: anywhere; }
    .result-row { margin: 5px 0; }
    #transcript { min-height: 76px; }
    #transcriptActions { display: none; }
  </style>
</head>
<body>
  <h1>ProCoder</h1>
  <div class="muted">Local VS Code control for the ProCoder backend</div>
  <textarea id="request" maxlength="${MAX_REQUEST_LENGTH}" placeholder="Describe the code you want ProCoder to build or change..."></textarea>
  <div>
    <button id="run">Run request</button>
    <button id="voice" class="secondary">Start voice</button>
    <button id="stopVoice" class="secondary" hidden>Stop voice</button>
    <button id="cancel" class="secondary" hidden>Cancel voice</button>
  </div>
  <section id="statusCard" class="card">
    <div id="status">Idle</div>
    <div id="runId" class="muted"></div>
    <div id="error"></div>
  </section>
  <section id="transcriptCard" class="card" hidden>
    <strong>Review voice transcript</strong>
    <textarea id="transcript" maxlength="4000"></textarea>
    <div id="transcriptActions">
      <button id="approve">Approve and run</button>
      <button id="reject" class="secondary">Reject</button>
    </div>
  </section>
  <section id="results" class="card" hidden>
    <strong>Latest result</strong>
    <div id="summary"></div>
    <div id="files"></div>
  </section>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const request = document.getElementById("request");
    const status = document.getElementById("status");
    const runId = document.getElementById("runId");
    const error = document.getElementById("error");
    const transcriptCard = document.getElementById("transcriptCard");
    const transcript = document.getElementById("transcript");
    const transcriptActions = document.getElementById("transcriptActions");
    const results = document.getElementById("results");
    const summary = document.getElementById("summary");
    const files = document.getElementById("files");
    const stopVoice = document.getElementById("stopVoice");
    const cancel = document.getElementById("cancel");
    document.getElementById("run").addEventListener("click", () => vscode.postMessage({ type: "run", request: request.value }));
    document.getElementById("voice").addEventListener("click", () => vscode.postMessage({ type: "voiceStart" }));
    stopVoice.addEventListener("click", () => vscode.postMessage({ type: "voiceStop" }));
    cancel.addEventListener("click", () => vscode.postMessage({ type: "cancel" }));
    document.getElementById("approve").addEventListener("click", () => vscode.postMessage({ type: "transcriptDecision", approved: true, transcript: transcript.value }));
    document.getElementById("reject").addEventListener("click", () => vscode.postMessage({ type: "transcriptDecision", approved: false }));
    window.addEventListener("message", (event) => {
      const message = event.data;
      if (message.type === "state") {
        const state = message.state;
        status.textContent = state.stage || state.status;
        runId.textContent = state.runId ? "Run ID: " + state.runId : "";
        error.textContent = state.error || "";
        stopVoice.hidden = state.status !== "recording";
        cancel.hidden = state.status !== "recording";
        if (state.status !== "awaiting_confirmation") {
          transcriptActions.style.display = "none";
        }
      }
      if (message.type === "event" && message.event.type === "transcript_ready") {
        transcript.value = message.event.transcript;
        transcriptCard.hidden = false;
        transcriptActions.style.display = "block";
      }
      if (message.type === "event" && (message.event.type === "transcript_rejected" || message.event.type === "workflow_started")) {
        transcriptCard.hidden = true;
      }
      if (message.type === "result") {
        const result = message.result;
        results.hidden = false;
        summary.replaceChildren();
        const values = [
          "Run ID: " + (result.run_id || "unknown"),
          "Termination: " + (result.termination_reason || "unknown"),
          "Repair attempts: " + (result.repair_attempts ?? 0),
          "Docker: " + (result.docker ? (result.docker.passed ? "PASS" : "FAIL") + " (exit " + result.docker.exit_code + (result.docker.timed_out ? ", timed out" : "") + ")" : "No result"),
          "NVIDIA: " + (result.nvidia ? result.nvidia.verdict : "No review"),
          "Final: " + (result.success ? "SUCCESS" : "FAILED")
        ];
        for (const text of values) {
          const row = document.createElement("div");
          row.className = "result-row";
          row.textContent = text;
          summary.appendChild(row);
        }
        files.replaceChildren();
        for (const file of result.files_created || []) {
          const row = document.createElement("div");
          row.className = "result-row";
          row.textContent = "Created: " + file;
          summary.appendChild(row);
        }
        for (const file of result.files_modified || []) {
          const row = document.createElement("div");
          row.className = "result-row";
          row.textContent = "Modified: " + file;
          summary.appendChild(row);
        }
        for (const attempt of result.repair_history || []) {
          const row = document.createElement("div");
          row.className = "result-row";
          row.textContent = "Repair " + attempt.attempt + ": " + (attempt.patch_applied ? "applied" : "not applied");
          summary.appendChild(row);
        }
        for (const file of result.files || []) {
          const button = document.createElement("button");
          button.className = "secondary";
          button.textContent = file;
          button.addEventListener("click", () => vscode.postMessage({ type: "openFile", runId: result.run_id, file }));
          files.appendChild(button);
        }
      }
    });
  </script>
</body>
</html>`;
  }
}

export function activate(context: vscode.ExtensionContext): void {
  const provider = new ProcoderViewProvider(context);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("procoder.sidebar", provider),
    vscode.commands.registerCommand("procoder.focus", () =>
      vscode.commands.executeCommand("workbench.view.extension.procoder"),
    ),
    provider,
  );
}

export function deactivate(): void {}

function getBackendRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders ?? [];
  for (const folder of folders) {
    const candidate = folder.uri.fsPath;
    if (hasBackend(candidate)) {
      return candidate;
    }
    if (path.basename(candidate).toLowerCase() === "vscode-extension") {
      const parent = path.dirname(candidate);
      if (hasBackend(parent)) {
        return parent;
      }
    }
  }
  return undefined;
}

function hasBackend(directory: string): boolean {
  return (
    fs.existsSync(path.join(directory, "main.py")) &&
    fs.existsSync(path.join(directory, "bridge.py"))
  );
}

function getNonce(): string {
  return crypto.randomBytes(16).toString("base64");
}
