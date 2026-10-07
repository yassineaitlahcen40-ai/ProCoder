# ProCoder

ProCoder is an AI-powered multi-agent coding system that turns a natural-language programming request into a structured specification and code, tests that code in a restricted Docker sandbox, reviews the results with AI, and can automatically repair failures.

## System Overview

The current pipeline is:

```text
User Request
  -> Gemini Prompt/Planning Agent
  -> OpenAI Codex CLI Coding Agent
  -> Generated Project
  -> Docker Execution & Testing
  -> NVIDIA NIM Review Agent
  -> Python Repair Controller
  -> Codex Repair Agent (when required)
  -> Docker Retest
  -> NVIDIA Re-review
  -> Final Result
```

The Prompt Agent converts the request into a validated structured specification. Gemini is the current provider. The Coding Agent uses the OpenAI Codex CLI to return structured project files; ProCoder validates and writes them into a per-run workspace. When repair is needed, Codex returns a structured file patch instead of editing files or running generated code.

Docker is the authoritative environment for generated-code tests. NVIDIA NIM, configured to use Nemotron by default, reviews the specification, project source and actual Docker test evidence; it can diagnose issues but cannot claim that a failed Docker test passed.

Python controls workflow state, repair attempts, Docker retests, NVIDIA reviews and termination. LLMs do not control the loop. Generated project code is not executed directly on the host. Agent interfaces and provider adapters are separated so providers can be changed without rewriting the orchestration.

## Current Features

- Natural-language requests and validated software specifications.
- Structured code generation through the local Codex CLI.
- Isolated per-run generated project directories and path validation.
- Python standard-library `unittest` discovery and execution in Docker.
- Structured test results with exit code, captured output, duration, timeout and detectable test counts.
- NVIDIA NIM review of generated source and Docker evidence.
- A deterministic repair loop that obtains structured Codex patches, validates and applies them in the run workspace, then retests and re-reviews.
- Configurable bounded repair attempts.
- JSON Lines runtime logging and per-run persisted specification, generated-file metadata, test results, repair history and final result.
- Protections for generated workspace paths, symlinks, hard links, credential-like files and Docker configuration.

## Requirements

For a fresh development machine, install:

- **Python 3.12 or newer** and its standard `venv`/`pip` tools. The sandbox image itself uses Python 3.12. Python 3.12+ is the recommended host version.
- **Git**, to clone the repository.
- **Docker Desktop** on Windows or macOS, or a Docker Engine installation on Linux. Docker must be running for project tests and the complete workflow. No NVIDIA GPU is required by the local Docker runner.
- **OpenAI Codex CLI**, available as `codex` on `PATH` or in the installed OpenAI Codex VS Code extension. The CLI must be authenticated before generation or repair. The adapter uses Codex CLI authentication; ProCoder does not use an OpenAI API key for it.
- **Gemini API access** for specification generation. Obtain a key through [Google AI Studio](https://aistudio.google.com/) and keep it private.
- **NVIDIA NIM API access** for code review and diagnosis. Obtain/configure an API key through the [NVIDIA API Catalog](https://build.nvidia.com/) and verify access to the configured model. Availability, account access and any trial credits depend on NVIDIA's current terms.

**Node.js and npm are not runtime requirements for ProCoder.** They are needed only if you choose the npm installation method below for the Codex CLI.

## Installation

### 1. Clone the repository

Replace the URL placeholder with the repository URL after publishing:

```text
git clone <REPOSITORY_URL>
cd ProCoder
```

### 2. Create and activate a Python virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the Python launcher is unavailable, use the full path to a Python 3.12+ executable to create the environment. If PowerShell blocks activation, do not change the machine execution policy; use `.venv\Scripts\python.exe` directly in place of `python`.

macOS/Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

With the environment activated:

```sh
python -m pip install --requirement requirements.txt
```

The pinned runtime dependencies are listed in `requirements.txt`.

### 4. Install and verify Docker

Install Docker Desktop on Windows/macOS or Docker Engine on Linux using the vendor instructions for your operating system. Start Docker Desktop/the Docker service, then verify the CLI can reach the daemon:

```sh
docker info
```

The command should return server information without a daemon-connection error. Docker is required for the sandbox image, testing, and complete workflow.

### 5. Install and authenticate the Codex CLI

One available installation method, when Node.js/npm is installed, is:

```sh
npm install -g @openai/codex
```

Then start the CLI and complete its sign-in flow:

```sh
codex
```

Authenticate using the supported Codex sign-in option for your account, then exit the interactive CLI. ProCoder invokes `codex exec`; it does not receive or require an OpenAI API key. If Codex is already bundled with the OpenAI Codex VS Code extension, ProCoder can also discover that executable.

### 6. Configure Gemini and NVIDIA access

Create a Gemini API key in Google AI Studio and an NVIDIA API key with access to the hosted NVIDIA API Catalog/NIM model you intend to use. The default NVIDIA model is configurable; account availability is not guaranteed by the repository. Keep both keys private and configure them in the local `.env` file described below.

## Environment Configuration

`.env.example` is the source of truth for supported environment variable names and defaults. Copy it to a local `.env`:

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```sh
cp .env.example .env
```

Edit the copy and set the two provider keys required for the complete workflow:

```dotenv
GEMINI_API_KEY=your_key_here
NVIDIA_API_KEY=your_key_here
```

Do not commit `.env`; it is already excluded by `.gitignore`. Never put real keys in `.env.example`, source files, logs or documentation.

The remaining variables in `.env.example` configure model selection, provider timeouts/retries, the Codex executable/model, repair attempts and Docker resources:

| Variable | Purpose | Default |
| --- | --- | --- |
| `GEMINI_MODEL` | Primary specification-generation model | `gemini-2.5-flash-lite` |
| `GEMINI_FALLBACK_MODEL` | Optional fallback model; blank disables fallback | `gemini-3.1-flash-lite` |
| `GEMINI_TIMEOUT` | Gemini request timeout in seconds | `30` |
| `GEMINI_MAX_RETRIES` | Additional retries, from 0 to 5 | `2` |
| `CODEX_CLI` | Codex CLI executable name/path | `codex` |
| `CODEX_MODEL` | Optional Codex model override; blank uses Codex's own default | blank |
| `CODEX_TIMEOUT` | Codex operation timeout in seconds | `600` |
| `NVIDIA_MODEL` | NVIDIA NIM review model | `nvidia/nemotron-3.5-lightning-30b-a3b` |
| `NVIDIA_BASE_URL` | NVIDIA API base URL; must use HTTPS | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_TIMEOUT` | NVIDIA request timeout in seconds | `60` |
| `NVIDIA_MAX_RETRIES` | Additional retries, from 0 to 5 | `2` |
| `MAX_REPAIR_ATTEMPTS` | Maximum repair cycles after initial generation/test/review | `3` |
| `SANDBOX_TIMEOUT_SECONDS` | Docker execution time limit | `30` |
| `SANDBOX_MEMORY_LIMIT` | Docker memory limit | `256m` |
| `SANDBOX_CPU_LIMIT` | Docker CPU limit | `0.5` |
| `SANDBOX_PIDS_LIMIT` | Maximum container process count | `64` |
| `SANDBOX_MAX_OUTPUT_BYTES` | Combined stdout/stderr capture limit | `1048576` |
| `SANDBOX_DOCKER_IMAGE` | Local sandbox image tag | `procoder-sandbox:local` |

`OPENAI_API_KEY` appears in the example only as a reserved, blank setting; the current Codex adapter intentionally uses the Codex CLI's own authentication and does not use that key.

Validate configuration without calling providers:

```sh
python main.py --check-config
```

## Docker Sandbox Setup

From the repository root, build the image using the Dockerfile included in this repository:

```sh
docker build --tag procoder-sandbox:local --file sandbox\docker\Dockerfile .
```

On macOS/Linux, use forward slashes for the Dockerfile path:

```sh
docker build --tag procoder-sandbox:local --file sandbox/docker/Dockerfile .
```

Verify that the image exists:

```sh
docker image inspect procoder-sandbox:local
```

Optionally run the predetermined, non-AI smoke test:

```sh
python -m sandbox.smoke
```

Generated Python tests execute inside this image, not directly on the host. The runner uses a read-only mount of only the selected generated project, disables container networking, applies CPU/memory/PID limits and a timeout, and captures bounded output.

## Running ProCoder

Run commands from the repository root with the virtual environment activated. If it is not activated, replace `python` with the environment's Python executable (on Windows: `.venv\Scripts\python.exe`; on macOS/Linux: `.venv/bin/python`).

Create and display a structured specification using Gemini:

```sh
python main.py prompt "Create a Python function that determines whether a number is prime."
```

Generate and save a project using Gemini and Codex, without executing generated code:

```sh
python main.py generate "Create a Python function that determines whether a number is prime."
```

Test an existing generated run in Docker:

```sh
python main.py test <run_id>
```

Review an existing run's saved specification and Docker result using NVIDIA NIM, without rerunning Gemini, Codex or Docker:

```sh
python main.py review <run_id>
```

Run the complete generate, Docker-test, review and bounded-repair workflow:

```sh
python main.py run "Create a Python function that determines whether a number is prime."
```

The complete workflow requires a configured Gemini key, authenticated Codex CLI, Docker daemon and sandbox image, and a configured NVIDIA key/model. Output includes the run ID, generated files, test/review status, repair attempts and final termination/success summary. Generated projects are under `workspace/generated_code/<run_id>/`; per-run evidence is in `workspace/test_results/<run_id>.json`.

## Automatic Repair Loop

When the initial Docker tests fail or NVIDIA requires a repair, the Python controller runs:

```text
Docker FAIL or NVIDIA requires repair
  -> NVIDIA diagnosis/review (after Docker test)
  -> Codex proposes a structured repair patch
  -> ProCoder validates/applies the patch in the generated run workspace
  -> Docker retest
  -> NVIDIA re-review
  -> repeat until success, a terminal error, or attempt limit
```

The default maximum is **3 repair attempts**, configurable with `MAX_REPAIR_ATTEMPTS`. A run succeeds only when Docker passes with exit code 0 and no timeout or infrastructure error, and NVIDIA returns `PASS` with `repair_required=false`, `specification_satisfied=true`, and `tests_passed=true`. Docker infrastructure errors, provider failures, invalid repairs and persistence errors terminate the loop; an LLM cannot increase the attempt limit or override Docker evidence.

## Testing ProCoder

Run the repository test suite from the root:

```sh
python -m unittest discover -s tests -v
```

Docker-dependent integration tests require a running Docker daemon and the locally built `procoder-sandbox:local` image. The repair-loop integration test exercises a deliberately broken calculator fixture with offline provider doubles; it does not make live Gemini, NVIDIA or Codex requests. To run it separately:

```sh
python -m unittest tests.test_repair_controller.RepairLoopDockerIntegrationTests.test_broken_fixture_fails_docker_then_real_repair_passes_docker -v
```

## Project Structure

```text
agents/
  prompt_agent/       Specification agent and Gemini adapter
  coding_agent/       Coding interface and Codex CLI adapter
  review_agent/       Review interface and NVIDIA NIM adapter
core/                 Settings, typed models and logging
orchestrator/         Deterministic repair controller
sandbox/              Docker runner, image definition and smoke test
workspace/            Generated projects and run-metadata storage
tests/                Offline tests and Docker-conditional integrations
docs/                 Architecture and implementation notes
main.py               CLI entry point
requirements.txt      Pinned Python dependencies
.env.example          Supported configuration names/defaults
```

## Security Model

- Generated and repaired project code is run only by the Docker sandbox; Codex is instructed to return structured file data and does not execute generated code.
- The test container has `--network=none`, a read-only root filesystem, a non-root user, dropped capabilities, `no-new-privileges`, memory/CPU/PID limits, and a timeout.
- The container receives only a read-only bind mount of the selected generated run. It is not given the Docker socket or provider credentials; the Docker CLI process also uses a restricted environment allowlist.
- Workspace path validation rejects traversal, absolute paths, symlinks/junctions, hard links, protected credential/configuration files and Docker configuration. Repair patches are confined to the generated run workspace.
- `.env` and runtime outputs are excluded from Git. Provider prompts and credentials are not copied into runtime logs.
- Generated source, comments and process output are treated as untrusted data by review/repair prompts. Docker results, not LLM statements, determine whether tests actually passed.

Docker adds isolation but is not a guarantee against vulnerabilities in Docker, the host kernel or dependencies. Review generated code and keep the container runtime patched.

## Runtime Data

Runtime data is intentionally excluded from version control:

- `workspace/generated_code/` contains generated projects organized by run ID.
- `workspace/test_results/` contains saved specifications, generated metadata, test evidence, repair history and final results.
- `logs/` contains JSON Lines runtime logs.
- Python bytecode/cache and `.venv/` are also ignored.

Do not store credentials or unrelated sensitive files in generated project directories.

## Troubleshooting

- **Docker daemon not running:** Start Docker Desktop or the Docker Engine service, then confirm `docker info` succeeds. Verify the sandbox image with `docker image inspect procoder-sandbox:local`.
- **Codex CLI not found:** Install the Codex CLI and ensure `codex` is on `PATH`, or set `CODEX_CLI` to its executable path. The adapter can also discover the CLI bundled with the OpenAI Codex VS Code extension.
- **Codex not authenticated:** Run `codex` interactively and complete its supported sign-in process. ProCoder does not authenticate with `OPENAI_API_KEY`.
- **Gemini key missing:** Set `GEMINI_API_KEY` in `.env` or the process environment. The CLI reports that the key is missing before making a prompt request.
- **Gemini model unavailable/provider error:** Check that the configured model is enabled for your Gemini API project. `GEMINI_MODEL` and `GEMINI_FALLBACK_MODEL` are configurable; blank the fallback to disable it. Transient errors use bounded retries and supported availability errors may use the configured fallback.
- **NVIDIA key missing:** Set `NVIDIA_API_KEY` in `.env` or the process environment and confirm the NVIDIA API Catalog account can use `NVIDIA_MODEL`.
- **NVIDIA network/provider timeout:** Check internet access, DNS/proxy/firewall configuration and NVIDIA service/model availability. The client does not inherit proxy environment settings (`trust_env=False`); the configured `NVIDIA_TIMEOUT` and bounded retries apply. Safe network diagnostics are recorded in `logs/procoder.jsonl`.
- **Virtual environment not activated:** Activate it using the platform-specific command in Installation, or call the venv Python executable directly. If `pip` is unavailable, use `python -m pip`.

## Roadmap

Completed core: multi-agent coding pipeline, Docker execution/testing, NVIDIA review and bounded automatic repair.

Planned next: voice interaction, a VS Code extension/UI, and final product integration.

## Disclaimer

ProCoder executes AI-generated code in a restricted Docker environment, but container isolation is not absolute. Review generated code and use the project responsibly.

## License

No license has been selected yet.
