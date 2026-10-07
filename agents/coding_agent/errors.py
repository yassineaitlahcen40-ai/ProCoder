"""Safe, provider-independent errors from coding-agent operations."""


class CodingAgentError(Exception):
    """Base error with a user-safe message."""

    safe_message = "The Coding Agent failed. Check the local Codex CLI setup."


class CodexCliNotFoundError(CodingAgentError):
    safe_message = (
        "Codex CLI was not found. Install the official Codex CLI and sign in "
        "before using the generate command."
    )


class CodexExecutionError(CodingAgentError):
    safe_message = (
        "Codex CLI could not complete code generation. Check its local "
        "installation and authentication."
    )


class CodexResponseError(CodingAgentError):
    safe_message = "Codex returned an invalid coding result. No files were written."


class CodingRepairUnavailableError(CodingAgentError):
    safe_message = "The Coding Agent could not process the requested repair."
