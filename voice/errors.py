"""Safe user-facing errors for voice capture and transcription."""


class VoiceError(Exception):
    """Base class for expected voice-input failures."""

    safe_message = "Voice input could not be completed."


class MicrophoneUnavailableError(VoiceError):
    safe_message = (
        "No microphone is available. Connect or enable an input device, "
        "then try again."
    )


class VoiceRecordingError(VoiceError):
    safe_message = "Audio recording failed. Check microphone access and try again."


class VoiceTranscriptionError(VoiceError):
    safe_message = (
        "Local transcription failed. Check the voice model installation/download "
        "and try again."
    )

    def __init__(self, diagnostics: dict[str, object] | None = None) -> None:
        super().__init__(self.safe_message)
        self.diagnostics = diagnostics or {}


class InvalidTranscriptError(VoiceError):
    safe_message = "The transcription was empty or exceeded the allowed length."
