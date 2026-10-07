"""Voice request service and untrusted transcript validation."""

from voice.base import Recorder, TranscriptionProvider
from voice.errors import InvalidTranscriptError, VoiceRecordingError


MAX_TRANSCRIPT_CHARACTERS = 4_000


def validate_transcript(transcript: str) -> str:
    cleaned = transcript.strip()
    if (
        not cleaned
        or len(cleaned) > MAX_TRANSCRIPT_CHARACTERS
        or any(ord(character) < 32 and character not in "\n\t" for character in cleaned)
    ):
        raise InvalidTranscriptError()
    return cleaned


class VoiceInputService:
    def __init__(self, recorder: Recorder, provider: TranscriptionProvider) -> None:
        self._recorder = recorder
        self._provider = provider

    def transcribe_request(self, max_duration_seconds: int) -> str:
        recording = self._recorder.record(max_duration_seconds)
        try:
            return validate_transcript(self._provider.transcribe(recording))
        finally:
            try:
                recording.cleanup()
            except OSError as exc:
                raise VoiceRecordingError() from exc
