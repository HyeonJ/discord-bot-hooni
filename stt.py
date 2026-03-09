import io
import wave
import numpy as np
from faster_whisper import WhisperModel

# 모델 로드 (최초 1회만, small이 속도/정확도 균형 최적)
# CPU에서도 충분히 빠름. GPU 있으면 device="cuda"
_model = WhisperModel("small", device="cpu", compute_type="int8")


def transcribe_audio(pcm_bytes: bytes, sample_rate: int = 48000) -> str:
    """48kHz mono PCM bytes → 텍스트"""
    # WAV 포맷으로 변환 (faster-whisper는 파일/스트림 입력)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    wav_buffer.seek(0)

    segments, info = _model.transcribe(
        wav_buffer,
        language="ko",
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()
