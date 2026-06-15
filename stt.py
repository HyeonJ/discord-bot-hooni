import os
import io
import wave
from faster_whisper import WhisperModel

# 모델 로드 (최초 1회만)
# - "small"        : CPU 기본값. 속도/정확도 균형
# - "large-v3-turbo": 2026 기준 정확도/속도 균형의 표준 (GPU 권장, ~1.5GB VRAM)
# - "distil-large-v3.5": 장문에서 turbo보다 ~1.5배 빠름
# GPU 있으면 WHISPER_DEVICE=cuda, WHISPER_COMPUTE=float16 권장
_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
_model = WhisperModel(_MODEL_NAME, device=_DEVICE, compute_type=_COMPUTE)


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
