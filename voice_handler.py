import asyncio
import time
import logging
import numpy as np
import webrtcvad
import discord
import discord.ext.voice_recv as voice_recv

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000       # Discord 기본 샘플레이트
FRAME_DURATION_MS = 20    # 20ms 프레임
SILENCE_THRESHOLD = 0.8   # 발화 종료 판정 침묵 시간 (초)
MIN_SPEECH_DURATION = 0.3 # 최소 발화 길이 (초)
VAD_AGGRESSIVENESS = 2    # 0~3, 높을수록 민감

# webrtcvad는 16kHz, 10/20/30ms 프레임만 지원 (16000 * 0.02 * 2bytes = 640 bytes)
VAD_RATE = 16000
VAD_FRAME_BYTES = int(VAD_RATE * FRAME_DURATION_MS / 1000) * 2  # 640 bytes


class UserAudioBuffer:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.speech_frames: list[bytes] = []
        self.last_speech_time = 0.0
        self.is_speaking = False

    def process_frame(self, mono_48k: bytes) -> bytes | None:
        """48kHz mono PCM 프레임을 처리하고, 발화가 끝났으면 전체 오디오 반환"""
        # 48kHz → 16kHz 다운샘플링 (3:1)
        arr = np.frombuffer(mono_48k, dtype=np.int16)
        pcm_16k = arr[::3].tobytes()

        # 들어온 버퍼의 모든 완전한 20ms 16k 프레임에 대해 VAD 판정.
        # (Discord write가 20ms보다 길게 들어와도 뒷부분을 놓치지 않도록 전체 순회)
        is_speech = False
        for off in range(0, len(pcm_16k) - VAD_FRAME_BYTES + 1, VAD_FRAME_BYTES):
            frame = pcm_16k[off:off + VAD_FRAME_BYTES]
            try:
                if self.vad.is_speech(frame, VAD_RATE):
                    is_speech = True
                    break
            except Exception:
                continue

        now = time.time()
        if is_speech:
            self.is_speaking = True
            self.last_speech_time = now
            self.speech_frames.append(mono_48k)
        elif self.is_speaking:
            self.speech_frames.append(mono_48k)
            silence_duration = now - self.last_speech_time
            if silence_duration >= SILENCE_THRESHOLD:
                # 발화 종료 판정
                speech_duration = len(self.speech_frames) * FRAME_DURATION_MS / 1000
                result = None
                if speech_duration >= MIN_SPEECH_DURATION:
                    result = b"".join(self.speech_frames)
                self.speech_frames = []
                self.is_speaking = False
                return result

        return None


class AudioSink(voice_recv.AudioSink):
    def __init__(self, bot, vc, channel_id: int, respond_callback):
        super().__init__()
        self.bot = bot
        self.vc = vc
        self.channel_id = channel_id
        self.respond_callback = respond_callback
        self.user_buffers: dict[int, UserAudioBuffer] = {}
        self.processing: set[int] = set()  # STT 처리 중인 유저 (중복 STT 방지)

    def wants_opus(self) -> bool:
        return False  # PCM으로 받기

    def write(self, user: discord.User | None, data: voice_recv.VoiceData):
        if user is None or user.bot:
            return
        buf = self.user_buffers.setdefault(user.id, UserAudioBuffer())

        # data.pcm: 48kHz stereo PCM → mono (좌 채널만 사용)
        arr = np.frombuffer(data.pcm, dtype=np.int16)
        mono = arr[::2].tobytes()

        result = buf.process_frame(mono)
        if result and user.id not in self.processing:
            asyncio.run_coroutine_threadsafe(
                self._handle_speech(user, result),
                self.bot.loop,
            )

    async def _handle_speech(self, user: discord.User, audio_bytes: bytes):
        self.processing.add(user.id)
        t0 = time.time()
        try:
            text = await asyncio.get_running_loop().run_in_executor(
                None, transcribe_audio_sync, audio_bytes
            )
            logger.info(f"[STT] {user.name}: '{text}' ({time.time()-t0:.2f}s)")

            if not text or len(text.strip()) < 2:
                return

            # 봇이 말하는 중이어도 호출 → respond_callback이 barge-in(중단 후 재응답) 처리
            await self.respond_callback(self.vc, text, self.channel_id)
        except Exception:
            logger.exception("발화 처리 중 오류")
        finally:
            self.processing.discard(user.id)

    def cleanup(self):
        self.user_buffers.clear()
        self.processing.clear()


def transcribe_audio_sync(audio_bytes: bytes) -> str:
    from stt import transcribe_audio
    return transcribe_audio(audio_bytes)
