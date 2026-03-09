import asyncio
import time
import logging
import collections
import numpy as np
import webrtcvad
import discord
import discord.ext.voice_recv as voice_recv

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000       # Discord 기본 샘플레이트
FRAME_DURATION_MS = 20    # 20ms 프레임
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 960 샘플
SILENCE_THRESHOLD = 0.8   # 발화 종료 판정 침묵 시간 (초)
MIN_SPEECH_DURATION = 0.3 # 최소 발화 길이 (초)
VAD_AGGRESSIVENESS = 2    # 0~3, 높을수록 민감


class UserAudioBuffer:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.frames: list[bytes] = []
        self.speech_frames: list[bytes] = []
        self.last_speech_time = 0.0
        self.is_speaking = False
        # 16kHz 변환용 (webrtcvad는 16kHz 지원)
        self.ring = collections.deque(maxlen=FRAME_SIZE * 4)

    def process_frame(self, pcm_48k: bytes) -> bytes | None:
        """48kHz PCM 프레임을 처리하고, 발화가 끝났으면 전체 오디오 반환"""
        # 48kHz → 16kHz 다운샘플링 (3:1)
        arr = np.frombuffer(pcm_48k, dtype=np.int16)
        arr_16k = arr[::3].tobytes()

        # webrtcvad는 16kHz 10/20/30ms 프레임 필요 (16000 * 0.02 * 2 = 640 bytes)
        frame_16k = arr_16k[:640]
        if len(frame_16k) < 640:
            return None

        try:
            is_speech = self.vad.is_speech(frame_16k, 16000)
        except Exception:
            is_speech = False

        now = time.time()
        if is_speech:
            self.is_speaking = True
            self.last_speech_time = now
            self.speech_frames.append(pcm_48k)
        elif self.is_speaking:
            self.speech_frames.append(pcm_48k)
            silence_duration = now - self.last_speech_time
            if silence_duration >= SILENCE_THRESHOLD:
                # 발화 종료 판정
                speech_duration = len(self.speech_frames) * FRAME_DURATION_MS / 1000
                if speech_duration >= MIN_SPEECH_DURATION:
                    result = b"".join(self.speech_frames)
                    self.speech_frames = []
                    self.is_speaking = False
                    return result
                else:
                    self.speech_frames = []
                    self.is_speaking = False

        return None


class AudioSink(voice_recv.AudioSink):
    def __init__(self, bot, vc, channel_id: int, respond_callback):
        super().__init__()
        self.bot = bot
        self.vc = vc
        self.channel_id = channel_id
        self.respond_callback = respond_callback
        self.user_buffers: dict[int, UserAudioBuffer] = {}
        self.processing: set[int] = set()  # 처리 중인 유저 (중복 방지)

    def wants_opus(self) -> bool:
        return False  # PCM으로 받기

    def write(self, user: discord.User | None, data: voice_recv.VoiceData):
        if user is None or user.bot:
            return
        if user.id not in self.user_buffers:
            self.user_buffers[user.id] = UserAudioBuffer()

        buf = self.user_buffers[user.id]
        # data.pcm: 48kHz stereo PCM
        pcm = data.pcm
        # stereo → mono (좌 채널만 사용)
        arr = np.frombuffer(pcm, dtype=np.int16)
        mono = arr[::2].tobytes()

        result = buf.process_frame(mono)
        if result and user.id not in self.processing:
            asyncio.run_coroutine_threadsafe(
                self._handle_speech(user, result),
                self.bot.loop
            )

    async def _handle_speech(self, user: discord.User, audio_bytes: bytes):
        self.processing.add(user.id)
        t0 = time.time()
        try:
            # 봇이 말하는 중이면 무시
            if self.vc.is_playing():
                return

            text = await asyncio.get_event_loop().run_in_executor(
                None, transcribe_audio_sync, audio_bytes
            )
            logger.info(f"[STT] {user.name}: '{text}' ({time.time()-t0:.2f}s)")

            if not text or len(text.strip()) < 2:
                return

            await self.respond_callback(self.vc, text, self.channel_id)
            logger.info(f"[총 응답시간] {time.time()-t0:.2f}s")
        finally:
            self.processing.discard(user.id)

    def cleanup(self):
        pass


def transcribe_audio_sync(audio_bytes: bytes) -> str:
    from stt import transcribe_audio
    return transcribe_audio(audio_bytes)
