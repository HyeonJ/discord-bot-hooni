import discord
import asyncio
import os
import io
import re
import time
import logging
from discord.ext import commands
import discord.ext.voice_recv as voice_recv

from llm import get_response
from tts import synthesize_speech
from voice_handler import AudioSink

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

MAX_HISTORY = 20

# 채널별 대화 히스토리 (채널ID → 메시지 리스트)
conversation_history: dict[int, list[dict]] = {}
# 채널별 진행 중인 응답 태스크 (barge-in 시 취소용)
current_response: dict[int, asyncio.Task] = {}

# 문장 단위 분할 (마침표/물음표/느낌표/줄바꿈 기준). 구두점이 없으면 통째로 1문장.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？\n])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or [text.strip()]


async def _play_audio(vc: discord.VoiceClient, audio_bytes: bytes):
    """오디오 한 덩어리를 재생하고 끝날 때까지 대기. 취소되면 즉시 재생 중단."""
    source = discord.FFmpegPCMAudio(io.BytesIO(audio_bytes), pipe=True, options="-vn")
    loop = asyncio.get_running_loop()
    finished = asyncio.Event()

    def _after(err):
        if err:
            logger.warning(f"재생 오류: {err}")
        loop.call_soon_threadsafe(finished.set)

    if vc.is_playing():
        vc.stop()
    vc.play(source, after=_after)
    try:
        await finished.wait()
    except asyncio.CancelledError:
        if vc.is_connected():
            vc.stop()
        raise


async def _generate_and_speak(vc: discord.VoiceClient, channel_id: int):
    """LLM 응답 생성 + 문장 단위 스트리밍 TTS 재생 (취소 가능)."""
    t0 = time.time()
    history = conversation_history.setdefault(channel_id, [])

    llm_text = await asyncio.get_running_loop().run_in_executor(
        None, get_response, list(history)
    )
    logger.info(f"[LLM] {time.time()-t0:.2f}s → {llm_text[:60]}")
    history.append({"role": "assistant", "content": llm_text})

    sentences = _split_sentences(llm_text)
    # 현재 문장을 재생하는 동안 다음 문장 TTS를 미리 합성 (지연 최소화)
    next_audio = asyncio.create_task(synthesize_speech(sentences[0]))
    try:
        for i, _ in enumerate(sentences):
            audio = await next_audio
            if i + 1 < len(sentences):
                next_audio = asyncio.create_task(synthesize_speech(sentences[i + 1]))
            await _play_audio(vc, audio)
    except asyncio.CancelledError:
        next_audio.cancel()
        logger.info("[barge-in] 응답 중단됨")
        raise
    finally:
        if not next_audio.done():
            next_audio.cancel()
    logger.info(f"[응답완료] {time.time()-t0:.2f}s total")


async def respond_in_voice(vc: discord.VoiceClient, text: str, channel_id: int):
    """새 발화 수신 → 진행 중인 응답을 중단(barge-in)하고 새로 응답."""
    prev = current_response.get(channel_id)
    if prev and not prev.done():
        prev.cancel()
        try:
            await prev
        except asyncio.CancelledError:
            pass

    history = conversation_history.setdefault(channel_id, [])
    history.append({"role": "user", "content": text})
    if len(history) > MAX_HISTORY:
        del history[: len(history) - MAX_HISTORY]  # 리스트 객체 유지하며 in-place 정리

    current_response[channel_id] = asyncio.create_task(
        _generate_and_speak(vc, channel_id)
    )


async def _cancel_response(channel_id: int):
    task = current_response.pop(channel_id, None)
    if task and not task.done():
        task.cancel()


async def _connect_and_listen(channel: discord.VoiceChannel, text_channel_id: int) -> discord.VoiceClient:
    if channel.guild.voice_client:
        await channel.guild.voice_client.disconnect()
    vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
    sink = AudioSink(
        bot=bot,
        vc=vc,
        channel_id=text_channel_id,
        respond_callback=respond_in_voice,
    )
    vc.listen(sink)
    logger.info(f"음성 채널 입장: {channel.name}")
    return vc


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or message.guild is None:
        await bot.process_commands(message)
        return

    content = message.content.strip()
    if content == "후니 들어와":
        if not message.author.voice:
            await message.channel.send("먼저 음성 채널에 들어와줘!")
        else:
            await _connect_and_listen(message.author.voice.channel, message.channel.id)
            await message.channel.send("ㅇㅇ 들어왔어! 말 걸어봐")
    elif content == "후니 나가":
        if message.guild.voice_client:
            await _cancel_response(message.channel.id)
            await message.guild.voice_client.disconnect()
            conversation_history.pop(message.channel.id, None)
            await message.channel.send("ㅇㅋ 나갈게")

    await bot.process_commands(message)


@tree.command(name="join", description="후니를 음성 채널에 불러오기")
async def join(interaction: discord.Interaction):
    if not interaction.user.voice:
        await interaction.response.send_message("먼저 음성 채널에 들어와줘!", ephemeral=True)
        return
    await _connect_and_listen(interaction.user.voice.channel, interaction.channel_id)
    await interaction.response.send_message("ㅇㅇ 들어왔어! 말 걸어봐")


@tree.command(name="leave", description="후니 음성 채널에서 내보내기")
async def leave(interaction: discord.Interaction):
    if interaction.guild.voice_client:
        await _cancel_response(interaction.channel_id)
        await interaction.guild.voice_client.disconnect()
        conversation_history.pop(interaction.channel_id, None)
        await interaction.response.send_message("ㅇㅋ 나갈게")
    else:
        await interaction.response.send_message("나 원래 없었는데ㅋㅋ", ephemeral=True)


@bot.event
async def on_ready():
    await tree.sync()
    logger.info(f"후니 봇 준비 완료: {bot.user}")


bot.run(DISCORD_TOKEN)
