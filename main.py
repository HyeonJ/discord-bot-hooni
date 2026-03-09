import discord
import asyncio
import os
import io
import time
import logging
from discord.ext import commands
from discord import app_commands
import discord.ext.voice_recv as voice_recv

from stt import transcribe_audio
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

# 채널별 대화 히스토리 (채널ID → 메시지 리스트)
conversation_history: dict[int, list[dict]] = {}


async def respond_in_voice(vc: discord.VoiceClient, text: str, channel_id: int):
    """LLM 응답 생성 + TTS 재생"""
    t0 = time.time()

    history = conversation_history.setdefault(channel_id, [])
    history.append({"role": "user", "content": text})
    if len(history) > 20:
        history = history[-20:]
        conversation_history[channel_id] = history

    llm_text = await asyncio.get_event_loop().run_in_executor(None, get_response, history)
    logger.info(f"[LLM] {time.time()-t0:.2f}s → {llm_text[:60]}")

    history.append({"role": "assistant", "content": llm_text})

    audio_bytes = await synthesize_speech(llm_text)
    logger.info(f"[TTS] {time.time()-t0:.2f}s total")

    # Discord에 오디오 재생
    audio_source = discord.FFmpegPCMAudio(
        io.BytesIO(audio_bytes),
        pipe=True,
        options="-vn"
    )
    if vc.is_playing():
        vc.stop()
    vc.play(audio_source)


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    content = message.content.strip()
    if content == "후니 들어와":
        if not message.author.voice:
            await message.channel.send("먼저 음성 채널에 들어와줘!")
            return
        channel = message.author.voice.channel
        if message.guild.voice_client:
            await message.guild.voice_client.disconnect()
        vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
        sink = AudioSink(
            bot=bot,
            vc=vc,
            channel_id=message.channel.id,
            respond_callback=respond_in_voice,
        )
        vc.listen(sink)
        await message.channel.send("ㅇㅇ 들어왔어! 말 걸어봐")
    elif content == "후니 나가":
        if message.guild.voice_client:
            await message.guild.voice_client.disconnect()
            conversation_history.pop(message.channel.id, None)
            await message.channel.send("ㅇㅋ 나갈게")
    await bot.process_commands(message)


@tree.command(name="join", description="후니를 음성 채널에 불러오기")
async def join(interaction: discord.Interaction):
    if not interaction.user.voice:
        await interaction.response.send_message("먼저 음성 채널에 들어와줘!", ephemeral=True)
        return

    channel = interaction.user.voice.channel
    if interaction.guild.voice_client:
        await interaction.guild.voice_client.disconnect()

    vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
    logger.info(f"Joined voice channel: {channel.name}")

    sink = AudioSink(
        bot=bot,
        vc=vc,
        channel_id=interaction.channel_id,
        respond_callback=respond_in_voice,
    )
    vc.listen(sink)

    await interaction.response.send_message(f"ㅇㅇ 들어왔어! 말 걸어봐")


@tree.command(name="leave", description="후니 음성 채널에서 내보내기")
async def leave(interaction: discord.Interaction):
    if interaction.guild.voice_client:
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
