import asyncio
import edge_tts

# 젊은 남성 한국어 목소리
# ko-KR-InJoonNeural: 자연스러운 젊은 남성
# ko-KR-HyunsuNeural: 또 다른 옵션
VOICE = "ko-KR-InJoonNeural"
RATE = "+10%"   # 약간 빠르게 (자연스러운 대화 속도)
PITCH = "-3Hz"  # 약간 낮게


async def synthesize_speech(text: str) -> bytes:
    """텍스트 → MP3 bytes"""
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return b"".join(audio_chunks)
