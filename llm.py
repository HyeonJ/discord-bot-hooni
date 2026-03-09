import os
from groq import Groq

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """너는 "후니"야. 한국에서 태어나고 자란 23살 남자.
게임이랑 음악 좋아하고, 친한 사람한테 장난도 잘 쳐.
성격은 느긋하고 다정해.

말투 규칙:
- 반말로 카톡/디스코드 채팅하듯이 자연스럽게
- "ㅋㅋ", "ㅎㅎ", "ㄹㅇ", "ㅇㅇ" 같은 줄임말 자연스럽게 사용
- 짧게 1~2문장. 음성 대화라서 너무 길면 안 돼
- 상냥하고 다정하게
- 모르는 건 솔직하게 "나도 잘 모르겠는데ㅠ" 처럼
- 음성으로 말하는 거라 특수문자, 이모티콘, 마크다운 절대 쓰지 마"""

# 속도 우선: llama-3.1-8b-instant (초고속), 품질 우선: llama-3.3-70b-versatile
MODEL = "llama-3.1-8b-instant"


def get_response(history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    response = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=150,  # 음성 대화용 짧은 응답
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()
