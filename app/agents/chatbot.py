from datetime import date
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from app.tools import tools_all

# 오늘 날짜
today_date = date.today().strftime("%Y-%m-%d")

# System Prompt
system_prompt = f"""
당신은 친절한 AI 어시스턴트입니다.
사용자의 질문에 대해 명확하고 도움이 되는 답변을 제공하세요.

사용 가능한 도구를 적극적으로 활용하세요.
- 사용자의 이름, 나이, 직업 등 사용자 정보에 관한 질문에는 추측하지 말고 반드시 getuserinfo 도구를 사용하세요.
- 사용자가 자신의 정보를 저장하거나 변경해달라고 하면 saveuserinfo 도구를 사용하세요.
- 이미지 생성을 요청하면 generate_image_with_nano_banana 도구를 사용하세요.
- 웹 탐색, 코드 작성·실행, 파일 검증이 필요한 요청에는 해당 전문 도구를 사용하세요.

오늘의 날짜 : {today_date}
"""

def get_agent_executor():
    llm = init_chat_model(model="gpt-4o", model_provider="openai")
    memory = MemorySaver()
    
    # 도구가 없는 순수 LLM 챗봇
    basic_agent = create_agent(
        model=llm,
        tools=tools_all,
        system_prompt=system_prompt,
        checkpointer=memory
    )
    return basic_agent

agent_executor = get_agent_executor()
