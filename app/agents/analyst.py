from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from app.prompts import ANALYST_SYSTEM_PROMPT
from app.tools import tools_analyst


def create_analyst_agent(
    model_name: str = "google_genai:gemini-flash-latest",
    temperature: float = 0.2,
):
    """데이터 분석 및 시각화 전문 에이전트를 생성합니다."""
    model = init_chat_model(model_name, temperature=temperature)
    checkpointer = InMemorySaver()
    return create_agent(
        model=model,
        system_prompt=ANALYST_SYSTEM_PROMPT,
        tools=tools_analyst,
        checkpointer=checkpointer,
        name="analyst_agent",
    )
