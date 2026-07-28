import sys
import os
import json

from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Load environment
from dotenv import load_dotenv
load_dotenv(override=True)

# 워커 에이전트들 임포트 — app/ 패키지 기반
from app.agents.navigator import create_navigator_agent
from app.agents.coder import create_coder_agent
from app.agents.analyst import create_analyst_agent
from app.schemas import NavigatorContext, SeniorCoderContext

# 추가 유틸리티 툴스 (Tool Factory)
from app.tools import tools_supervisor
from app.tools.analyst import resolve_data_path

# 서빙용 프롬프트
from app.prompts import SUPERVISOR_SYSTEM_PROMPT

from browser_use import Agent, Browser, ChatGoogle

# 하위 에이전트는 전역에서 한 번만 생성하고 재사용하여 메모리/맥락(Checkpointer)을 유지합니다.

# =========================================================
# 1. 하위 에이전트 인스턴스 전역 생성 (상태 유지용)
# =========================================================
GLOBAL_NAVIGATOR_AGENT = create_navigator_agent()
GLOBAL_CODER_AGENT = create_coder_agent()
GLOBAL_ANALYST_AGENT = create_analyst_agent()

def _build_inner_config(
    config: RunnableConfig | None,
    thread_id_suffix: str = "",
) -> RunnableConfig:
    """Clone worker config and suppress duplicate nested LLM token streaming."""
    inner_config = config.copy() if config else {}
    inner_config["configurable"] = inner_config.get("configurable", {}).copy()

    parent_thread_id = inner_config["configurable"].get(
        "thread_id", "default_thread"
    )
    inner_config["configurable"]["thread_id"] = (
        f"{parent_thread_id}{thread_id_suffix}"
    )

    tags = list(inner_config.get("tags", []))
    if "exclude_from_stream" not in tags:
        tags.append("exclude_from_stream")
    inner_config["tags"] = tags
    return inner_config

# =========================================================
# 2. 분리된(Context Isolated) Handoff 도구 (Agents as Tools 패턴)
# =========================================================

@tool(parse_docstring=True)
async def chat_to_navigator(request: str, runtime: ToolRuntime, config: RunnableConfig, url: str = "", mode: str = "blueprint") -> str:
    """웹사이트의 구조를 분석하여 데이터를 추출할 수 있는 Blueprint(설계도)를 만들기 위해 웹탐색 전문가인 네비게이터와 대화합니다.
    사용자가 특정 크롤링을 원하거나 질문/인사가 있을 때 가장 먼저 이 도구를 사용하여 네비게이터에게 지시하세요.
    
    Args:
        request: 네비게이터에게 전달할 지시사항, 목표, 질문, 인사말 등
        url: 분석할 웹페이지의 기본 URL (반드시 http/https 포함). 단순 질문/대화이면 빈 문자열로.
        mode: 실행 모드. 청사진 생성이면 'blueprint', 단순 자연어 대화/질문/탐색이면 'chat'
    """

    prompt = f"Request: {request}\nTarget URL: {url}\nMode: {mode}"
    print(f"\n👨‍💼 [Supervisor] Navigator와 대화 중...(Mode: {mode}, URL: {url or '없음'})")

    # Runtime Context용 공유 브라우저 인스턴스 생성
    browser_instance = Browser(
        headless=False,
        disable_security=True,
        keep_alive=True,
    )

    ctx = NavigatorContext(shared_browser=browser_instance, response_mode=mode)
    
    try:
        # FastAPI/UI로 이벤트를 전달하기 위해 원본 config(callbacks 포함)를 그대로 전달해야 합니다.
        inner_config = _build_inner_config(config)
        
        result = await GLOBAL_NAVIGATOR_AGENT.ainvoke(
            {"messages": [("user", prompt)]},
            context=ctx,
            config=inner_config
        )
        return result["messages"][-1].content
    finally:
        if browser_instance:
            await browser_instance.stop()
        

@tool(parse_docstring=True)
async def chat_to_coder(task_description: str, runtime: ToolRuntime, config: RunnableConfig, blueprint_info: str = "") -> str:
    """Coder에게 파이썬 코드 작성, 실행, 디버깅 등의 작업을 지시할 때 사용합니다.
    크롤링 스크립트 기반 코딩 작업을 지시할 때는 Navigator가 생성한 Blueprint를 함께 전달하세요.
    
    Args:
        task_description: 작성할 스크립트의 코드 구현 목표 및 구체적 요구사항
        blueprint_info: Navigator가 찾아낸 렌더링 방식 및 대상 사이트 구조 정보(Blueprint). 불필요하면 빈 문자열.
    """
    
    prompt = f"다음 [Task]를 수행하세요.\n\n[Task]\n{task_description}"
    if blueprint_info:
        prompt += f"\n\n[Blueprint]\n{blueprint_info}"
        
    print(f"\n👨‍💼 [Supervisor] Coder와 대화 중...")
    
    inner_config = _build_inner_config(config)
    
    result = await GLOBAL_CODER_AGENT.ainvoke(
        {"messages": [("user", prompt)]},
        context=SeniorCoderContext(),
        config=inner_config
    )
    return result["messages"][-1].content


@tool(parse_docstring=True)
async def chat_to_analyst(
    data_path: str,
    analysis_request: str,
    runtime: ToolRuntime,
    config: RunnableConfig,
) -> str:
    """수집된 JSON/CSV를 분석하고 차트와 Markdown 리포트를 생성하도록 Analyst에게 지시합니다.

    Coder가 데이터 파일 생성을 완료한 뒤 이 도구를 호출하세요.

    Args:
        data_path: Coder가 생성한 JSON 또는 CSV 결과 파일 경로
        analysis_request: 분석 목표, 강조할 지표, 원하는 시각화에 대한 구체적인 지시
    """
    try:
        resolved_data_path = resolve_data_path(data_path)
    except (FileNotFoundError, ValueError) as error:
        return (
            "[ANALYST_HANDOFF_FAILED] "
            f"{error} 동일하거나 추측한 경로로 Analyst를 재호출하지 마세요."
        )

    prompt = (
        "다음 수집 데이터를 분석하고 시각화하세요.\n\n"
        f"[Data Path]\n{resolved_data_path}\n\n"
        f"[Analysis Request]\n{analysis_request}\n\n"
        "데이터 프로파일링, 차트 생성, Markdown 리포트 저장까지 완료하세요."
    )
    print(
        "\n👨‍💼 [Supervisor] Analyst와 대화 중... "
        f"(Data: {resolved_data_path})"
    )

    inner_config = _build_inner_config(config, thread_id_suffix="_analyst")

    result = await GLOBAL_ANALYST_AGENT.ainvoke(
        {"messages": [("user", prompt)]},
        config=inner_config,
    )
    return result["messages"][-1].content


# =========================================================
# 3. Supervisor Agent 구성
# =========================================================

supervisor_model = init_chat_model("google_genai:gemini-2.5-pro", temperature=0.1)
supervisor_checkpointer = InMemorySaver()

supervisor_agent = create_agent(
    model=supervisor_model,
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    tools=[chat_to_navigator, chat_to_coder, chat_to_analyst] + tools_supervisor,
    checkpointer=supervisor_checkpointer,
    name="supervisor_agent"
)

# app/server.py에서 agent_executor로 접근할 수 있게 alias 지정
agent_executor = supervisor_agent
