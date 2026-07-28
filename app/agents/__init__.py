from .navigator import create_navigator_agent
from .coder import create_coder_agent
from .analyst import create_analyst_agent
from .utils import dynamic_response_format

# 🏭 에이전트 레지스트리 (Agent Registry)
# 새 에이전트 추가 시 여기에 정보 한 줄만 작성하면 서버, UI, CLI에 모두 반영됩니다.
AGENT_REGISTRY = [
    {
        "name": "supervisor",
        "module": "app.agents.supervisor",
        "prefix": "/supervisor",
        "tags": ["Supervisor"],
        "description": "전체 수집 프로세스를 조율하고 네비게이터와 코더를 지휘하는 감독 에이전트"
    },
    {
        "name": "navigator",
        "module": "app.agents.navigator",
        "prefix": "/navigator",
        "tags": ["Navigator"],
        "description": "웹페이지 구조를 분석하여 최적의 수집 설계도(Blueprint)를 제작하는 네비게이터 에이전트"
    },
    {
        "name": "coder",
        "module": "app.agents.coder",
        "prefix": "/coder",
        "tags": ["Coder"],
        "description": "수집 설계도를 바탕으로 수집 코드를 작성하고 자가 수정(Self-Healing)을 수행하는 코더 에이전트"
    },
    {
        "name": "analyst",
        "module": "app.agents.analyst",
        "prefix": "/analyst",
        "tags": ["Analyst"],
        "description": "수집된 JSON/CSV를 분석하여 인사이트, 차트, Markdown 리포트를 만드는 분석 에이전트"
    },
    {
        "name": "chatbot",
        "module": "app.agents.chatbot",
        "prefix": "/chatbot",
        "tags": ["Chatbot"],
        "description": "등록된 공용 도구를 활용해 다양한 요청을 처리하는 범용 AI 어시스턴트"
    }
]

__all__ = [
    "AGENT_REGISTRY",
    "create_navigator_agent",
    "create_coder_agent",
    "create_analyst_agent",
    "dynamic_response_format"
]
