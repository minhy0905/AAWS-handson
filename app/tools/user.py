import json
import os
from typing import Any

from langchain.tools import tool


_USER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "user",
)
_USER_FILE = os.path.join(_USER_DIR, "user.json")


@tool(parse_docstring=True)
def getuserinfo() -> dict[str, Any]:
    """저장된 사용자 정보를 불러옵니다."""
    if not os.path.exists(_USER_FILE):
        return {
            "success": True,
            "data": None,
            "message": "저장된 사용자 정보가 없습니다.",
        }

    try:
        with open(_USER_FILE, "r", encoding="utf-8") as file:
            user_info = json.load(file)
        return {
            "success": True,
            "data": user_info,
            "message": "사용자 정보를 불러왔습니다.",
        }
    except (OSError, json.JSONDecodeError) as error:
        return {
            "success": False,
            "data": None,
            "error": f"사용자 정보를 불러오지 못했습니다: {error}",
        }


@tool(parse_docstring=True)
def saveuserinfo(user_info: dict[str, Any]) -> dict[str, Any]:
    """사용자 정보를 JSON 파일에 저장합니다.

    Args:
        user_info: 저장할 사용자 정보 객체. 이름, 이메일, 선호 설정 등 자유로운
            JSON 호환 필드를 포함할 수 있습니다.
    """
    try:
        os.makedirs(_USER_DIR, exist_ok=True)
        with open(_USER_FILE, "w", encoding="utf-8") as file:
            json.dump(user_info, file, ensure_ascii=False, indent=2)
        return {
            "success": True,
            "data": user_info,
            "message": "사용자 정보를 저장했습니다.",
        }
    except (OSError, TypeError, ValueError) as error:
        return {
            "success": False,
            "data": None,
            "error": f"사용자 정보를 저장하지 못했습니다: {error}",
        }
