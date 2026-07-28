import base64
import binascii
import os
import uuid
from typing import Any

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, Modality


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_IMAGE_DIR = os.path.join(_PROJECT_ROOT, "artifacts", "images")
_DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


def _unique_output_path(filename: str) -> str:
    """기존 이미지를 덮어쓰지 않는 고유한 출력 경로를 만듭니다."""
    candidate = os.path.abspath(os.path.join(_IMAGE_DIR, filename))
    if not os.path.exists(candidate):
        return candidate

    stem, extension = os.path.splitext(filename)
    return os.path.abspath(
        os.path.join(
            _IMAGE_DIR,
            f"{stem}_{uuid.uuid4().hex[:8]}{extension}",
        )
    )


def _extract_image_data(content: Any) -> tuple[bytes | None, str]:
    """Gemini/LangChain 응답 블록에서 이미지 bytes와 텍스트를 추출합니다."""
    if not isinstance(content, list):
        return None, content if isinstance(content, str) else ""

    response_texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            response_texts.append(str(block.get("text", "")))
            continue

        encoded_data = None
        if block_type == "image":
            encoded_data = block.get("data")
        elif block_type == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if isinstance(url, str) and url.startswith("data:") and "," in url:
                encoded_data = url.split(",", 1)[1]
        elif block_type == "media":
            encoded_data = block.get("data") or block.get("content")

        if isinstance(encoded_data, bytes):
            return encoded_data, "\n".join(response_texts)
        if isinstance(encoded_data, str) and encoded_data:
            try:
                return base64.b64decode(encoded_data, validate=True), "\n".join(
                    response_texts
                )
            except (binascii.Error, ValueError):
                continue

    return None, "\n".join(response_texts)


@tool(parse_docstring=True)
def generate_image_with_nano_banana(
    prompt: str,
    filename: str = "",
    model_name: str = _DEFAULT_MODEL,
) -> str:
    """Nano Banana(Gemini 이미지 모델)로 프롬프트에 맞는 이미지를 생성합니다.

    생성한 이미지는 artifacts/images 폴더에 PNG 파일로 저장됩니다.

    Args:
        prompt: 생성할 이미지의 모습, 스타일, 구도 등을 설명하는 프롬프트
        filename: 선택적인 출력 파일명. 비어 있으면 고유한 이름을 자동 생성합니다.
        model_name: 사용할 Gemini 이미지 생성 모델 ID
    """
    if not prompt.strip():
        return "[Error] 이미지 생성 프롬프트가 비어 있습니다."

    if not os.getenv("GOOGLE_API_KEY"):
        return "[Error] GOOGLE_API_KEY 환경 변수가 설정되어 있지 않습니다."

    safe_filename = os.path.basename(filename.strip()) if filename else ""
    if not safe_filename:
        safe_filename = f"nano_banana_{uuid.uuid4().hex[:12]}.png"
    elif not safe_filename.lower().endswith(".png"):
        safe_filename += ".png"

    os.makedirs(_IMAGE_DIR, exist_ok=True)
    output_path = _unique_output_path(safe_filename)

    try:
        image_model = ChatGoogleGenerativeAI(
            model=model_name,
            response_modalities=[Modality.IMAGE, Modality.TEXT],
        )
        response = image_model.invoke(prompt)
        image_data, response_text = _extract_image_data(response.content)

        if image_data is None:
            detail = f" 모델 응답: {response_text}" if response_text else ""
            return f"[Error] 모델 응답에서 이미지 데이터를 찾지 못했습니다.{detail}"

        with open(output_path, "wb") as image_file:
            image_file.write(image_data)

        return f"<Render_Image>{output_path}</Render_Image>"
    except Exception as error:
        return f"[Error] 이미지 생성에 실패했습니다: {error}"
