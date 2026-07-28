import json
import os
import uuid
from typing import Any

from langchain.tools import tool


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_ARTIFACTS_ROOT = os.path.join(_PROJECT_ROOT, "artifacts")
_ANALYSIS_ROOT = os.path.join(_ARTIFACTS_ROOT, "analysis")
_CHART_DIR = os.path.join(_ANALYSIS_ROOT, "charts")
_REPORT_DIR = os.path.join(_ANALYSIS_ROOT, "reports")


def _resolve_data_path(filepath: str) -> str:
    candidate = (
        os.path.abspath(filepath)
        if os.path.isabs(filepath)
        else os.path.abspath(os.path.join(_PROJECT_ROOT, filepath))
    )
    if os.path.commonpath([candidate, _ARTIFACTS_ROOT]) != os.path.abspath(
        _ARTIFACTS_ROOT
    ):
        raise ValueError("artifacts 폴더 내부의 데이터 파일만 읽을 수 있습니다.")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {candidate}")
    return candidate


def _load_dataframe(filepath: str):
    import pandas as pd

    resolved_path = _resolve_data_path(filepath)
    extension = os.path.splitext(resolved_path)[1].lower()
    if extension == ".json":
        with open(resolved_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
        if isinstance(raw_data, dict):
            for key in ("data", "items", "results", "records"):
                if isinstance(raw_data.get(key), list):
                    raw_data = raw_data[key]
                    break
        dataframe = pd.DataFrame(raw_data)
    elif extension == ".csv":
        dataframe = pd.read_csv(resolved_path)
    else:
        raise ValueError("JSON 또는 CSV 파일만 분석할 수 있습니다.")
    return resolved_path, dataframe


def _safe_output_path(directory: str, filename: str, extension: str) -> str:
    os.makedirs(directory, exist_ok=True)
    safe_name = os.path.basename(filename.strip()) if filename else ""
    if not safe_name:
        safe_name = f"analysis_{uuid.uuid4().hex[:12]}{extension}"
    elif not safe_name.lower().endswith(extension):
        safe_name += extension

    candidate = os.path.abspath(os.path.join(directory, safe_name))
    if not os.path.exists(candidate):
        return candidate
    stem, suffix = os.path.splitext(safe_name)
    return os.path.abspath(
        os.path.join(directory, f"{stem}_{uuid.uuid4().hex[:8]}{suffix}")
    )


@tool(parse_docstring=True)
def load_json_data(filepath: str) -> dict[str, Any]:
    """JSON 또는 CSV 데이터를 읽어 분석 가능한 프로파일을 반환합니다.

    Args:
        filepath: 프로젝트 artifacts 폴더 내부의 JSON 또는 CSV 파일 경로
    """
    try:
        resolved_path, dataframe = _load_dataframe(filepath)
        numeric = dataframe.select_dtypes(include="number")
        categorical = dataframe.select_dtypes(exclude="number")

        top_values = {}
        for column in categorical.columns:
            counts = dataframe[column].astype(str).value_counts(dropna=False).head(10)
            top_values[str(column)] = {
                str(key): int(value) for key, value in counts.items()
            }

        numeric_summary = {}
        if not numeric.empty:
            numeric_summary = json.loads(
                numeric.describe().round(4).to_json(force_ascii=False)
            )

        return {
            "success": True,
            "data": {
                "filepath": resolved_path,
                "row_count": int(len(dataframe)),
                "column_count": int(len(dataframe.columns)),
                "columns": [str(column) for column in dataframe.columns],
                "dtypes": {
                    str(column): str(dtype)
                    for column, dtype in dataframe.dtypes.items()
                },
                "missing_values": {
                    str(column): int(value)
                    for column, value in dataframe.isna().sum().items()
                },
                "numeric_summary": numeric_summary,
                "top_values": top_values,
                "sample": json.loads(
                    dataframe.head(5).to_json(
                        orient="records",
                        force_ascii=False,
                        date_format="iso",
                    )
                ),
            },
        }
    except Exception as error:
        return {
            "success": False,
            "data": None,
            "error": f"데이터를 불러오지 못했습니다: {error}",
        }


@tool(parse_docstring=True)
def create_chart(
    filepath: str,
    chart_type: str,
    x_field: str,
    y_field: str = "",
    title: str = "",
    filename: str = "",
) -> str:
    """수집 데이터로 차트를 생성하고 렌더링 태그를 반환합니다.

    Args:
        filepath: 프로젝트 artifacts 폴더 내부의 JSON 또는 CSV 파일 경로
        chart_type: count, bar, line, scatter, hist, pie 중 하나
        x_field: X축 또는 범주로 사용할 필드
        y_field: Y축으로 사용할 숫자 필드. count, hist, pie에서는 생략 가능
        title: 차트 제목
        filename: 저장할 PNG 파일명. 비어 있으면 자동 생성
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, dataframe = _load_dataframe(filepath)
        if dataframe.empty:
            raise ValueError("차트를 생성할 데이터가 없습니다.")
        if x_field not in dataframe.columns:
            raise ValueError(f"필드를 찾을 수 없습니다: {x_field}")
        if y_field and y_field not in dataframe.columns:
            raise ValueError(f"필드를 찾을 수 없습니다: {y_field}")

        chart_type = chart_type.lower().strip()
        figure, axis = plt.subplots(figsize=(10, 6))

        if chart_type in ("count", "bar") and not y_field:
            series = dataframe[x_field].astype(str).value_counts().head(30)
            series.plot(kind="bar", ax=axis)
            axis.set_ylabel("Count")
        elif chart_type == "bar":
            grouped = dataframe.groupby(x_field, dropna=False)[y_field].mean()
            grouped.head(30).plot(kind="bar", ax=axis)
            axis.set_ylabel(y_field)
        elif chart_type == "line":
            if not y_field:
                raise ValueError("line 차트에는 y_field가 필요합니다.")
            dataframe.plot(x=x_field, y=y_field, kind="line", marker="o", ax=axis)
        elif chart_type == "scatter":
            if not y_field:
                raise ValueError("scatter 차트에는 y_field가 필요합니다.")
            dataframe.plot(x=x_field, y=y_field, kind="scatter", ax=axis)
        elif chart_type == "hist":
            dataframe[x_field].plot(kind="hist", bins=20, ax=axis)
            axis.set_xlabel(x_field)
        elif chart_type == "pie":
            series = dataframe[x_field].astype(str).value_counts().head(12)
            series.plot(kind="pie", autopct="%1.1f%%", ax=axis)
            axis.set_ylabel("")
        else:
            raise ValueError(
                "chart_type은 count, bar, line, scatter, hist, pie 중 하나여야 합니다."
            )

        axis.set_title(title or f"{chart_type.title()}: {x_field}")
        figure.tight_layout()
        output_path = _safe_output_path(_CHART_DIR, filename, ".png")
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return f"<Render_Image>{output_path}</Render_Image>"
    except Exception as error:
        return f"[Error] 차트를 생성하지 못했습니다: {error}"


@tool(parse_docstring=True)
def write_report(content: str, filename: str = "") -> dict[str, Any]:
    """분석 결과를 Markdown 리포트로 저장합니다.

    Args:
        content: 인사이트, 근거 수치, 차트 설명을 포함한 Markdown 본문
        filename: 저장할 Markdown 파일명. 비어 있으면 자동 생성
    """
    try:
        output_path = _safe_output_path(_REPORT_DIR, filename, ".md")
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(content)
        return {
            "success": True,
            "data": {"report_path": output_path},
            "message": "분석 리포트를 저장했습니다.",
        }
    except Exception as error:
        return {
            "success": False,
            "data": None,
            "error": f"분석 리포트를 저장하지 못했습니다: {error}",
        }
