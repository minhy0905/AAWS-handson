from .common import ARTIFACT_DIR
from .navigator import get_page_structure, verify_selectors_with_samples, browse_web
from .coder import read_code_file, edit_code_file, create_new_file, write_text_file, run_python_script, validate_collected_data
from .supervisor_tools import read_image_and_analyze, web_search_custom_tool
from .user import getuserinfo, saveuserinfo
from .nano_banana import generate_image_with_nano_banana

# 🏭 도구 팩토리 그룹화 (Tool Factory Groups)
tools_navigator = [get_page_structure, verify_selectors_with_samples, browse_web]
tools_coder = [read_code_file, edit_code_file, create_new_file, write_text_file, run_python_script, validate_collected_data]
tools_supervisor = [
    read_image_and_analyze,
    web_search_custom_tool,
    generate_image_with_nano_banana,
]
tools_user = [getuserinfo, saveuserinfo]
tools_image = [generate_image_with_nano_banana]

__all__ = [
    "ARTIFACT_DIR",
    "tools_navigator", "tools_coder", "tools_supervisor", "tools_user", "tools_image",
    "get_page_structure", "verify_selectors_with_samples", "browse_web",
    "read_code_file", "edit_code_file", "create_new_file", "write_text_file", "run_python_script", "validate_collected_data",
    "read_image_and_analyze", "web_search_custom_tool",
    "getuserinfo", "saveuserinfo",
    "generate_image_with_nano_banana",
]
