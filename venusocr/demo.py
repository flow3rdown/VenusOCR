# Copyright (c) 2026 VenusOCR. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradio demo for interactive VenusOCR document parsing."""

from __future__ import annotations

from typing import Any, Dict, Optional

from venusocr.utils.logging import get_logger

logger = get_logger(__name__)


def build_demo(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    """Build the Gradio Blocks app.

    Args:
        config_path: Optional YAML config path.
        overrides: Optional dotted-path config overrides.

    Returns:
        A ``gradio.Blocks`` instance (call ``.launch()`` to serve).
    """
    import gradio as gr

    from venusocr.api import VenusOCR

    ocr = VenusOCR(config_path=config_path, overrides=overrides)

    def run(file_obj):
        if file_obj is None:
            return "请先上传文件", None, None
        input_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        logger.info("Demo parsing: %s", input_path)
        try:
            result = ocr.parse(input_path)
        except Exception as e:  # noqa: BLE001 - surface errors in the UI
            logger.exception("Demo parsing failed")
            return f"解析失败: {e!r}", None, None

        json_res = None
        vis_images = []
        try:
            if result.page_results:
                json_res = result.page_results[0].json
        except Exception:  # noqa: BLE001 - JSON 仅用于展示
            json_res = None

        try:
            from venusocr.results.layout_pdf import render_layout_images

            vis_images = render_layout_images(result.input_path, result.page_results)
        except Exception:  # noqa: BLE001 - 可视化失败不影响主结果
            logger.exception("Failed to render layout visualization")

        return result.markdown, json_res, vis_images

    with gr.Blocks(title="VenusOCR Demo") as demo:
        gr.Markdown(
            "# VenusOCR 文档解析 Demo\n\n"
            "上传 PDF 或图片，输出 Markdown 解析结果、中间 JSON 与版面检测可视化。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="上传文档（PDF / 图片）",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
                )
                run_btn = gr.Button("开始解析", variant="primary")
            with gr.Column(scale=2):
                md_output = gr.Markdown(label="Markdown 结果")
        with gr.Row():
            with gr.Column():
                vis_output = gr.Gallery(label="版面检测可视化", columns=2, height=400)
            with gr.Column():
                json_output = gr.JSON(label="中间 JSON 结果（第一页）")

        run_btn.click(
            fn=run,
            inputs=[file_input],
            outputs=[md_output, json_output, vis_output],
        )

    return demo


def main() -> None:
    """Launch the demo with default settings."""
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
