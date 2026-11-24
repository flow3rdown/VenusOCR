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

"""VenusOCR command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any, Dict

from venusocr.utils.logging import configure_logging, get_logger
from venusocr.version import __version__

logger = get_logger(__name__)


def _auto_coerce(raw: str) -> Any:
    """Coerce a CLI string to a Python scalar."""
    lowered = raw.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def _collect_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI shortcut flags and ``--set`` pairs into config overrides."""
    overrides: Dict[str, Any] = {}
    if getattr(args, "server_url", None):
        overrides["engine.vl_rec_server_url"] = args.server_url
    if getattr(args, "api_model_name", None):
        overrides["engine.vl_rec_api_model_name"] = args.api_model_name
    if getattr(args, "layout_model_dir", None):
        overrides["engine.layout_model_dir"] = args.layout_model_dir
    if getattr(args, "layout_backend", None):
        overrides["engine.layout_backend"] = args.layout_backend
    if getattr(args, "device", None):
        overrides["engine.device"] = args.device
    if getattr(args, "max_concurrency", None) is not None:
        overrides["engine.vl_rec_max_concurrency"] = args.max_concurrency
    if getattr(args, "no_markdown", False):
        overrides["output.save_markdown"] = False
    if getattr(args, "no_layout_vis", False):
        overrides["output.save_layout_vis"] = False
    if getattr(args, "pretty", False):
        overrides["output.pretty_markdown"] = True
    for key, value in getattr(args, "config_overrides", None) or []:
        overrides[key] = _auto_coerce(value)
    return overrides


def _add_common_parse_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", type=str, default=None, help="YAML 配置文件路径")
    p.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="VenusOCR vLLM 服务地址，如 http://127.0.0.1:8000/v1",
    )
    p.add_argument(
        "--api-model-name",
        type=str,
        default=None,
        help="服务端模型名（默认自动从服务获取）",
    )
    p.add_argument(
        "--layout-backend",
        choices=["paddlex", "transformers"],
        default=None,
        help="版面检测后端；paddlex 用于严格对齐，transformers 用于减少依赖",
    )
    p.add_argument(
        "--layout-model-dir",
        type=str,
        default=None,
        help="PP-DocLayoutV3 本地模型目录（默认自动下载）",
    )
    p.add_argument("--device", type=str, default=None, help="版面检测设备，如 gpu:0 / cpu")
    p.add_argument("--max-concurrency", type=int, default=None, help="识别请求并发上限")
    p.add_argument(
        "--set",
        nargs=2,
        action="append",
        metavar=("KEY", "VALUE"),
        dest="config_overrides",
        help="覆盖任意配置项，如 --set predict.temperature 0.1",
    )


def _cmd_parse(args: argparse.Namespace) -> int:
    from venusocr.api import VenusOCR
    from venusocr.utils.fileio import collect_input_files

    overrides = _collect_overrides(args)
    paths, input_root = collect_input_files(args.input)

    with VenusOCR(config_path=args.config, overrides=overrides) as ocr:
        logger.info(
            "╭─ VenusOCR · 任务配置\n"
            "│ 输入       %s\n"
            "│ 文档       %d 个\n"
            "│ 输出       %s\n"
            "│ Layout     %s · %s\n"
            "╰────────────────────────────────────────",
            args.input,
            len(paths),
            args.output,
            ocr.config.engine.layout_backend,
            ocr.config.engine.device,
        )
        if input_root is not None:
            summary = ocr.parse_dir(args.input, args.output, num_workers=args.num_workers)
            return 1 if summary.crashed or summary.failed else 0

        result = ocr.parse(paths[0])
        result.save(args.output, ocr.config.output)
        logger.info("Saved results for: %s", result.input_path)
        if args.stdout:
            print(f"\n=== {result.stem} - Markdown ===")
            print(result.markdown)

    logger.info("结果已写入: %s", args.output)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from venusocr.server import build_vllm_command, launch_server

    cmd = build_vllm_command(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        served_model_name=args.served_model_name,
        max_num_batched_tokens=args.max_num_batched_tokens,
        extra_args=args.extra_args or None,
    )
    return launch_server(cmd)


def _cmd_demo(args: argparse.Namespace) -> int:
    try:
        import gradio  # noqa: F401
    except ImportError:
        logger.error(
            "gradio is not installed. Run: pip install 'venusocr[paddlex,demo]' "
            "or install the demo extra with your chosen layout backend"
        )
        return 1
    from venusocr.demo import build_demo

    overrides = _collect_overrides(args)
    demo = build_demo(config_path=args.config, overrides=overrides)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="venusocr",
        description="VenusOCR - 基于 VenusOCR 视觉语言模型的文档解析工具",
    )
    parser.add_argument("--version", "-V", action="version", version=f"venusocr {__version__}")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO）",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # --- parse ---
    parse_p = subparsers.add_parser(
        "parse",
        help="解析文档（单个 pdf/图片，或整个目录并发解析）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 解析单个 PDF
  venusocr parse paper.pdf -o ./output --server-url http://127.0.0.1:8000/v1

  # 并发解析整个目录（共享模型流水线，8 个批处理 Worker）
  venusocr parse ./docs/ -o ./output --num-workers 8

  # 覆盖采样参数
  venusocr parse img.png --set predict.temperature 0.1 --set predict.top_p 0.9
""",
    )
    parse_p.add_argument("input", type=str, help="输入文件（pdf/图片）或目录")
    parse_p.add_argument("--output", "-o", type=str, default="./output", help="输出目录")
    parse_p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="目录批量解析的页面批处理并发数（默认取配置 batch.num_workers）",
    )
    parse_p.add_argument("--stdout", action="store_true", help="同时将 markdown 打印到标准输出")
    parse_p.add_argument("--no-markdown", action="store_true", help="不保存 markdown 结果")
    parse_p.add_argument("--no-layout-vis", action="store_true", help="不保存版面画框 PDF")
    parse_p.add_argument("--pretty", action="store_true", help="美化 markdown 输出（速度略慢）")
    _add_common_parse_args(parse_p)

    # --- serve ---
    serve_p = subparsers.add_parser(
        "serve",
        help="使用 vLLM 启动 VenusOCR 识别服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  venusocr serve /path/to/venusocr-checkpoint --port 8000
  CUDA_VISIBLE_DEVICES=0 venusocr serve /path/to/ckpt -- --gpu-memory-utilization 0.9
""",
    )
    serve_p.add_argument("model_path", type=str, help="VenusOCR 模型目录")
    serve_p.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    serve_p.add_argument("--port", type=int, default=8000, help="监听端口")
    serve_p.add_argument(
        "--served-model-name", type=str, default="VenusOCR", help="对外暴露的模型名"
    )
    serve_p.add_argument(
        "--max-num-batched-tokens", type=int, default=16384, help="vLLM 批处理 token 上限"
    )

    # --- demo ---
    demo_p = subparsers.add_parser("demo", help="启动 Gradio 可视化 demo")
    demo_p.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    demo_p.add_argument("--port", type=int, default=7860, help="监听端口")
    demo_p.add_argument("--share", action="store_true", help="生成公网分享链接")
    _add_common_parse_args(demo_p)

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    cli_args = sys.argv[1:]
    extra_args = []
    if "--" in cli_args:
        separator_index = cli_args.index("--")
        extra_args = cli_args[separator_index + 1 :]
        cli_args = cli_args[:separator_index]
    args = parser.parse_args(cli_args)
    if extra_args and args.command != "serve":
        parser.error("Arguments after -- are only supported by the serve command")
    if args.command == "serve":
        args.extra_args = extra_args

    if not args.command:
        parser.print_help()
        sys.exit(1)

    configure_logging(level=args.log_level)

    handlers = {"parse": _cmd_parse, "serve": _cmd_serve, "demo": _cmd_demo}
    try:
        sys.exit(handlers[args.command](args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - top-level error reporting
        logger.error("Error: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
