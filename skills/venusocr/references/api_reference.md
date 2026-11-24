# Python API 与结果读取

在需要逐页 JSON、坐标引用、批量程序控制或单独区域识别时读取。执行前仍需完成 `SKILL.md` 的环境、权限、模型和服务检查；这些示例不是无需环境准备即可运行的离线实现。

## 1. 选择正确入口

| 入口 | 返回值 / 行为 |
|---|---|
| `VenusOCR(config_path=..., overrides=...)` | 延迟初始化的解析器；第一次 `parse()` 或访问 `.pipeline` 才创建模型/客户端 |
| `ocr.parse(str_path)` | 单文档 `DocumentResult`，包含全部页面；不会自动保存 |
| `ocr.parse_files(paths, output_dir=None, input_root=None)` | 顺序迭代；有输出根目录时自动保存，捕获解析异常并跳过该文件；保存异常仍可能抛出 |
| `ocr.parse_dir(input_dir, output_dir, num_workers=...)` | `BatchSummary`；不会返回每个文档的 `DocumentResult`，也不自动导出页 JSON |
| `venusocr.parse(input_path, output_dir=None, ...)` | 文件返回一个 `DocumentResult`，目录返回顺序处理后的结果列表；不是并发批量入口 |
| `result.save(output_dir, output_config)` | 创建 `<output_dir>/<stem>/`，保存整文档 Markdown、引用图、可选版面 PDF |
| `result.page_results` | `PageResult` 列表；不要虚构 `result.pages` 或文档级 `result.json` |

使用 `with VenusOCR(...) as ocr:` 管理资源；多文件尽量复用同一个实例，不自行用多个外部线程并发调用同一个 `ocr.parse()`。并发需求使用项目提供的 `parse_dir()`，不要额外创建大量模型进程。

向公开单文档 API 传入绝对路径字符串。即使底层采样器接受某些数组、目录或 URL，也不要混入主工作流：这些输入可能不符合文档命名、文件保存和可视化阶段的假设。Python `Path` 对象先转为 `str`。

## 2. 单文档保存 Markdown、逐页 JSON 与逐页 Markdown

先提供经检查的任务配置。下面的环境变量由示例代码显式读取，**不是 VenusOCR 自动识别的通用环境变量**。`VENUSOCR_JOB_OUTPUT` 指向尚不存在的运行目录，避免覆盖旧产物。

```bash
export VENUSOCR_JOB_INPUT='/absolute/input/report.pdf'
export VENUSOCR_JOB_OUTPUT='/absolute/output/api-run-001'
export VENUSOCR_JOB_CONFIG='/absolute/job/venusocr.yaml'
```

在已激活的目标 Python 环境使用以下代码；只对受支持的本地单文件执行：

```python
import json
import os
from pathlib import Path

from venusocr import VenusOCR

source = Path(os.environ["VENUSOCR_JOB_INPUT"]).resolve(strict=True)
config_path = Path(os.environ["VENUSOCR_JOB_CONFIG"]).resolve(strict=True)
output = Path(os.environ["VENUSOCR_JOB_OUTPUT"]).resolve()
if not source.is_file() or not config_path.is_file():
    raise ValueError("Input and config must be files")
if source.suffix.lower() not in {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
    raise ValueError("Unsupported input suffix")
output.mkdir(parents=True, exist_ok=False)

overrides = {"output.save_markdown": True}
api_key = os.environ.get("VENUSOCR_API_KEY")
if api_key:
    overrides["engine.vl_rec_api_key"] = api_key

with VenusOCR(config_path=str(config_path), overrides=overrides) as ocr:
    result = ocr.parse(str(source))
    if not result.page_results:
        raise RuntimeError("No pages returned; inspect the source and parsing logs")
    if source.suffix.lower() == ".pdf":
        expected = result.page_results[0]["page_count"]
        indexes = [page["page_index"] for page in result.page_results]
        if expected != len(indexes) or indexes != list(range(expected)):
            raise RuntimeError("Incomplete or unordered PDF page results")

    result.save(str(output), ocr.config.output)
    doc_dir = output / result.stem
    for position, page in enumerate(result.page_results, start=1):
        page.save_to_json(str(doc_dir / f"{result.stem}.page_{position:06d}.json"))
        page.save_to_markdown(
            str(doc_dir / f"{result.stem}.page_{position:06d}.md"),
            pretty=ocr.config.output.pretty_markdown,
        )
    (doc_dir / "metrics.json").write_text(
        json.dumps(result.metrics.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path = doc_dir / f"{result.stem}.md"
    if not markdown_path.is_file():
        raise RuntimeError("Expected Markdown was not saved")
    if ocr.config.output.save_layout_vis:
        if not (doc_dir / f"{result.stem}_layout.pdf").is_file():
            raise RuntimeError("Parsing succeeded, but the layout PDF was not saved")
    print(doc_dir)
```

说明与验收：

- `report.page_000001.json`、`report.page_000001.md`、`metrics.json` 是示例显式保存的文件，不是 CLI 默认输出。逐页文件使用 `<stem>.page_<序号>` 命名，避免与整文档 `<stem>.md` 冲突；按需求可省略逐页 Markdown 或 metrics 导出。
- 先 `result.save()` 再导出逐页 Markdown：整文档保存会为图片路径加页面前缀，避免多页中相同坐标的图片文件互相覆盖。逐页文件保持在同一文档目录，才能正确引用同一个 `imgs/`。
- `save()` 若不传 `ocr.config.output` 会重新使用 `OutputConfig()` 默认值，可能忽略解析器中关闭可视化或美化格式等设置。
- 上面的存在性和页索引检查不能替代识别质量验收；空白页可合法无文本，有文字的页面异常为空时必须复核。
- `result.markdown` 默认是 `pretty=True` 的字符串；保存默认 `pretty_markdown=False`。需要一致时使用 `result.to_markdown(pretty=ocr.config.output.pretty_markdown)`，或直接读取保存文件。
- `page.markdown` 是字典，文本在 `page.markdown["markdown_texts"]`；`page.json` 是字典属性，不是 JSON 字符串，也不是可调用函数。
- `ParseMetrics.to_dict()` 的 `total_seconds` 不含模型初始化耗时；`model_initialization_seconds` 单列。不要据此误报完整冷启动时长。

## 3. JSON 结构与定位

内存中访问 `page.json["res"]`；`page.save_to_json("page.json")` 写出的文件顶层**直接就是 `res` 的内容**，不保留外层 `{"res": ...}`。不能对保存文件再强行访问 `["res"]`。

关键字段：

| 字段 | 含义 |
|---|---|
| `input_path` | 原输入路径；裁页/转换副本需调用方额外保留来源映射 |
| `page_index` | PDF 从 `0` 开始；普通图片通常为 `null` |
| `page_count` | PDF 总页数；普通图片通常为 `null` |
| `width` / `height` | 本页用于解析的渲染图像尺寸 |
| `model_settings` | 该页生效的预测开关，不是完整 engine 配置 |
| `parsing_res_list` | 按流水线组装顺序排列的解析块 |
| `layout_det_res` | 开启版面检测时附带的检测结果；布局分数不是 OCR 文本置信度 |

每个解析块包含 `block_label`、`block_content`、`block_bbox`、`block_id`、`block_order`、`group_id`；`block_polygon_points`、`global_block_id`、`global_group_id` 仅在实际存在时出现，不要求它们总是存在。

- `block_id` 是该页解析列表的 **0-based** 下标。它不是检测框 ID，也不是跨运行稳定的全局 ID。
- `block_order` 是部分标签的 **1-based** 序号；图片、表格、图表、页眉页脚等标签可能为 `null`。不能丢弃所有 `block_order=null` 的块，也不能用它代替 `block_id`。
- `block_bbox` 是渲染页面像素坐标 `[x1, y1, x2, y2]`，通常左上为原点；不是归一化 0–1000 坐标，也不是 PDF 原始 point 单位。结合 `width`、`height` 和实际渲染变换进行映射。
- 页索引、bbox 与对应产物一起保留。来自 PDF 副本的索引不能直接作为原文档页号；跨次修改布局阈值、合并规则或模型版本后不要复用旧块编号。
- `block_content` 可能是文本、HTML 表格或 LaTeX 公式；图像块可以为空。JSON 默认不序列化实际图片数据，不要假设它是可独立恢复所有视觉资产的归档。
- 内存中的 `page["parsing_res_list"]` 是 `Block` 对象列表；JSON 中才是普通字典。不要把内部对象直接 `json.dumps()`。

读取已导出的页面文件而不重新推理：

```python
import json
from pathlib import Path

page = json.loads(Path("/absolute/output/api-run-001/report/report.page_000003.json").read_text(encoding="utf-8"))
for block in page["parsing_res_list"]:
    if block["block_label"] == "table":
        print(page["page_index"], block["block_id"], block["block_bbox"])
        print(block["block_content"])
```

只展示回答所需内容，不把整份长文档无界打印到上下文。页面可视化或原图可用于复核，但版面 PDF 只是画框后的图像 PDF，不是添加了可搜索文字层的 PDF。

## 4. 批量 API 的审计

使用 `summary = ocr.parse_dir(input_dir, output_dir, num_workers=2)` 后，至少核验 `summary.crashed`、`summary.failed`、`summary.succeeded` 与 `summary.total`。需要可序列化汇总时使用标准库 `dataclasses.asdict(summary)`，不要虚构 `summary.to_dict()`。

并发流程内部创建并释放自己的共享流水线，不复用此前 `ocr.parse()` 初始化的 `ocr._pipeline`。先做单文件冒烟后关闭该实例，再创建实例执行大批量，避免两份版面模型同时占用资源。

`parse_dir()` 不暴露逐文档回调或逐页 JSON 保存开关。批量且每个文档都需要 JSON 时：

1. 明确枚举受支持输入并检查输出冲突。
2. 使用单个 `VenusOCR` 实例顺序调用 `parse()`，逐个保存本节所述产物并记录成功/失败。
3. 若必须并发导出 JSON，明确说明这需要额外集成，不伪造现有 API 参数，也不先跑完整批量后再次识别所有文件。

使用 `parse_files()` 时，不能按“返回顺序与输入一一对应”推断全成功，因为解析异常会被跳过。按 `result.input_path` 与原输入清单对账；若传 `input_root`，确保所有输入的父目录都位于该根目录之下。

## 5. 纯区域识别，不运行完整版面流程

仅用于用户给出的、已裁好的文字/公式/表格区域，或明确接受整图单一任务识别的输入。不能替代复杂多栏整页的版面分析，也不能直接输入 PDF。

**不要通过 `predict.use_layout_detection=false` 规避版面依赖。** 当前 `DocumentParsingPipeline.__init__()` 仍无条件创建版面模型；该开关仅跳过后续检测，后端和权重仍需就绪。

已经具备完整环境时，可以在 CLI 中关闭检测并设定 `predict.prompt_label` 为 `text`、`formula` 或 `table`。没有版面模型、只需识别单个区域时，可以直接使用项目自带的 `VLRecognizer`，它只调用识别服务：

```python
import os

from venusocr.models.vlm import VLRecognizer

recognizer = VLRecognizer(
    server_url=os.environ["VENUSOCR_REGION_SERVER_URL"],
    model_name=os.environ["VENUSOCR_REGION_MODEL_NAME"],
    api_key=os.environ.get("VENUSOCR_API_KEY"),
    max_concurrency=1,
)
try:
    records = recognizer.recognize(
        [{"image": os.environ["VENUSOCR_REGION_IMAGE"], "query": "Text Recognition:"}],
        max_new_tokens=4096,
        skip_special_tokens=True,
        temperature=0,
        min_pixels=112896,
        max_pixels=1003520,
    )
    print(records[0]["result"])
finally:
    recognizer.close()
```

先设置示例显式读取的 `VENUSOCR_REGION_SERVER_URL`、`VENUSOCR_REGION_MODEL_NAME`、`VENUSOCR_REGION_IMAGE`；图片为可读的绝对文件路径，模型名来自服务的模型列表。不要把这些变量误当作 CLI 支持的环境配置项。

准确使用任务 prompt：`Text Recognition:`、`Table Recognition:`、`Formula Recognition:`。不替换成“请总结文档”等聊天指令。

该调用返回原始区域识别文本，不产生布局、页 JSON、文档 Markdown 或图片归档；表格还可能需要完整流水线的 OTSL → HTML 后处理。不要把结果误报为全功能解析产物。调用原始 OpenAI 接口时也必须遵守同样的图片、prompt、模型名与采样语义；优先复用该封装，不手写不一致的请求。
