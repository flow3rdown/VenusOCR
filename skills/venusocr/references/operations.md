# 运行配置与排错

在首次配置、服务启动、批量调优或错误恢复时读取。执行推理、下载或安装前，先确认目标运行环境的资源和操作权限。

## 1. 环境与安装

### 1.1 检查实际环境

先激活已确认的 Python 环境，再检查。若使用 conda，先按实际安装方式初始化 shell，并将下例中的占位符替换为已有环境名；不要假定固定安装路径或自动创建环境。

```bash
conda activate '<existing-environment-name>'
```

```bash
command -v python
python --version
python -m pip --version
command -v venusocr
python -m venusocr --version
python -m venusocr parse --help
python -c 'import venusocr; print(venusocr.__file__)'
```

要求 Python 至少 `3.10`。包元数据列出 `3.10`、`3.11`、`3.12`，实际还要满足目标 Paddle / torch / vLLM 的版本限制。`--help` 成功不代表权重、GPU 或识别服务可用。

### 1.2 按角色安装，不一次安装全部

优先复用已验证环境。需安装本仓库版本时，先将 `REPO` 设为目标机器上真实仓库路径；不要因 PyPI 上存在同名包就认为它与当前源码一致。

```bash
REPO='/absolute/path/to/VenusOCR2.0'
test -f "$REPO/pyproject.toml"
```

检查成功且已获准安装后，只选所需角色的一条命令：

| 角色 | 安装命令 | 额外条件 |
|---|---|---|
| 解析客户端 + Transformers 版面 | `python -m pip install -e "${REPO}[transformers]"` | 本地 safetensors 版面模型；能导入 PPDocLayoutV3 模型与处理器 |
| 解析客户端 + PaddleX 版面 | `python -m pip install -e "${REPO}[paddlex]"` | 单独安装与设备/CUDA 匹配的 PaddlePaddle；extra 只固定 PaddleX `3.6.0` |
| VenusOCR vLLM 服务端 | `python -m pip install -e "${REPO}[serve]"` | 合适的 GPU、可信且完整的 VenusOCR checkpoint，以及相互兼容的 vLLM / torch / Transformers |
| Gradio 客户端 | `python -m pip install -e "${REPO}[transformers,demo]"` | 仍需版面模型和已启动的识别服务；使用 PaddleX 时对应替换 extra |

只连接已有服务的解析机不需要安装 vLLM。仅启动识别服务的机器不必安装 PaddleX 版面运行时。

**不要把依赖下限当成兼容性保证。** 本项目已实测并固定 `vllm==0.20.0` 与 `transformers==5.5.4`；Transformers 5.24.0 与当前识别模型的缓存输出接口不兼容。插件引用多处 vLLM 内部接口，不承诺旧版或未来版本均兼容。升级前记录版本，在独立环境验证。PTPv2 另需模型仓库附带的 vLLM 0.20.0 补丁，标准 AR 插件不会自动启用它。

在资源合适的目标机器检查版面导入能力：

```bash
python -c 'from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessor; print("layout imports OK")'
```

PaddleX 路径则检查 `paddle` 与 `paddlex`；缺少依赖时先确认安装权限及资源条件，不为通过检查直接补装大型依赖。

## 2. 模型与设备

- **PaddleX 版面权重：** 使用 PP-DocLayoutV3 的 Paddle 推理目录，不是 Transformers safetensors 目录。以目标 PaddleX 运行时所需的配置、结构和参数文件为准。
- **Transformers 版面权重：** 使用 PP-DocLayoutV3 的完整 Transformers 模型目录，包含模型配置、图像处理器配置与对应权重；不能只给单个 `.safetensors` 文件。省略路径会使用 `PaddlePaddle/PP-DocLayoutV3_safetensors`，可能联网下载。
- **VenusOCR 识别权重：** 使用完整 VenusOCR checkpoint，核对配置中的 `architectures` 与插件注册的 `VenusOCRForConditionalGeneration` 一致，并核对 tokenizer / processor / 权重文件。没有 checkpoint 时请求用户提供可信来源，不编造公开模型 ID。
- **版面设备：** 使用明确的 `gpu:0` 或 `cpu`。CPU 只适用于本地版面模型，不代表 VLM 在 CPU 上运行；同时确认 CPU 和内存能承载页面渲染。
- **设备隔离：** 两卡机器可让服务进程 `CUDA_VISIBLE_DEVICES=0`，解析进程 `CUDA_VISIBLE_DEVICES=1`；解析进程内部仍应选 `--device gpu:0`。仅一张卡时，确认共享显存余量，或在有足够 CPU/内存的解析机运行 CPU 版面。
- **可信代码：** 服务启动器固定加 `--trust-remote-code`；只对可信 checkpoint 使用。目录字符串本身不是安全验证。

## 3. 启动与检查识别服务

先复用用户指定的服务。只有明确需要新服务、目标机器可访问且资源允许时才启动。

### 3.1 使用正确参数顺序

`serve` 使用 `argparse.REMAINDER`：**把 VenusOCR 封装层选项放在模型路径之前，原生 vLLM 选项放在模型路径后的 `--` 之后**，避免封装参数被当成透传参数，或与默认参数重复。

```bash
VLM_MODEL_DIR='/absolute/models/VenusOCR2-1B'
CUDA_VISIBLE_DEVICES=0 venusocr serve \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name VenusOCR \
  --max-num-batched-tokens 16384 \
  "$VLM_MODEL_DIR" \
  -- --gpu-memory-utilization 0.8
```

`0.8` 只是示例，不是最低资源保证；结合目标 GPU、并行任务和模型配置确定。封装器默认关闭 prefix caching、把多模态处理器缓存设为 `0`，并添加 `--trust-remote-code`。只透传目标 vLLM `serve --help` 确认支持的参数，不添加其他 OCR 项目的专用选项。

该命令前台阻塞；用独立终端或用户已有进程管理器运行，另一个终端检查就绪。不要因为它未立即退出就再次启动。

跨机器访问时，`127.0.0.1` 指解析机自己而非服务机。需服务机绑定可访问的网卡地址，并正确配置网络访问控制；不要把 `0.0.0.0` 作为客户端 URL，也不要默认把无鉴权服务暴露到公网。

### 3.2 核对服务与模型

从实际解析机访问：

```bash
BASE_URL='http://127.0.0.1:8000/v1'
curl --fail --silent --show-error --connect-timeout 5 --max-time 15 "${BASE_URL%/}/models"
```

检查返回的 `data` 列表及每个 `id`，选择真实 VenusOCR 模型名。`BASE_URL` 必须是 OpenAI API 根地址（通常以 `/v1` 结尾），不是 `/v1/models`、`/v1/chat/completions` 或 Gradio 页面。如网关有前缀，以网关约定为准。

服务要求鉴权时，按用户批准的安全方式给探测请求提供 Bearer key，并将同一 key 传给 SDK 的 `engine.vl_rec_api_key`。不要回显、提交密钥或把密钥直接写入可见命令历史。未配置 key 时 SDK 实际使用字符串 `"null"`；**不能指望设置 `OPENAI_API_KEY` 自动替代它**。

模型列表为空、模型身份不确定、服务未就绪时停止，不发整批请求。`/models` 成功后仍需小样本推理验收。

### 3.3 插件检查

vLLM 必须能发现当前环境已安装包的 `vllm.general_plugins` entry point，名称是 `venusocr`。仅 `cd` 到仓库或设置 `PYTHONPATH` 不等同于安装 entry point。

遇到架构未注册时，检查 `venusocr` 与 PATH 中的 `vllm` 是否来自同一环境；如果显式设置了 `VLLM_PLUGINS`，确认允许列表包含 `venusocr`。不要默认修改 vLLM 源码，也不要把 checkpoint 改成其他模型的架构名来绕过错误。

## 4. 显式配置模板

在独立任务 YAML 中配置，不修改包内 `venusocr/config.yaml`。以下模板适用于已有 Transformers 权重和已有服务的解析机；路径、设备、URL、模型名必须替换并核验。低并发值用于初次验证，不代表性能最优。

```yaml
engine:
  layout_backend: transformers
  layout_model_name: PP-DocLayoutV3
  layout_model_dir: /absolute/models/PP-DocLayoutV3_safetensors
  device: gpu:0
  layout_batch_size: 1
  vl_rec_backend: vllm-server
  vl_rec_server_url: http://127.0.0.1:8000/v1
  vl_rec_api_model_name: VenusOCR
  vl_rec_api_key: null
  vl_rec_max_concurrency: 4
predict:
  use_layout_detection: true
  use_chart_recognition: false
  use_seal_recognition: false
  use_ocr_for_image_block: false
  markdown_ignore_labels: []
output:
  save_markdown: true
  save_layout_vis: true
  pretty_markdown: false
batch:
  num_workers: 2
  page_batch_size: 4
  loader_workers: 1
  saver_workers: 1
```

选择 PaddleX 时同时更换 `layout_backend`、对应版面权重目录和运行时依赖。正式对齐 PaddleOCR-VL 1.6 时使用 `paddlex`；Transformers 不能承诺检测框逐浮点一致，不在恢复过程中无提示切换。

配置优先级：包内 YAML → 用户 `--config` 深合并 → CLI 快捷参数 → `--set KEY VALUE`。重复 `--set` 的同名项最后一个生效。Python 中显式传入 `config=VenusOCRConfig(...)` 会优先于 `config_path`，然后再应用 `overrides`。

- `--set` 是两个参数，不是 `KEY=VALUE`。布尔值使用 `true` / `false`，列表和字典使用合法 JSON 并加 shell 引号。
- 未知配置项会被拒绝。不要添加 `output.save_json`、`engine.vl_rec_timeout`、`predict.use_doc_orientation_classify` 或其他项目的配置。
- YAML 不做 shell 环境变量插值，不能把 `${API_KEY}` 当作已展开的密钥。优先在调用代码中从环境显式读取并通过 `overrides` 传入；配置文件含密钥时妥善限制访问且不提交。
- 用户 YAML 会先校验再应用 CLI overrides。非法 YAML 值不能依赖后续 `--set` 修复，先修正配置源。

在目标环境只加载并校验配置、不创建模型：

```bash
CONFIG='/absolute/job/venusocr.yaml'
python -c 'import sys; from venusocr.config import load_config; c=load_config(sys.argv[1]); print("config OK", c.engine.layout_backend, c.engine.device)' "$CONFIG"
```

不要为了检查配置访问 `ocr.pipeline`，该属性会加载版面模型并创建识别客户端。

## 5. 参数语义与调优边界

| 参数 | 实际含义及注意事项 |
|---|---|
| `engine.layout_batch_size` | 版面模型内部批大小；默认 `8`，缩小有助于减轻版面推理压力 |
| `engine.vl_rec_max_concurrency` / `--max-concurrency` | 同一个识别客户端的并发请求上限；默认 `200`，不是所有服务都能承受 |
| `batch.num_workers` / `--num-workers` | 目录任务的页面批处理线程数；设为 `1` 会切换到串行实现 |
| `batch.page_batch_size` | 只控制并发目录路径，默认 `64`；**不控制单文件 `parse()` 的内部采样批大小** |
| `batch.loader_workers` / `batch.saver_workers` | 并发目录的加载、保存线程数；PDFium 操作受全局锁保护，不保证渲染线性加速 |
| `predict.max_new_tokens` | 完整流水线缺省为每个识别区域 `4096`，不是整文档 token 预算 |
| `predict.min_pixels` / `predict.max_pixels` | 每个区域的 VLM 图像预处理约束；流水线缺省为 `112896` / `1003520`，要求 min 不大于 max |
| `predict.temperature` | 客户端未显式设置时使用 `0`；不能据此保证完全确定性 |
| `predict.prompt_label` | 关闭版面检测后使用 `text` / `formula` / `table`；`ocr` 是 `text` 的兼容别名，不接受任意自然语言 prompt |
| `predict.markdown_ignore_labels` | Markdown 排除标签，当前有效默认值是 `[]`；不等于跳过识别，也不等于从原始 JSON 删除块 |
| `predict.format_block_content` | 默认不格式化 JSON 的 `block_content`；开启会改变内容表达 |
| `VENUSOCR_PDF_RENDER_SCALE` | PDF 渲染比例，默认 `2.0`，约等于 `144 DPI`；若设置，使用有限正数，并在解析及可视化保存期间保持一致 |

批量内存问题优先降低页面批大小、worker、版面批大小和识别并发，再考虑关闭非必需的版面可视化。单文件内部以 `64` 页采样并积累页面结果；降低 `batch.page_batch_size` 对此无效。超长 PDF 可在用户允许时拆分副本，或使用包含该文件的隔离输入目录走并发批量路径并设置小页面批。即便并发路径也按文档累积结果，不保证恒定内存。

降低渲染比例或像素预算会影响识别，不能为了跑通悄悄降低质量。增加 token 上限前核对服务的上下文上限和实际截断证据。客户端只取识别文本，不将 `finish_reason` 暴露在页面结果中，因此不能用结果字段直接证明“未截断”。

`predict.use_chart_recognition`、`predict.use_seal_recognition`、`predict.use_ocr_for_image_block` 必须为 `false`，设成 `true` 会触发配置校验错误。

## 6. 错误恢复

没有稳定的 JSON 错误码协议。结合退出状态、当前日志、实际 HTTP 错误和产物判断；需要 traceback 时把全局参数放在子命令之前：`venusocr --log-level DEBUG parse ...`。调试日志可能含路径和响应信息，分享前去敏。

| 现象 | 检查与动作 |
|---|---|
| `venusocr` 找不到 / 导入失败 | 核对激活环境、包路径与安装角色；不要在另一个环境反复安装 |
| `PPDocLayoutV3...` 无法导入 | 核对 Transformers 是否确实提供这两个类；按已验证依赖组合修复，不只看包版本下限 |
| PaddleX / Paddle 缺失 | 核对选择的后端及 PaddlePaddle 安装；服务端 vLLM 安装不能补齐本地版面依赖 |
| 下载超时 / 找不到权重 | 核对本地完整目录、权重格式及路径归属机器；下载须获授权，不换成不可信镜像 |
| `invalid device ordinal` / CUDA OOM | 检查可见设备编号、占用和配置中的 `gpu:0`，区分版面模型和服务模型；按上节逐项降低并发/批大小 |
| 无法获取模型列表 / 连接失败 | 从解析机检查 URL、`/v1`、网络、服务就绪与鉴权；显式模型名只绕过发现，不会修复服务不可用 |
| HTTP 401 / 403 | 检查 SDK 实际传入的 `engine.vl_rec_api_key` 和网关权限，不回显密钥 |
| HTTP 404 / model not found | 区分 URL 路径错误与模型名错误；核对 `/models` 返回的 ID |
| 架构不支持 / vLLM 内部导入错误 | 分别检查插件注册与版本兼容；注册成功也不代表实现可成功导入或权重可加载 |
| HTTP 429 / 5xx / 超时 | 看服务日志、并发、显存和代理超时。当前区域请求 timeout 是 `600s`，没有公开 timeout 配置项；先修复原因后仅重试受影响输入 |
| 参数或配置校验错误 | 对照当前 `--help` / `config.py`，删除不存在的键；不能通过随意添加 `--set` 扩展功能 |
| 文档为空 / 页数不足 | 检查原件、图片解码、加密/损坏 PDF、扩展名漏扫、多帧图片，以及实际检测框；有合法密码的文档可由独立工具生成获准的解锁副本 |
| 文本缺失、乱码、重复、表格截断 | 回查原页和逐页 JSON，检查模型身份、布局裁剪、生成/像素限制；不直接认定原文为空或编造补全 |
| 没有 JSON / `_DONE` | 区分默认保存行为和串行/并发路径，按主文档的输出契约处理 |
| 没有版面 PDF但退出成功 | 检查保存阶段日志与资源；向用户报告该产物失败，不能把 Markdown 成功等同于全部成功 |

瞬时故障在确认服务恢复后至多重试一次；配置、权重或数据问题先修正原因再重试。若故障持续，报告失败阶段、已完成范围与所需操作，停止扩大任务。不为排障终止他人服务或修改底层库。

## 7. Gradio 的使用范围

仅在用户需要交互展示时使用 `venusocr demo --config '/absolute/job/venusocr.yaml' --host 127.0.0.1 --port 7860`。Demo 不启动 vLLM，不替代正式结果导出；当前 JSON 面板只展示第一页，运行逻辑未调用 `DocumentResult.save()`，不能把页面展示当作完整持久化产物。默认不加 `--share`。
