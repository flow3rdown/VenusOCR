# VenusOCR 2.0 开源前运行验证

验证日期：**2026-11-24**。范围：本地分发模型、推理 SDK、可复现使用文档与发布阻断项。代码基线为 `98c2d82`，本轮修复尚未提交 Git。

## 结论

**在经过修复和版本固定后，主要推理路径已实际跑通；不能直接把原始目录视为无需修改即可发布。**

- Transformers 能加载当前权重并生成文本、LaTeX 和 OTSL 表格。
- SDK 标准 AR 服务无需 vLLM 源码补丁即可启动；单图、两页 PDF 和目录批处理均已完成真实 GPU 推理。
- 模型仓库附带的 vLLM 0.20.0 补丁可应用并通过检查；PTPv2 服务完成真实识别，三个样例的输出 Token IDs 与 SDK AR **完全一致**。
- 同名不同扩展名输入的输出覆盖已修复，PDF 与 PNG 各有独立 Markdown 和可视化 PDF。
- Python API 的 Markdown、指标、逐页 JSON 导出已验证；Gradio 界面构建及本地 HTTP 服务已验证。
- **公开地址、模型许可及检查点评测映射仍未确认**，这些是发布流程的剩余项，不是本次推理成功能自动证明的内容。

## 实测环境

| 项目 | 版本 / 说明 |
| --- | --- |
| 环境 | `env-3.11.13` / Python 3.11.13 |
| GPU | 单张 NVIDIA A100-SXM4-40GB |
| PyTorch / Torchvision | 2.11.0+cu130 / 0.26.0+cu130 |
| Transformers | 初始 5.24.0；验证后固定 5.5.4 |
| vLLM | 0.20.0；AR 插件测试先在未打补丁状态完成 |
| 版面模型 | `PaddlePaddle/PP-DocLayoutV3_safetensors`，下载后使用本地路径 |
| Gradio | 6.32.0 |
| 测试框架 | pytest 9.2.0 |

源码所在的 FUSE 暂存目录及其复制来的 `venusocr.egg-info` 由 root 拥有，普通用户的直接 editable build 首次因目录时间戳权限失败。本次把源码复制到本地临时目录构建/安装，保持模型权重在原位置读取。**这属于本机暂存权限问题，不是正常用户自行 clone 后的包构建错误**；发布内容不能包含旧 `*.egg-info`、嵌套 `.git` 和缓存。

## 验证矩阵

“通过”表示真实命令成功且检查了产物；不代表全量评测或所有硬件版本均兼容。

| 路径 | 输入 / 检查 | 结果 |
| --- | --- | --- |
| Transformers 原始环境 | 5.24.0，自定义模型加载并调用 `generate()` | **失败**：缓存输出 API 不兼容 |
| Transformers 固定版本 | 5.5.4，英文文本、公式、表格、正文四个区域 | **通过** |
| 独立 Transformers 脚本 | `scripts/infer_transformers.py`，表格样例 | **通过**：OTSL 输出保留结构 Token |
| PP-DocLayoutV3 | 合成示意页面 | **通过**：4 个区域，包含标题/正文/公式/表格 |
| SDK AR 服务 | 未修改 vLLM 源码，SDK plugin，`/health` 与三个真实请求 | **通过** |
| SDK 默认编译模式，无补丁 | 恢复原 vLLM 后启动 SDK AR，完成 CUDA graph 捕获，三个真实请求 | **通过**：18 / 40 / 84 Token，与首次 AR 结果一致 |
| 单图解析 | 示例页面 | **通过**：Markdown 和布局 PDF |
| 多页 PDF | 同一示例构造的两页 PDF | **通过**：2 页、8 区域、8 次识别 |
| 目录批处理，修复前 | 同名 `sample.pdf` 与 `sample.png` | **推理通过，但输出覆盖**；统计成功数不能证明产物完备 |
| 目录批处理，修复后 | 同名 PDF + PNG，3 页 | **通过**：2 文档、12 区域，4 个独立输出文件，失败数 0 |
| vLLM PTPv2 补丁 | 应用、静态编译和 `--check` | **通过** |
| 补丁回退与重新应用 | `--reverse` 后 SDK 模式运行，再次应用并检查 | **通过**；验证结束保留已检查的 PTPv2 补丁 |
| PTPv2 识别 | 文本/公式/表格，greedy，返回 Token IDs | **通过**：分别 18 / 40 / 84 Token，与 SDK AR 完全一致 |
| 默认编译模式 AR / PTPv2 | 不设置 `ENFORCE_EAGER`，完成编译和 CUDA graph 捕获，三个区域请求 | **通过**：正文/公式/表格分别 55 / 40 / 84 Token，两个模式的 Token IDs 完全一致 |
| SDK + PTPv2 | 自动发现服务名，页面识别及目录批处理 | **通过** |
| Python API | `result.save()`、`metrics.to_dict()`、`page.json` 序列化 | **通过** |
| Gradio | `build_demo()`、本地启动、HTTP `/` | **通过**：HTTP 200；不把此项视为所有浏览器交互均已覆盖 |
| 单元测试 | 配置、批处理、结果、文件 IO、服务参数等 | **通过：51 项** |
| 干净 wheel 安装 | 无 `build/` 缓存的源码构建，安装后逐文件核对、隔离导入模式运行测试 | **通过：包内 Python/YAML 与源码一致，51 项测试通过** |
| 安装后运行仓库示例 | wheel 安装，从 SDK 外运行 `examples/parse_file.py`，连接默认编译模式 PTPv2 服务 | **通过**：真实页面识别和文件保存 |
| PaddleX / PaddlePaddle | 可选原生版面后端 | **未验证**；当前环境缺少对应 PaddlePaddle GPU 运行时 |
| CPU-only / 其他 GPU | 其他硬件或 CUDA 组合 | **未验证** |
| 全量 OmniDocBench | 所有页面与官方评测脚本 | **本次未执行** |
| H20 性能复现 | 论文的整套并发/吞吐实验 | **本次未执行** |

PTPv2 的首次功能测试使用 `ENFORCE_EAGER=1`。随后关闭调试模式，分别启动 AR 与 PTPv2，实际完成 `torch.compile` 和 CUDA graphs 捕获，并运行正文/公式/表格请求；三个请求的 Token IDs 均一致。默认编译模式测试保留了脚本的 `MAX_MODEL_LEN=10240`、`MAX_NUM_SEQS=8`，仅将端口设为测试端口、绑定回环地址并将显存预算设为 0.60。**这仍不是论文全量精度或性能复现**。

## 已修复的问题

### 1. 推理依赖无上限导致 API 不兼容

原环境的 Transformers 5.24.0 在 `generate()` 中报：

```text
TypeError: VenusOCRCausalLMOutputWithPast.__init__() got an unexpected keyword argument 'cache'
```

采用**经实测的版本固定**，而不是声称已经完成新版缓存 API 的模型代码迁移：模型 `requirements.txt` 和 SDK 推理 extras 固定 Transformers 5.5.4；vLLM 固定 0.20.0。显式补齐 Torchvision，PyTorch 下限与 Transformers 5.5.4 的要求对齐。升级兼容性仍需单独测试。

### 2. 单卡默认配置和文档互相矛盾

原 YAML 默认 `transformers` + `gpu:1`，而配置类/测试/README 仍写 PaddleX 或 `gpu:0`。当前 YAML、配置类、测试、README 与仓库自带 agent skill 统一为 **Transformers + `gpu:0`**。

### 3. `serve` 参数透传错误

`argparse.REMAINDER` 会把模型路径后面的正常 `--host` / `--port` 也吞入透传参数，导致重复参数和残留的 `--`，vLLM 无法启动。当前在 CLI 层分离 `--` 前后的参数，并新增回归测试。

服务 launcher 默认只启用 `venusocr` 插件，避免扫描环境里不相关的 PaddleX 插件。本机旧 PaddleX 插件还曾触发旧 Matplotlib 与 NumPy 的导入错误；它不属于默认 Transformers 推理管线的必需依赖，因此不靠修改无关包来解决。

### 4. 同名 PDF / 图片静默覆盖结果

初次目录测试中，2 个文档均报告成功，但 `sample.pdf` / `sample.png` 都写入 `sample/`，最后只剩一份产物。当前在同名输入或扩展名碰撞时保留输入扩展名，输出为：

```text
sample.pdf/sample.pdf.md
sample.pdf/sample.pdf_layout.pdf
sample.png/sample.png.md
sample.png/sample.png_layout.pdf
```

增加了同名文件、无关文本文件和带点文件名的回归测试；真实批处理复跑成功。正常不冲突文件继续使用原来的 stem 命名。

### 5. 开源后的测试依赖与文件遗漏

补齐测试用 PyMuPDF / pytest-asyncio；移除忽略全部测试的 `.gitignore` 条目，并允许提交测试所需的 `tools/compare_outputs.py`，避免本机有文件而 GitHub 用户没有文件。现有测试输出仅有外部 pytest 插件和 `fitz` API 的弃用警告，不影响通过结论。

### 6. README 的使用和结果口径

- 移除面向外部用户不可用的内网模型下载流程及未经确认的“已正式开源”日期。
- 修正 **Qwen3-0.6B** 架构名，不再写 Qwen3.5-0.6B 或“从零构建全部组件”。
- 分开 **96.24 / PP-DocLayoutV3** 与 **96.45 / Tele-layout**，不在默认 SDK 上套用后者。
- 明确 OTSL 原始输出、HTML 表格转换、JSON 的显式导出和不支持的任务。
- 加入论文原图导出的 PNG 与真实推理输出的合成样例；示例表格数字不作为评测指标。
- 中英文 SDK README、模型卡、可运行示例和版本说明同步更新。

### 7. 本机残留构建缓存

首次 wheel 构建复制到了暂存目录里的旧 `build/lib/`，setuptools 按时间戳复用旧文件，导致安装包中的版面模块与当前源码不一致。已使用排除 `build/`、`dist/`、`*.egg-info` 和缓存的干净源码重新构建，并逐一比较 wheel 内全部 Python/YAML 文件与当前源码；安装后以 `--import-mode=importlib` 运行测试，**51 项通过**。这些生成目录本来就不应进入 GitHub 发布包，不需要把旧缓存当作源代码修补。

此外，当前共享环境的 `pip check` 仍提示 `Werkzeug 3.1.8` 与已经存在的 `MarkupSafe 2.0.1` 版本要求不匹配。它不在 VenusOCR 的核心推理依赖链中，本轮没有修改无关包，也没有声称整个共享环境依赖完全无冲突。

## 复现命令

以下从 SDK 仓库根目录执行，权重路径请换为本地完整模型目录。两套服务不要同时占用同一端口。

```bash
pip install -e '.[transformers,serve,dev]'
python -m pytest -q

venusocr serve ./models/VenusOCR2-1B \
  --host 127.0.0.1 --port 8000 --max-num-batched-tokens 4096 -- \
  --max-model-len 8192 --max-num-seqs 8 \
  --gpu-memory-utilization 0.35 --enforce-eager

venusocr parse paper.pdf -o ./output \
  --layout-model-dir ./models/PP-DocLayoutV3 \
  --set predict.temperature 0 --set predict.max_new_tokens 768
```

停止标准服务后，在模型仓库内验证 PTPv2：

```bash
bash scripts/patch_vllm_v0_20_ptp2.sh
bash scripts/patch_vllm_v0_20_ptp2.sh --check
HOST=127.0.0.1 PORT=8000 ENFORCE_EAGER=1 \
  GPU_MEMORY_UTILIZATION=0.35 MAX_MODEL_LEN=8192 \
  bash scripts/vllm_serve_ptp2.sh
```

补丁修改已安装的 vLLM；请用独立服务环境，保留备份。本次实际执行了 `--reverse`，在恢复后的 vLLM 上验证 SDK 默认编译模式 AR，再重新应用补丁并通过检查。测试服务已关闭，GPU 已释放；当前环境保留 Transformers 5.5.4、最终安装包和已检查的 PTPv2 补丁。

## 发布前仍需确认

1. **最终公开地址**：GitHub、Hugging Face repository ID、论文 URL。替换 README 的 `YOUR_ORG/YOUR_MODEL`，同步包元数据和草稿配置里的 verifier repository ID；不要假定目前硬编码的地址已经上线。
2. **模型权重许可**：SDK 为 Apache-2.0，不等于权重自动使用 Apache-2.0。确认后再补 Hugging Face `license` 元数据和模型许可证文件。
3. **检查点来源**：主模型配置没有提供足以证明其对应论文 Stage 4 的训练记录。须确认训练阶段、权重 SHA256 与论文实验的映射；当前把表格标为论文结果，不宣称本次烟测复现了分数。
4. **发布根目录**：已将原 `VenusOCR2.0/` 内的源码和真实 Git 历史移到 `opensource/github/VenusOCR/` 根目录；外层空 Git 元数据与旧构建缓存另行备份，不再保留嵌套项目目录。
5. **可选后端范围**：若正式宣传 PaddleX / CPU 支持，应补充对应环境的真实运行记录。PyPI 发布状态也尚未验证，因此 README 从本地源码安装，不假定 `pip install venusocr` 会安装本次版本。

## 参数量与文件体积

主模型 safetensors 保存 **1,190,049,392 个 BF16 Tensor 元素**，包含分别保存的 Embedding / LM head 张量；这不是去重后的模型参数计数。草稿 safetensors 保存 **377,755,648 个 BF16 Tensor 元素**；其中冻结词表 Embedding 和 LM head 合计约 310.6M，其余约 **67.1M** 才是论文标注的额外草稿参数。不要用“67M”估算整份草稿权重下载量。

## 文档参考与结果来源

README 表达参考了以下**官方模型卡**的介绍、环境、下载、推理、限制和引用结构；未复制其模型专属提示词或未经本机验证的安装命令。读取日期均为 2026-11-24。

- [PaddlePaddle/PaddleOCR-VL-1.6 模型卡](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/blob/132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd/README.md)。
- [tencent/HunyuanOCR 模型卡，标题为 HunyuanOCR-1.5](https://huggingface.co/tencent/HunyuanOCR/blob/47644ecc4fc854efa4f505155158831f36773ee4/README.md)。实际官方仓库 ID 是 `tencent/HunyuanOCR`，不是猜测的 `tencent/HunyuanOCR-1.5`。
- VenusOCR 论文来源：`table/eval_result/OmniDocBench.tex`、`sections/02_model.tex`、`sections/05_decoding_system.tex`、`sections/06_experiments.tex`、`figures/hero_overview.pdf`、`figures/architecture.pdf` 与对应运行指标 JSON。论文公开 URL 尚待确认。

日志、逐请求输出及 Token IDs 保留在本次本地验证产物中；没有上传任何模型、文档或结果，也没有提交 Git。

## 目录与展示整理

按后续要求，模型卡和中英文项目 README 不再包含识别样例图片及输出展示，新增的样例附件移出发布目录并另行备份。安装、调用说明以及论文结果图保留，复现时需使用自己的 PDF 或图片。

本记录中的模型运行结果只对应当时实际测试的权重路径。模型路径既然需要重新确认，不应将这些历史烟测结果视为最终发布权重的验证结论；本轮仅整理单层代码项目，不搬动或删除模型权重。

### 合并后验证（2026-11-24）

- 项目根目录：`/group/40119/arleyli/DocOCR/PreTrain/opensource/github/VenusOCR`。原嵌套目录已移空并移除，Git HEAD 保持 `98c2d82dcbbdb50a67f76c8910f649e6c561b65f`，未提交的源码修改保留。
- 在 `env-3.11.13` 中，从新根目录执行 `python -m pytest -q -p no:cacheprovider`：**51 passed**，仅有 AnyIO / PyMuPDF 依赖弃用警告。
- `python -m venusocr --help` 与 `python examples/parse_file.py --help` 均成功。通用调用脚本要求显式提供 `--input`，不依赖已移出的样例附件。
- 中英文项目 README 和模型 README 的本地链接、图片路径以及 Python 代码块语法检查通过；不再引用样例展示附件。
- 外层空 Git 元数据、旧入口 README、旧构建缓存和本轮新增的样例附件保存在 `/group/40119/arleyli/DocOCR/PreTrain/opensource/release_validation/VenusOCR-2026-11-24/layout_cleanup_backup/`，未删除模型权重，也未重新运行模型推理。
