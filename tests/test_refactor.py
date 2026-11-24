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

"""Unit tests for the vendored parsing/result layer (no GPU / paddle needed)."""

from venusocr.config import load_config
from venusocr.models.layout import (
    DEFAULT_LAYOUT_THRESHOLD,
    PP_DOCLAYOUT_V3_ID2LABEL,
    create_layout_detector,
)
from venusocr.pipeline.merge import filter_overlap_boxes
from venusocr.pipeline.tables import convert_otsl_to_html
from venusocr.pipeline.text import truncate_repetitive_content
from venusocr.results.page import Block, PageResult
from venusocr.results.document import DocumentResult


def test_convert_otsl_to_html_simple_table():
    otsl = "<fcel>a<fcel>b<nl><fcel>c<fcel>d<nl>"
    html = convert_otsl_to_html(otsl)
    assert html.startswith("<table>") and html.endswith("</table>")
    assert html.count("<tr>") == 2
    assert ">a</td>" in html and ">d</td>" in html


def test_convert_otsl_to_html_colspan():
    otsl = "<fcel>a<lcel><nl><fcel>c<fcel>d<nl>"
    html = convert_otsl_to_html(otsl)
    assert 'colspan="2"' in html


def test_truncate_repetitive_content_short_text_unchanged():
    assert truncate_repetitive_content("hello world") == "hello world"


def test_truncate_repetitive_content_line_level():
    content = "\n".join(["same line"] * 20) + "\n" + "x" * 4000
    content = "x" * 4000 + "\n" + "\n".join(["same line"] * 20)
    out = truncate_repetitive_content(content, min_count=3000)
    assert out == "same line"


def test_filter_overlap_boxes_drops_tiny_and_keeps_reference_out():
    res = {
        "boxes": [
            {"label": "text", "coordinate": [0, 0, 100, 100], "score": 0.9},
            {"label": "text", "coordinate": [1, 1, 99, 99], "score": 0.8},
            {"label": "text", "coordinate": [200, 200, 203, 203], "score": 0.9},
            {"label": "reference", "coordinate": [0, 200, 100, 300], "score": 0.9},
        ]
    }
    out = filter_overlap_boxes(res, "rect")
    labels_boxes = out["boxes"]
    # tiny box dropped; inner overlapping box dropped; reference removed
    assert len(labels_boxes) == 1
    assert labels_boxes[0]["coordinate"] == [0, 0, 100, 100]


def _make_page_result():
    blocks = [
        Block(label="doc_title", bbox=[0, 0, 100, 30], content="标题"),
        Block(label="text", bbox=[0, 40, 100, 90], content="正文内容"),
    ]
    return PageResult(
        {
            "input_path": "/tmp/doc.pdf",
            "page_index": 0,
            "page_count": 1,
            "width": 100,
            "height": 100,
            "doc_preprocessor_res": {"output_img": None},
            "layout_det_res": {"boxes": []},
            "table_res_list": [],
            "parsing_res_list": blocks,
            "spotting_res": {},
            "imgs_in_doc": [],
            "model_settings": {
                "use_doc_preprocessor": False,
                "use_layout_detection": True,
                "use_chart_recognition": False,
                "use_seal_recognition": False,
                "use_ocr_for_image_block": False,
                "format_block_content": False,
                "merge_layout_blocks": True,
                "markdown_ignore_labels": [],
                "return_layout_polygon_points": True,
            },
        }
    )


def test_page_result_markdown_and_json():
    res = _make_page_result()
    md = res.markdown
    assert "markdown_texts" in md
    assert "# 标题" in md["markdown_texts"]
    assert "正文内容" in md["markdown_texts"]

    js = res.json
    assert "res" in js
    blocks = js["res"]["parsing_res_list"]
    assert blocks[0]["block_label"] == "doc_title"
    assert blocks[0]["block_order"] == 1
    assert js["res"]["page_index"] == 0


def test_page_result_markdown_compatibility_snapshot():
    blocks = [
        Block(label="doc_title", bbox=[0, 0, 100, 20], content="文档标题"),
        Block(label="paragraph_title", bbox=[0, 25, 100, 45], content="1.2 小节"),
        Block(label="text", bbox=[0, 50, 100, 80], content="第一行\n第二行"),
        Block(
            label="table",
            bbox=[0, 90, 100, 140],
            content="<html><body><table><tr><td>A</td></tr></table></body></html>",
        ),
        Block(label="display_formula", bbox=[0, 150, 100, 170], content="$$x=1$$"),
        Block(label="formula_number", bbox=[105, 150, 130, 170], content="(1)"),
    ]
    result = _make_page_result()
    result["parsing_res_list"] = blocks

    markdown = result._to_markdown(pretty=False, show_formula_number=True)
    assert markdown["markdown_texts"] == (
        "# 文档标题\n\n### 1.2 小节\n\n第一行\n\n第二行\n\n"
        "\n\n<table><tr><td>A</td></tr></table>\n\n$$x=1 \\tag*{(1)}$$"
    )


def test_vllm_request_compatibility_snapshot(monkeypatch):
    from venusocr.models.vlm import VLRecognizer

    monkeypatch.setattr(
        "venusocr.models.vlm._image_to_data_url",
        lambda image, image_format="JPEG": "data:image/jpeg;base64,fixture",
    )
    recognizer = object.__new__(VLRecognizer)
    specs = recognizer._build_request_specs(
        [{"image": object(), "query": "Text Recognition:"}],
        max_new_tokens=4096,
        skip_special_tokens=True,
        repetition_penalty=1.05,
        temperature=None,
        top_p=None,
        min_pixels=112896,
        max_pixels=1003520,
    )

    assert specs == [
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,fixture"
                            },
                        },
                        {"type": "text", "text": "Text Recognition:"},
                    ],
                }
            ],
            {
                "temperature": 0,
                "max_completion_tokens": 4096,
                "extra_body": {
                    "skip_special_tokens": True,
                    "repetition_penalty": 1.05,
                    "mm_processor_kwargs": {
                        "min_pixels": 112896,
                        "max_pixels": 1003520,
                    },
                },
            },
        )
    ]


def test_document_result_saves_combined_markdown_in_stem_subdir(tmp_path):
    first = _make_page_result()
    second = _make_page_result()
    first["page_index"] = 0
    second["page_index"] = 1
    document = DocumentResult("/tmp/doc.pdf", [first, second])
    config = load_config().output.model_copy(
        update={"save_markdown": True, "save_layout_vis": False}
    )

    document.save(str(tmp_path), config)

    doc_dir = tmp_path / "doc"
    assert (doc_dir / "doc.md").read_text(
        encoding="utf-8"
    ) == document.to_markdown(pretty=config.pretty_markdown)
    # no legacy per-artifact sub-directories
    assert not (tmp_path / "markdown").exists()
    assert not (tmp_path / "json").exists()
    assert not (tmp_path / "layout_vis").exists()


def test_document_result_can_skip_markdown(tmp_path):
    first = _make_page_result()
    second = _make_page_result()
    first["page_index"] = 0
    second["page_index"] = 1
    document = DocumentResult("/tmp/doc.pdf", [first, second])
    config = load_config().output.model_copy(
        update={"save_markdown": False, "save_layout_vis": False}
    )

    document.save(str(tmp_path), config)

    assert not (tmp_path / "doc" / "doc.md").exists()


def test_document_save_records_saving_time(tmp_path):
    document = DocumentResult("/tmp/doc.pdf", [_make_page_result()])
    config = load_config().output.model_copy(
        update={"save_markdown": False, "save_layout_vis": False}
    )

    document.save(str(tmp_path), config, log_summary=False)

    assert document.metrics.stage_seconds["result_saving"] >= 0
    assert document.metrics.total_seconds >= document.metrics.save_seconds


def test_config_new_fields_and_overrides():
    cfg = load_config()
    assert cfg.engine.layout_model_name == "PP-DocLayoutV3"
    assert cfg.engine.layout_backend == "transformers"
    assert cfg.engine.layout_threshold is None
    assert DEFAULT_LAYOUT_THRESHOLD == 0.3
    assert PP_DOCLAYOUT_V3_ID2LABEL[5] == "display_formula"
    assert PP_DOCLAYOUT_V3_ID2LABEL[8] == "footer_image"
    assert PP_DOCLAYOUT_V3_ID2LABEL[12] == "header_image"
    assert PP_DOCLAYOUT_V3_ID2LABEL[15] == "inline_formula"
    assert PP_DOCLAYOUT_V3_ID2LABEL[23] == "vertical_text"
    assert cfg.predict.layout_shape_mode == "auto"
    assert cfg.predict.max_new_tokens is None

    cfg2 = cfg.apply_overrides(
        {
            "engine.layout_nms": False,
            "engine.vl_rec_api_key": "secret",
            "predict.max_new_tokens": 2048,
        }
    )
    assert cfg2.engine.layout_nms is False
    assert cfg2.engine.vl_rec_api_key == "secret"
    assert cfg2.predict.max_new_tokens == 2048


def test_layout_parsing_does_not_consume_sampling_options():
    from venusocr.pipeline.parsing import DocumentParsingPipeline

    pipeline = object.__new__(DocumentParsingPipeline)
    pipeline.layout_prep_cpu_workers = 0
    pipeline._prepare_page = lambda payload: (payload[0], [], [], False, set())
    pipeline._assemble_parsing_results = lambda *args: ([[]], [{}])
    pipeline.vl_recognizer = None
    options = {"min_pixels": 123, "max_pixels": 456, "max_new_tokens": None}

    pipeline.get_layout_parsing_results(
        images=[object()],
        layout_det_results=[{"boxes": []}],
        imgs_in_doc=[[]],
        vlm_kwargs=options,
    )

    assert options == {"min_pixels": 123, "max_pixels": 456, "max_new_tokens": None}


def test_pipeline_collects_stage_metrics_without_models():
    from types import SimpleNamespace

    import numpy as np

    from venusocr.pipeline.parsing import DocumentParsingPipeline

    pipeline = object.__new__(DocumentParsingPipeline)
    pipeline.config = SimpleNamespace(
        predict=SimpleNamespace(
            use_layout_detection=True,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_ocr_for_image_block=False,
            prompt_label=None,
            layout_shape_mode="auto",
            merge_layout_blocks=True,
            format_block_content=False,
            markdown_ignore_labels=[],
            repetition_penalty=None,
            temperature=None,
            top_p=None,
            min_pixels=None,
            max_pixels=None,
            max_new_tokens=None,
        )
    )
    batch = SimpleNamespace(
        instances=[np.zeros((10, 10, 3), dtype=np.uint8)],
        input_paths=["/tmp/a.png"],
        page_indexes=[None],
        page_counts=[None],
    )
    pipeline.batch_sampler = lambda input_value: iter([batch])
    pipeline.img_reader = lambda instances: instances
    pipeline.layout_detector = SimpleNamespace(
        predict=lambda images, **kwargs: [{"boxes": []}]
    )
    pipeline._process_vlm = lambda *args, **kwargs: ([], [])
    pipeline.layout_prep_cpu_workers = 0
    pipeline.last_metrics = None
    pipeline._active_metrics = None

    assert pipeline.predict("/tmp/a.png") == []
    assert pipeline.last_metrics.page_count == 1
    assert pipeline.last_metrics.page_batches == 1
    assert "input_loading" in pipeline.last_metrics.stage_seconds
    assert "layout_detection" in pipeline.last_metrics.stage_seconds


def test_layout_factory_rejects_unknown_backend():
    import pytest

    with pytest.raises(ValueError, match="Unsupported layout backend"):
        create_layout_detector(backend="unknown")


def test_venusocr_vlm_capabilities_are_restricted():
    import pytest

    from venusocr.config import VenusOCRConfig

    for field in (
        "use_chart_recognition",
        "use_seal_recognition",
        "use_ocr_for_image_block",
    ):
        with pytest.raises(ValueError, match="only supports text/formula/table"):
            VenusOCRConfig.model_validate({"predict": {field: True}})


def test_collect_vlm_entries_only_recognizes_text_formula_and_table():
    import numpy as np

    from venusocr.pipeline.parsing import DocumentParsingPipeline

    pipeline = object.__new__(DocumentParsingPipeline)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    blocks = [
        {"img": image.copy(), "box": [0, 0, 20, 20], "label": label}
        for label in ("text", "formula", "table", "chart", "seal", "spotting")
    ]
    cfg = {
        "image_labels": ["image", "header_image", "footer_image", "chart", "seal", "spotting"],
        "use_chart_recognition": False,
        "use_seal_recognition": False,
        "ocr_min_pixels": 1,
        "ocr_max_pixels": 2,
        "table_min_pixels": 3,
        "table_max_pixels": 4,
        "chart_min_pixels": 5,
        "chart_max_pixels": 6,
        "formula_min_pixels": 7,
        "formula_max_pixels": 8,
        "seal_min_pixels": 9,
        "seal_max_pixels": 10,
    }

    entries, has_spotting, _ = pipeline._collect_page_vlm_entries(0, blocks, [], cfg)

    assert [entry[3] for entry in entries] == [
        "Text Recognition:",
        "Formula Recognition:",
        "Table Recognition:",
    ]
    assert has_spotting is False


def test_gather_imgs_collects_all_visual_only_labels():
    import numpy as np

    from venusocr.pipeline.crop import gather_imgs

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    boxes = [
        {
            "label": label,
            "coordinate": [0, 0, 10, 10],
            "score": 0.9,
        }
        for label in ("image", "chart", "seal", "spotting", "header_image", "footer_image")
    ]

    assert [item["label"] for item in gather_imgs(image, boxes)] == [
        "image",
        "chart",
        "seal",
        "spotting",
        "header_image",
        "footer_image",
    ]


def test_transformers_layout_postprocess_preserves_schema_and_order():
    import numpy as np

    from venusocr.models.layout_postprocess import format_layout_results

    class TensorStub:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    raw = [
        {
            "scores": TensorStub([0.9, 0.8]),
            "labels": TensorStub([22, 21]),
            "boxes": TensorStub([[30.2, 20.2, 80.8, 50.9], [5.1, 5.2, 25.9, 18.8]]),
            "order_seq": TensorStub([2, 1]),
            "polygon_points": [None, np.asarray([[5, 5], [26, 5], [26, 19], [5, 19]])],
        }
    ]
    results = format_layout_results(
        raw,
        {21: "table", 22: "text"},
        [(100, 100)],
        layout_nms=False,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
    )

    assert [item["label"] for item in results[0]] == ["table", "text"]
    assert results[0][0]["coordinate"] == [5, 5, 26, 19]
    assert results[0][1]["polygon_points"].shape == (4, 2)


def test_transformers_layout_containment_matches_paddlex_threshold():
    import numpy as np

    from venusocr.models.layout_postprocess import _is_contained

    inner = np.asarray([22, 0.9, 0, 0, 100, 100, 0, 0], dtype=float)
    eighty_five_percent = np.asarray([21, 0.8, 15, 0, 115, 100, 1, 0], dtype=float)
    ninety_percent = np.asarray([21, 0.8, 10, 0, 110, 100, 1, 0], dtype=float)

    assert not _is_contained(eighty_five_percent, inner)
    assert _is_contained(ninety_percent, inner)


def test_transformers_layout_order_sort_is_stable():
    import numpy as np

    from venusocr.models.layout_postprocess import format_layout_results

    class TensorStub:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    raw = [{
        "scores": TensorStub([0.8, 0.9, 0.7]),
        "labels": TensorStub([22, 21, 22]),
        "boxes": TensorStub([[1, 1, 11, 11], [20, 1, 30, 11], [40, 1, 50, 11]]),
        "order_seq": TensorStub([1, 0, 1]),
    }]
    results = format_layout_results(
        raw,
        {21: "table", 22: "text"},
        [(100, 100)],
        layout_nms=False,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
    )

    assert [item["coordinate"][0] for item in results[0]] == [20.0, 1.0, 40.0]
