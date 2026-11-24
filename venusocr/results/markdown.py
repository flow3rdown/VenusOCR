# Copyright (c) 2026 VenusOCR. All Rights Reserved.
# Portions Copyright (c) PaddlePaddle Authors.
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

"""Markdown conversion rules for page-level parsing results."""

from __future__ import annotations

import copy
import re
from functools import partial


def _compile_title_pattern():
    numbering_pattern = (
        r"(?:" + r"[1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?|" + r"[\(\（](?:[1-9][0-9]*|["
        r"一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[\)\）]|" + r"["
        r"一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+"
        r"[、\.]?|" + r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)" + r")"
    )
    return re.compile(r"^\s*(" + numbering_pattern + r")(\s*)(.*)$")


TITLE_RE_PATTERN = _compile_title_pattern()


def format_title(block):
    """Normalize chapter title with '#' level indicator."""
    title = block.content
    match = TITLE_RE_PATTERN.match(title)
    if match:
        numbering = match.group(1).strip()
        title_content = match.group(3).lstrip()
        title = numbering + " " + title_content

    title = title.rstrip(".")
    level = (
        title.count(
            ".",
        )
        + 1
        if "." in title
        else 1
    )
    return _collapse_soft_newlines(f"#{'#' * level} {title}")


def format_para_title(block):
    """Normalize paragraph title, using title_level if available."""
    if getattr(block, "title_level", None) is None:
        return format_title(block)
    level = block.title_level
    title = block.content
    return _collapse_soft_newlines(f"#{'#' * level} {title}")


def format_centered_by_html(content, collapse_newlines=True):
    """Wrap *content* in an HTML center-aligned div."""
    if collapse_newlines:
        content = _collapse_soft_newlines(content)
    return f'<div style="text-align: center;">{content}</div>' + "\n"


def format_text_plain(block):
    """Return the block's raw text content without any transformation."""
    return block.content


def format_image_scaled_by_html(block, original_image_width, show_ocr_content=False):
    """Render an image block as a width-scaled HTML ``<img>`` tag."""
    img_tags = []
    if block.image is None:
        return ""
    image_path = block.image["path"]
    image_width = block.bbox[2] - block.bbox[0]
    scale = int(image_width / original_image_width * 100)
    img_tags.append(
        '<img src="{}" alt="Image" width="{}%" />'.format(
            _collapse_soft_newlines(image_path), scale
        ),
    )
    image_info = "\n".join(img_tags)
    if show_ocr_content:
        ocr_content = block.content
        image_info += "\n\n" + ocr_content + "\n\n"
    return image_info


def format_image_plain(block, show_ocr_content=False):
    """Render an image block as a Markdown image reference ``![](path)``."""
    img_tags = []
    if block.image:
        image_path = block.image["path"]
        img_tags.append("![]({})".format(_collapse_soft_newlines(image_path)))
        image_info = "\n".join(img_tags)
        if show_ocr_content:
            ocr_content = block.content
            image_info += "\n\n" + ocr_content + "\n\n"
        return image_info
    return ""


def format_chart2html_table(block):
    """Chart -> HTML table."""
    lines_list = block.content.split("\n")
    header = lines_list[0].split("|")
    rows = [line.split("|") for line in lines_list[1:]]
    html = "<table border=1 style='margin: auto; width: max-content;'>\n"
    html += (
        "  <thead><tr>"
        + "".join(
            f"<th style='text-align: center;'>{cell.strip()}</th>" for cell in header
        )
        + "</tr></thead>\n"
    )
    html += "  <tbody>\n"
    for row in rows:
        html += (
            "    <tr>"
            + "".join(
                f"<td style='text-align: center;'>{cell.strip()}</td>" for cell in row
            )
            + "</tr>\n"
        )
    html += "  </tbody>\n"
    html += "</table>"
    return html


def simplify_table(table_code):
    """Strip ``<html>`` and ``<body>`` wrapper tags from a table HTML string."""
    return "\n" + table_code.replace("<html>", "").replace("</html>", "").replace(
        "<body>", ""
    ).replace("</body>", "")


def format_first_line(block, templates, format_func, splitter):
    """Format the first non-empty line of a block if it matches a template."""
    lines = block.content.split(splitter)
    for idx in range(len(lines)):
        line = lines[idx]
        if line.strip() == "":
            continue
        if line.lower() in templates:
            lines[idx] = format_func(line)
        break
    return splitter.join(lines)


def format_table_center(block):
    """Add center styling to table HTML."""
    table_content = block.content
    table_content = table_content.replace(
        "<table>", "<table border=1 style='margin: auto; word-wrap: break-word;'>"
    )
    table_content = table_content.replace(
        "<th>", "<th style='text-align: center; word-wrap: break-word;'>"
    )
    table_content = table_content.replace(
        "<td>", "<td style='text-align: center; word-wrap: break-word;'>"
    )
    return table_content


def merge_formula_and_number(formula, formula_number):
    """Merge a formula and its formula number for display."""
    formula = formula.replace("$$", "")
    merge_formula = r"{} \tag*{{{}}}".format(formula, formula_number)
    return f"$${merge_formula}$$"


def _collapse_soft_newlines(s: str) -> str:
    """Collapse soft-hyphen line breaks and newlines into spaces."""
    return s.replace("-\n", "").replace("\n", " ")


def _format_normalize_newlines(block):
    """Normalize double newlines to single, then single to double for markdown spacing."""
    return block.content.replace("\n\n", "\n").replace("\n", "\n\n")


def build_handle_funcs_dict(
    *,
    text_func,
    image_func,
    chart_func,
    table_func,
    formula_func,
    seal_func,
):
    """Build a dictionary mapping block labels to their formatting functions.

    Returns:
        dict: A mapping from block label to handler function.
    """
    return {
        "paragraph_title": format_para_title,
        "abstract_title": format_title,
        "reference_title": format_title,
        "content_title": format_title,
        "doc_title": lambda block: _collapse_soft_newlines(f"# {block.content}"),
        "table_title": text_func,
        "figure_title": text_func,
        "chart_title": text_func,
        "vision_footnote": _format_normalize_newlines,
        "text": _format_normalize_newlines,
        "ocr": _format_normalize_newlines,
        "vertical_text": _format_normalize_newlines,
        "reference_content": _format_normalize_newlines,
        "abstract": partial(
            format_first_line,
            templates=["摘要", "abstract"],
            format_func=lambda l: f"## {l}\n",
            splitter=" ",
        ),
        "content": lambda block: block.content.replace("-\n", "  \n").replace(
            "\n", "  \n"
        ),
        "image": image_func,
        "chart": chart_func,
        "formula": formula_func,
        "display_formula": formula_func,
        "inline_formula": formula_func,
        "table": table_func,
        "reference": partial(
            format_first_line,
            templates=["参考文献", "references"],
            format_func=lambda l: f"## {l}",
            splitter="\n",
        ),
        "algorithm": lambda block: block.content.strip("\n"),
        "seal": seal_func,
        "spotting": image_func,
        "number": format_text_plain,
        "footnote": format_text_plain,
        "header": format_text_plain,
        "header_image": image_func,
        "footer": format_text_plain,
        "footer_image": image_func,
        "aside_text": format_text_plain,
    }


class MarkdownConverter:
    """Converts a list of document blocks into Markdown text + images."""

    @staticmethod
    def convert(
        blocks,
        *,
        handle_funcs_dict,
        show_formula_number=False,
        imgs_in_doc=None,
    ) -> dict:
        """Convert *blocks* to Markdown.

        Args:
            blocks: Iterable of block objects (``label``, ``content``, ``bbox``,
                ``image``).
            handle_funcs_dict: ``{label: formatting_func}`` mapping built by
                :func:`build_handle_funcs_dict`.
            show_formula_number: If *True*, merge adjacent formula + number blocks.
            imgs_in_doc: Extra images to include (list of ``{"path", "img"}``).

        Returns:
            dict with keys ``markdown_texts`` and ``markdown_images``.
        """
        blocks_list = list(blocks)  # ensure indexable for lookahead

        markdown_content = ""
        markdown_images: dict = {}
        for idx, block in enumerate(blocks_list):
            label = block.label

            # --- collect images ---
            if block.image is not None:
                markdown_images[block.image["path"]] = block.image["img"]

            # --- formula-number merging ---
            if (
                show_formula_number
                and label in ("display_formula", "formula")
                and idx < len(blocks_list) - 1
            ):
                next_block = blocks_list[idx + 1]
                if next_block.label == "formula_number":
                    block = copy.copy(block)
                    block.content = merge_formula_and_number(
                        block.content, next_block.content
                    )

            # --- format & append ---
            handle_func = handle_funcs_dict.get(label, None)
            if handle_func:
                markdown_content += (
                    "\n\n" + handle_func(block)
                    if markdown_content
                    else handle_func(block)
                )
        result = {
            "markdown_texts": markdown_content,
            "markdown_images": markdown_images,
        }

        if imgs_in_doc:
            for img in imgs_in_doc:
                result["markdown_images"][img["path"]] = img["img"]

        return result
