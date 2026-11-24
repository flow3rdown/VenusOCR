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

"""Minimal example: parse one document with the VenusOCR Python API.

Prerequisites:
    1. A running VenusOCR vLLM server, e.g.:
         venusocr serve /path/to/venusocr-checkpoint --port 8000
    2. VenusOCR installed with one layout backend:
         pip install 'venusocr[paddlex]'
       or:
         pip install 'venusocr[transformers]'
"""

import argparse

from venusocr import VenusOCR


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse one document with VenusOCR.")
    parser.add_argument("--input", required=True, help="Path to a PDF or document image")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--layout-model-dir")
    parser.add_argument("--output", default="./output")
    args = parser.parse_args()
    overrides = {"engine.vl_rec_server_url": args.server_url}
    if args.layout_model_dir:
        overrides["engine.layout_model_dir"] = args.layout_model_dir
    with VenusOCR(overrides=overrides) as ocr:
        result = ocr.parse(args.input)
        result.save(args.output)
        print(result.markdown)


if __name__ == "__main__":
    main()
