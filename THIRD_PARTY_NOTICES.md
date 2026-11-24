# Third-party notices

VenusOCR includes modified implementations derived from Apache-2.0 licensed
PaddleX components. The derived code is maintained locally so the public
document pipeline does not require the PaddleOCR Python package.

## PaddleX

- Project: PaddleX
- Copyright: PaddlePaddle Authors
- License: Apache License 2.0
- Compatibility source version: 3.6.0
- Upstream project: <https://github.com/PaddlePaddle/PaddleX>

Derived areas include document input reading, layout-region cropping and
merging, PaddleOCR-VL orchestration, OTSL table conversion, recognition client
request construction, result serialization and Markdown formatting.

VenusOCR modifies and reorganizes these components, removes unused export and
pipeline variants, and adds its own public API, configuration, batching and
optional Transformers layout backend. See `agent.md` for the source mapping.

The full Apache License 2.0 text is available in the repository `LICENSE` file.
