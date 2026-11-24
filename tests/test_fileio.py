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

"""Tests for venusocr.utils.fileio."""

import pytest

from venusocr.data.sampler import ImageBatchSampler
from venusocr.utils.fileio import collect_input_files, output_name_for, output_path_for


def test_single_file(tmp_path):
    f = tmp_path / "a.pdf"
    f.touch()
    paths, root = collect_input_files(str(f))
    assert paths == [str(f.absolute())]
    assert root is None


def test_unsupported_suffix(tmp_path):
    f = tmp_path / "a.txt"
    f.touch()
    with pytest.raises(ValueError):
        collect_input_files(str(f))


def test_directory_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.png").touch()
    (tmp_path / "sub" / "b.pdf").touch()
    (tmp_path / "c.txt").touch()
    paths, root = collect_input_files(str(tmp_path))
    assert len(paths) == 2
    assert root == str(tmp_path.absolute())


def test_empty_directory(tmp_path):
    with pytest.raises(ValueError):
        collect_input_files(str(tmp_path))


def test_missing_path():
    with pytest.raises(ValueError):
        collect_input_files("/nonexistent/path/xyz")


def test_output_path_mirrors_tree(tmp_path):
    root = tmp_path / "in"
    (root / "sub").mkdir(parents=True)
    f = root / "sub" / "a.pdf"
    f.touch()
    out = output_path_for(str(root.absolute()), str(f.absolute()), "/out")
    assert out == "/out/sub"
    # top-level file stays in the base output dir
    f2 = root / "b.pdf"
    f2.touch()
    assert output_path_for(str(root.absolute()), str(f2.absolute()), "/out") == "/out"
    # single file input
    assert output_path_for(None, str(f.absolute()), "/out") == "/out"


def test_sampler_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        ImageBatchSampler(batch_size=0)


def test_output_names_keep_suffix_for_same_stem_documents(tmp_path):
    pdf = tmp_path / "sample.pdf"
    image = tmp_path / "sample.png"
    pdf.touch()
    image.touch()
    assert output_name_for(str(pdf)) == "sample.pdf"
    assert output_name_for(str(image)) == "sample.png"


def test_output_name_ignores_unsupported_siblings(tmp_path):
    document = tmp_path / "sample.pdf"
    document.touch()
    (tmp_path / "sample.txt").touch()
    assert output_name_for(str(document)) == "sample"


def test_output_names_avoid_dotted_filename_collisions(tmp_path):
    paths = [tmp_path / name for name in ["sample.pdf", "sample.png", "sample.pdf.png"]]
    for path in paths:
        path.touch()
    assert len({output_name_for(str(path)) for path in paths}) == len(paths)


def test_sampler_rejects_unsupported_input(tmp_path):
    unsupported = tmp_path / "data.txt"
    unsupported.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        list(ImageBatchSampler().sample(str(unsupported)))


def test_sampler_rejects_unsupported_object():
    with pytest.raises(TypeError):
        list(ImageBatchSampler().sample(object()))
