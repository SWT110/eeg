# Third-Party Notices for Comparison Models

`comparison_models.py` is a plain-PyTorch adaptation of model implementations
from the official TCFormer repository:

- Repository: <https://github.com/Altaheri/TCFormer>
- Adapted commit: `701a361640708ef1cc07cf22660d47edc6d9bf3c`
- Upstream files: `models/eegnet.py`, `models/shallownet.py`,
  `models/atcnet.py`, `models/tcformer.py`, `models/modules.py`, and
  `models/channel_group_attention.py`

The upstream repository is distributed under the following license.

```text
MIT License

Copyright (c) 2025 Hamdi Altaheri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Scientific use of each model also requires citation of the corresponding paper;
the DOI and title are stored in `COMPARISON_MODEL_METADATA` and in the experiment
protocol artifacts.

## Additional architecture benchmark adapters

`benchmark_models.py` provides task-specific PyTorch implementations of nine
additional architecture families. Paper links, reference repositories, pinned
commits, configurations and adaptation notes are recorded in
`benchmark_vendor/sources.json`.

- DeepConvNet/Deep4 and EEG-ITNet use the Braindecode BSD-3-Clause implementation
  as their reference. The complete notice is retained in
  `benchmark_vendor/LICENSE-braindecode.txt`. The EEG-ITNet author's restricted
  notebook is not included in this repository.
- The TSception-related adaptation retains the CBCR notice in
  `benchmark_vendor/LICENSE-TSception-CBCR.txt`; its upstream terms permit
  redistribution and modification for non-commercial purposes. This benchmark
  is intended for non-commercial academic research.
- The CTNet reference notice is retained in `benchmark_vendor/LICENSE-CTNet.txt`.
- EEG-TCNet, FBCNet, EEGNeX, EEGSimpleConv and EEG-Deformer architecture references
  and explicit task adaptations are attributed in `sources.json`. Their full
  upstream training repositories, datasets and pretrained weights are not
  vendored here.

These adapters do not claim exact reproduction of every author's preprocessing,
training recipe or published benchmark result. Cite the relevant papers when
using the corresponding architecture.
