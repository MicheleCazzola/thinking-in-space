<div align="center" style="font-family: charter;">

<h1><img src="docs/resources/icons/brain.ico" width="4%"/><i>Thinking in Space</i>:</br> How Multimodal Large Language Models See, Remember and Recall Spaces</h1>

<img src="docs/resources/preview.png" width="50%"/>
<br />

<a href="https://arxiv.org/abs/2412.14171" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-thinking--in--space-red?logo=arxiv" height="20" />
</a>
<a href="https://vision-x-nyu.github.io/thinking-in-space.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-thinking--in--space-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/datasets/nyu-visionx/VSI-Bench" target="_blank">
    <img alt="HF Dataset: VSI-Bench" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-VSI--Bench-ffc107?color=ffc107&logoColor=white" height="20" />
</a>

<div>
    <a href="https://jihanyang.github.io/" target="_blank">Jihan Yang</a><sup>1*</sup>,</span>
    <a href="https://github.com/vealocia" target="_blank">Shusheng Yang</a><sup>1*</sup>, </span>
    <a href="https://anjaliwgupta.com/" target="_blank">Anjali W. Gupta</a><sup>1*</sup>,</span>
    <a href="https://rilynhan.github.io" target="_blank">Rilyn Han</a><sup>2*</sup>,</span>
    <a href="https://profiles.stanford.edu/fei-fei-li" target="_blank">Li Fei-Fei</a><sup>3</sup>,</span>
    <a href="https://www.sainingxie.com/" target="_blank">Saining Xie</a><sup>1</sup>,</span>
</div>

<div>
    <sup>1</sup>New York University&emsp;
    <sup>2</sup>Yale University&emsp;
    <sup>3</sup>Stanford University&emsp;
</div>

<img src="docs/resources/teaser.png" width="100%"/>
<p align="justify"><i>Whether at home, in the workplace, or elsewhere, the ability to perceive a space, remember its layout, and retrieve this spatial information to answer questions on demand is a key aspect of visual-spatial intelligence. Recent Multimodal LLMs can understand general videos, but can they "think spatially" when presented with a video recording of an environment? Can they build an accurate, implicit "cognitive map" that allows them to answer questions about a space? What are the strengths and limitations of using MLLMs to enhance spatial intelligence? We dig into these questions by setting up video data for MLLMs to watch, building a VQA benchmark to check their recall, and examining what the MLLMs actually remember and understand.</i></p>

</div>

## Release
- `2025-08-05` :hearts: We release the meta information used in our VSIBench! Check out at `/data/meta_info`!
- `2025-02-27` :hearts: Our paper `Thinking-in-Space` is accepted by CVPR 2025 as Oral! See u in Nashville!
- `2024-12-19` :rocket: We released our VSI-Bench and corresponding evaluation code.

## Contents

- [Release](#release)
- [Contents](#contents)
- [VSI-Bench](#vsi-bench)
- [Results](#results)
- [RUN Your Own Evaluation](#run-your-own-evaluation)
  - [Benchmark](#benchmark)
  - [Installation](#installation)
  - [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## VSI-Bench

**Overview:** We introduce VSI-Bench, a benchmark designed to evaluate the visual-spatial intelligence of Multimodal LLMs (MLLMs). VSI-Bench comprises over 5,000 question-answer pairs derived from 288 egocentric videos sourced from the validation sets of public indoor 3D scene reconstruction datasets ScanNet, ScanNet++, and ARKitScenes.
<img src="docs/resources/task-demo.png" width="100%"/>

**VSI-Bench** includes eight tasks categorized into three types: configurational, measurement estimation, and spatiotemporal. Iteratively refined for quality, VSI-Bench provides a foundational resource for studying the connection between MLLMs and 3D reconstruction.
<img src="docs/resources/benchmark-stats.png" width="100%"/>


## Results

**Evaluation Setups:** We benchmarked 15 video-supporting MLLMs from diverse model families. For proprietary models, we include Gemini-1.5 and GPT-4o. For open-source models, we evaluate models from InternVL2, ViLA, LongViLA, LongVA, LLaVA-OneVision, and LLaVA-NeXT-Video. All evaluations are conducted in zero-shot settings with default prompts and greedy decoding for reproducibility. Tasks are evaluated using either accuracy for Multiple-Choice Answer (MCA) tasks or our proposed Mean Relative Accuracy (MRA) for Numerical Answer (NA) tasks.

<img src="docs/resources/eval-bench.png" width="100%"/>

## RUN Your Own Evaluation

### Benchmark

Our benchmark is hosted on [HuggingFace](https://huggingface.co/datasets/nyu-visionx/VSI-Bench). You can simply access the benchmark data using the following code.
```python
# NOTE: pip install datasets

from datasets import load_dataset
vsi_bench = load_dataset("nyu-visionx/VSI-Bench")
print(dataset)
```

### Installation

```bash
conda create --name vsibench python=3.10
conda activate vsibench

git clone git@github.com:vision-x-nyu/thinking-in-space.git
cd thinking-in-space

git submodule update --init --recursive

cd transformers && pip install -e . && cd ..

pip install -e .
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
pip install deepspeed
```

### Evaluation

We provide a all-in-one evaluation scripts. You can simply run the following code to start your evaluation.

```bash
bash evaluate_all_in_one.sh --model all --num_processes 8 --benchmark vsibench
```

> Note: The evaluation results for open-source models may differ slightly from our tables due to additional data refinement. We will update the tables and our paper soon.

For local evaluation, use `evaluate_vsi.sh` and select exactly one benchmark:
`vsi-bench`, `vsi-bench-tiny`, or `vsi-bench-debiased`. The `vsi-bench-tiny`
task uses the official 400 question IDs published by the authors. Eight of
those IDs were removed in the later dataset release, so the current Hugging
Face data yields 392 available questions; do not use `--limit 400`.
The script supports one selected local model
without an API key or one Gemini model using the ignored `.google_api_key` file in
the repository root. Put the Google API key on a single line in that file:

```bash
printf '%s\n' 'YOUR_GOOGLE_API_KEY' > .google_api_key

conda activate vsibench
python -m pip install -e ".[gemini]"

bash evaluate_vsi.sh --model llava_one_vision_qwen2_0p5b_ov --benchmark vsi-bench \
  --num_frames 2 --limit 20 --data_dir data/VSI-Bench

bash evaluate_vsi.sh --model gemini_1p5_flash --benchmark vsi-bench-debiased \
  --limit 20

bash evaluate_vsi.sh --model gemini-3.6-flash --benchmark vsi-bench-tiny \
  --no-thinking --no-context-caching --num_processes 1 \
  --launcher python --output_path logs/reproduction/vsibench-tiny
```

The last command follows the ReVSI evaluation protocol (arXiv 2604.24300,
Appendix) for proprietary models on VSI-Bench: lmms-eval with greedy decoding
(`temperature=0`, `top_p=1`, `top_k=1`), the official VSI-Bench tiny set, the
VSI-Bench prompt template for proprietary models (`These are frames of a
video.` + question + the numerical / multiple-choice post-prompt), and the
whole video uploaded to Gemini so that Gemini's default FPS-based sampling
(1 FPS) is used. For this protocol do not pass `--resolution_mode`: the paper
uses Gemini's default media resolution.

#### Running Gemini through Vertex AI (e.g. from a Google Cloud VM)

`--backend vertex` calls Gemini through Vertex AI with Application Default
Credentials (on a Compute Engine VM: its service account), so no API key is
needed and usage is billed to the Google Cloud project. Vertex has no Files API,
so videos are passed to Gemini by GCS URI: `gs://BUCKET/PREFIX/<dataset>/<scene>.mp4`.
`--gcs_prefix` (default `vsibench`) is the folder in the bucket that holds the
videos. A video that already exists there is used as is and never overwritten;
only a missing one is uploaded from `--data_dir`. If the dataset is already in
the bucket (for example `gs://BUCKET/data/VSI-Bench/scannet/...`), pass
`--gcs_prefix data/VSI-Bench` and nothing is uploaded. The local videos in
`--data_dir` must still exist, because the task checks them. Gemini still
receives the whole video and samples it at its default 1 FPS.

```bash
# one-time setup (VM access scope: cloud-platform; service account roles:
# Vertex AI User + Storage Object Viewer on the bucket, or Storage Object Admin
# if videos must be uploaded)
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
python -m pip install -e ".[gemini]"

bash evaluate_vsi.sh --model gemini-3.8-flash --benchmark vsi-bench-tiny \
  --backend vertex --gcp_project YOUR_PROJECT --gcs_bucket YOUR_BUCKET \
  --gcs_prefix data/VSI-Bench --data_dir data/VSI-Bench \
  --no-thinking --no-context-caching --launcher python \
  --resume_dir logs/revsi-protocol/gemini-3.8-flash-responses \
  --output_path logs/revsi-protocol/gemini-3.8-flash
```

`--gcp_location` defaults to `global`. Uploads (Files API or GCS) and generation
requests are retried with exponential backoff (15 s doubling, capped at 300 s,
8 attempts; `GEMINI_MAX_ATTEMPTS`, `GEMINI_RETRY_BASE_SECONDS`,
`GEMINI_RETRY_MAX_SECONDS`) on 429/5xx, timeouts and dropped connections; each
HTTP request times out after `GEMINI_REQUEST_TIMEOUT` seconds (default 300) of
inactivity, and every single API call (generation, upload, polling) is also
abandoned and retried after a total wall-clock deadline of `GEMINI_CALL_DEADLINE`
seconds (default: same as `GEMINI_REQUEST_TIMEOUT`; `0` disables it).
Questions that still get no answer are listed in an error at the end of the
run, because they are scored as wrong. With `--resume_dir`, answers are saved
as they arrive and re-running the same command only redoes the missing ones.
The per-question metadata (see below) is saved next to them in
`<model>_metadata.json`, so resumed answers keep it.

Gemini thinking is disabled by default (`--no-thinking`): requests are sent with
`thinking_budget=0`. If a model rejects a zero budget, the backend logs a
warning and falls back to `thinking_level="minimal"`. Setting
`GEMINI_THINKING_OFF_MODE=minimal` requests `thinking_level="minimal"` directly.
Use `--thinking` to restore the model's default thinking behaviour.
Some models cannot turn thinking off completely. `gemini-3.8-flash` on Vertex
accepts `thinking_budget=0` but still thinks: in local tests it used about
900–1200 thinking tokens per answer on average, and rejects `minimal` with
`400 Thinking level is unsupported`. Thinking tokens are billed as output.
The token usage of every answer is logged (`Usage doc N: prompt=… output=…
thoughts=…`), with a total at the end of the run.

Even with greedy decoding the answers of these models are not fully
deterministic: in repeated `--limit 5` runs the same question got different
numerical answers (212 vs 205) and different numbers of thinking tokens.

#### VSI-Bench result files

Each run writes two JSON files in `--output_path/<task>/<date>_<model>.../`:

- `<task>.json` holds one entry per question. Its `doc` contains the question
  (`id`, `dataset`, `scene_name`, `question_type`, `question`, `options`), the
  `ground_truth`, the raw model answer (`prediction`), the value actually
  compared with the ground truth (`parsed_prediction`; `null` when a numerical
  answer cannot be read as a number, which scores 0), and the per-question
  metric (`MRA:.5:.95:.05` for numerical questions, `accuracy` for
  multiple-choice ones). For API models, `response_meta` holds the API status
  (`ok`, `api_error` or `upload_error`: no answer because of the API, scored as
  wrong), the number of attempts, the last error, the token counts
  (`prompt_tokens`, `output_tokens`, `thoughts_tokens`), the `finish_reason`
  and the thinking mode.
- `results.json` holds the official VSI-Bench scores: one value per question
  type and `overall`, the mean of the question types, together with the
  `tabulated_keys` and `tabulated_results` row. These values are computed
  exactly as in the official code. The extra key `details` adds:
  - `n_samples` and `n_per_question_type`;
  - `object_rel_direction_subtypes_accuracy`: the easy, medium and hard
    accuracies, whose mean is the official `object_rel_direction_accuracy`;
  - `mca_mean_accuracy` and `na_mean_MRA`: the means over the multiple-choice
    and the numerical question types;
  - `per_dataset`: the same scores for arkitscenes, scannet and scannetpp;
  - `n_unparsable_numerical_answers`;
  - `api`: the status counts, the questions with no answer because of the API,
    the answers taken from the resume cache, the thinking modes and the token
    totals.

The MRA is computed as in the official code, so the scores are comparable with
the published ones. That code compares the relative error with thresholds from
`np.linspace`, and floating-point rounding makes a few thresholds slightly
smaller than their decimal value: `1 - 0.8` gives `0.19999999999999996`. An
answer whose relative error is exactly 20% or 45% therefore scores 0.1 less
than with exact thresholds (for example ground truth 5 and answer 4 or 6 scores
0.6 instead of 0.7). This is kept on purpose.

Gemini context caching is disabled by default. Enable it explicitly when the
selected model and account support it:

```bash
bash evaluate_vsi.sh --model gemini-3-flash-preview \
  --benchmark vsi-bench --context-caching --limit 20
```

Gemini video resolution can be selected with `--resolution_mode`:

```bash
bash evaluate_vsi.sh --model gemini-3.6-flash \
  --benchmark vsi-bench --limit 20 --resolution_mode high
```

Supported values are `low`, `medium`, `high`, and `ultra_high`. The option is
passed to the Gemini API as `media_resolution`; it applies only to Gemini
models.

Use `--no-context-caching` to disable it explicitly. Without either flag, the
script uses only the upload cache and does not call the Gemini context-caching
API.

Gemini models can also be selected by passing their Google model ID directly.
The wrapper accepts IDs from every Gemini generation, including preview IDs:

```bash
bash evaluate_vsi.sh --model gemini-2.5-flash --benchmark vsi-bench --limit 20
bash evaluate_vsi.sh --model gemini-2.5-pro --benchmark vsi-bench-debiased --limit 20
bash evaluate_vsi.sh --model gemini-3-flash-preview --benchmark vsi-bench --limit 20
bash evaluate_vsi.sh --model models/gemini-2.5-flash --benchmark vsi-bench --limit 20
```

The model ID is passed unchanged as `model_version` to the Gemini backend.
The model must exist and support video input for the Google API key and SDK
installed in the environment.

### Gemini models available for the current API key

The following is the complete snapshot returned by `client.models.list()` for
the API key in `.google_api_key`, queried on 2026-11-03. Names below omit the
optional `models/` prefix. Availability and capabilities can change over time.
All models in the first list expose `generateContent`; that does not by itself
guarantee that a model accepts VSI-Bench video input.

```text
gemini-2.5-flash
gemini-2.5-pro
gemini-2.5-flash-preview-tts
gemini-2.5-pro-preview-tts
gemini-flash-latest
gemini-flash-lite-latest
gemini-pro-latest
gemini-2.5-flash-lite
gemini-2.5-flash-image
gemini-3-flash-preview
gemini-3.1-pro-preview
gemini-3.1-pro-preview-customtools
gemini-3.1-flash-lite-preview
gemini-3.1-flash-lite
gemini-3-pro-image-preview
gemini-3-pro-image
gemini-3.1-flash-image-preview
gemini-3.1-flash-image
gemini-3.1-flash-lite-image
gemini-3.5-flash
gemini-3.5-flash-lite
gemini-omni-flash-preview
gemini-omni-1.1-flash
gemini-3.5-transcribe
gemini-3.6-flash
gemini-3.7-flash
gemini-3.8-flash
gemini-4-argon
gemini-2.5-computer-use-preview-10-2025
gemini-robotics-er-2-preview
```

The same API key also exposes these Gemini-family models with non-standard
endpoints or non-generation methods:

```text
gemini-embedding-001
gemini-embedding-2-preview
gemini-embedding-2
gemini-3.5-transcribe-live
gemini-2.5-flash-native-audio-latest
gemini-2.5-flash-native-audio-preview-09-2025
gemini-2.5-flash-native-audio-preview-12-2025
gemini-3.1-flash-live-preview
gemini-3.8-live
gemini-3.8-live-extended-thinking
gemini-3.5-live-translate-preview
gemini-robotics-er-2-streaming-preview
```

To refresh this list without exposing the key, run:

```powershell
$env:GOOGLE_API_KEY = (Get-Content -Raw .google_api_key).Trim()
conda run --no-capture-output -n vsibench python -c "from google import genai; import os; c=genai.Client(api_key=os.environ['GOOGLE_API_KEY']); [print(m.name, getattr(m, 'supported_actions', None)) for m in c.models.list()]"
```

The Gemini backend uses `max_output_tokens=8192`, as in the original lmms-eval
Gemini setup (a tight cap truncated responses and produced empty answers;
empty responses are now logged with their `finish_reason`). Uploaded videos are cached locally in memory for the run,
and transient API failures are retried up to four times with exponential
backoff. It also creates a one-hour Gemini
context cache per video when supported by the installed SDK and selected model.
If context caching is unavailable, the run continues with the upload cache and
logs an explicit warning.

On Windows/Conda, if `SSL_CERT_FILE` points to a missing certificate bundle,
the Gemini backend automatically uses the installed `certifi` bundle while
keeping TLS certificate verification enabled.

## Limitations

We strive to maintain the highest quality in our benchmark, but some imperfections may persist. If you notice any, we encourage you to reach out and share your valuable feedback!

## Acknowledgement

Our evaluation code is build upon [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We acknowledge their team for providing this excellent toolkit for evaluating multimodal large language models.

## Citation

If you find our paper and code useful in your research, please consider giving us a star :star: and citing our work :pencil: :)
```
@article{yang2024think,
    title={{Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces}},
    author={Yang, Jihan and Yang, Shusheng and Gupta, Anjali and Han, Rilyn and Fei-Fei, Li and Xie, Saining},
    year={2024},
    journal={arXiv preprint arXiv:2412.14171},
}
```
