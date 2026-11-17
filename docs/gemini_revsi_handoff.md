# Handoff: Gemini Flash evaluation on VSI-Bench-tiny (ReVSI protocol)

Status as of 2026-11-17, 13:10 CET. Branch: `adapt-cpu`. **Nothing is committed yet.**
All changes listed below are in the working tree only.

## 1. Goal

Run online inference with **Gemini 3.8 Flash** on **VSI-Bench tiny** using the
**same evaluation setup as the ReVSI paper** (arXiv 2604.24300). Thinking is
**disabled**. The final run happens on a **Google Cloud VM**, billed to the
Google Cloud free-trial credits through **Vertex AI**.

Standing rules from the user:
- Keep **both Gemini backends**: Google AI Studio (API key) and Vertex AI. Both
  must be selectable from the CLI (`--backend api|vertex`).
- **Always test locally with `--limit 5` before anything runs on the cloud**,
  for each backend that will be used.
- Do not interrupt the user's running evaluations.

## 2. ReVSI protocol and how this repo matches it

Source: the ReVSI appendix ("For proprietary models…") and Table 8.

| ReVSI protocol | This repo |
|---|---|
| lmms-eval framework, greedy decoding | `temperature=0, top_p=1, top_k=1` (`_build_generation_config`) |
| Official VSI-Bench **tiny** set for proprietary models | task `vsibench_tiny`, using the 400 official ids in `lmms_eval/tasks/vsibench/vsibench_tiny_ids.txt` |
| Gemini 3: whole video, Gemini's default FPS-based sampling (1 FPS), no fixed frame count | whole mp4 uploaded (Files API or GCS URI) |
| VSI-Bench proprietary prompt: `[VIDEO][PRE][QUESTION][POST]`, pre = "These are frames of a video." | `vsibench_doc_to_text` default pre-prompt plus the `gemini_api` post-prompts in the yaml |
| Metrics: MRA (numerical), accuracy (MCA), official aggregation | `utils.py` metrics are unchanged from upstream; the upstream changes only add per-task logging |
| Media resolution: not specified, so the default is assumed | do **not** pass `--resolution_mode` |

Known deviations and open choices:
- **392 questions, not 400.** 8 of the official tiny ids were removed from the
  current HF dataset release. The paper does not say which dataset revision it
  used.
- **Numerical post-prompt wording.** The official code says "Do not response
  anything other than a single number!" (with the typo). The ReVSI Table 8
  prints "Do not respond with anything other than a single number!". The repo
  **keeps the official code string**. This is a one-line yaml change if the
  user prefers the paper's wording; it has not been decided yet.
- **Thinking disabled** is the user's choice. The paper does not specify it.
- `max_output_tokens=8192`, the same as the original lmms-eval Gemini code. The
  paper does not specify it.

## 3. Root-cause findings

- **Empty answers in earlier runs.** The earlier code capped output at
  `max_output_tokens=32`. Gemini 3.x thinking tokens count against that cap, so
  responses hit MAX_TOKENS and `response.text` was `None`, which was saved as
  `""`. Logs showed 5/10 empty (`logs/gemini-3.6-flash-vsibench-lim10`) and 3/4
  empty (`logs/gemini-3.6-flash-low-lim4`), with no errors logged. This is fixed.
- **`timeout` was unused.** The `timeout` parameter was never passed to the
  client, so stalled requests could hang forever. It is now passed
  (`HttpOptions(timeout=…)`). **However, see the open bug in §7.1.**
- **Video upload failures were silent.** When an upload failed (for example
  with a 503), the question was skipped and scored as wrong, with no retry.
  This is fixed (§4).
- **CRLF line endings.** `evaluate_vsi.sh` had CRLF endings, which break bash on
  Linux. It was converted to LF, and `.gitattributes` (`*.sh text eol=lf`) was
  added.

## 4. Code changes (uncommitted)

### `lmms_eval/models/gemini_api.py` (uses the `google-genai` SDK)

- **New args:**
  - `thinking=False`, `max_output_tokens=8192`.
  - `timeout` (seconds; env `GEMINI_REQUEST_TIMEOUT`, default 300).
  - `backend` (`api` | `vertex`; env `GEMINI_BACKEND`).
  - `gcp_project`, `gcp_location` (default `global`), `gcs_bucket`,
    `gcs_prefix` (default `vsibench`).
- **Thinking off.** Requests send `ThinkingConfig(thinking_budget=0)`. If the
  API answers 400 mentioning thinking, the code falls back once to
  `thinking_level="minimal"` and logs a warning. A warning is also logged once
  if `thoughts_token_count > 0`.
- **Empty responses** are logged with `finish_reason`, the thought token count
  and `prompt_feedback`.
- **Retries.** `_call_with_retries` and `_retry_delay` give exponential backoff
  of 15 s doubling, capped at 300 s, with 8 attempts. They are configured by
  the env vars `GEMINI_MAX_ATTEMPTS`, `GEMINI_RETRY_BASE_SECONDS` and
  `GEMINI_RETRY_MAX_SECONDS`.
  - Retryable errors: 429/5xx, plus httpx timeout, connect and
    remote-protocol errors.
  - The same retries apply to **generation** and to **video upload**.
  - Polling the file state tolerates transient errors.
  - A file stuck in PROCESSING raises `_TransientUploadError`, which is
    retried.
- **Vertex backend.** The client is created with
  `genai.Client(vertexai=True, project, location)` using ADC. Vertex has no
  Files API, so each video is uploaded once to
  `gs://BUCKET/vsibench/<dataset>/<scene>.mp4`. The upload is skipped when an
  object of the same size already exists. The video is then passed as
  `Part.from_uri(..., "video/mp4")`.
- **Failed questions.** At the end of `generate_until`, an ERROR lists every
  question with no answer (`_failed_docs`) and states that the score is not
  comparable to the paper.
- **Resume.** `continual_mode` and `response_persistent_folder`, already in the
  code, act as the resume mechanism. Empty answers are redone on re-run.

### `evaluate_vsi.sh`

New flags:
- `--thinking` / `--no-thinking` (default: no thinking);
- `--backend api|vertex` (default `api`);
- `--gcp_project`, `--gcp_location`, `--gcs_bucket` (each defaults to the
  matching env var: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`,
  `VSI_BENCH_GCS_BUCKET`);
- `--resume_dir PATH`, which maps to
  `continual_mode=true,response_persistent_folder=PATH`.

Other behavior:
- With `--backend vertex`, the script does not read `.google_api_key` and
  unsets `GOOGLE_API_KEY`, so calls cannot silently go to AI Studio.
- Values that contain `,` or `=` are rejected, because `model_args` uses them
  as separators.

### Other files

- `pyproject.toml`: the `[gemini]` extra now has `google-genai>=2.33.0` and
  `google-cloud-storage>=2.18`.
- `README.md`: documents the ReVSI protocol command, thinking, the Vertex
  setup, retries and resume.
- `.gitattributes`: new file.

### Earlier uncommitted work (before this session)

- The Gemini backend was migrated from `google-generativeai` to `google-genai`.
- The `vsibench_tiny` task was added (`vsibench_tiny.yaml`,
  `vsibench_tiny_ids.txt`, `process_docs_tiny`).
- `evaluate_vsi.sh` was added.
- `VSI_BENCH_DATA_DIR` support was added.

## 5. Tests done

All tests were run **offline**, with fake clients, in conda env `vsibench`.
The scripts are in the session scratchpad and are not in the repo.

| Area | Verified behavior |
|---|---|
| Default config | `thinking_budget=0`, 8192 tokens, temperature 0 |
| Thinking fallback | after a 400, the code switches to `minimal` |
| `thinking=True` | sends no `thinking_config` |
| Request timeout | a silent local server triggers the timeout, then the retry runs |
| Vertex client | created with `vertexai=True`, `location=global`, 300000 ms timeout |
| GCS upload | a transient failure is retried |
| GCS reuse | a video is reused within a run, and a new process does not re-upload it |
| Part URI | `gs://bkt/vsibench/scannet/<scene>.mp4`, `video/mp4`; the video comes before the text |
| AI Studio upload | a 503 on upload is retried and a 503 while polling is tolerated |
| Non-retryable upload error | a 400 is not retried and appears in the failed-questions summary |
| `evaluate_vsi.sh` dry run (fake `python`) | correct `model_args`; `GOOGLE_API_KEY` unset with Vertex; missing project gives a clear error |

**Live AI Studio test, `--limit 5`, gemini-3.8-flash (2026-11-17, 11:55–13:2x): FAILED, 0/5 answered.**
- Log: `logs/local-test/api-lim5/vsibench_tiny/run.log`.
- The 40 failed attempts were 14 × 503 UNAVAILABLE ("high demand"), 2 × 504 DEADLINE_EXCEEDED (one hung about 27 min) and 24 × 429 RESOURCE_EXHAUSTED.
- The 429 is the **free-tier quota of 20 generate requests per day per model**. The retries used up that quota, and the API asked to wait about 19 h 38 min.
- The reported score of 0.0 is an artifact of the empty answers and is not a result.
- Consequence: the AI Studio free tier cannot run even `--limit 5` reliably. Use Vertex (trial credits) or an AI Studio key with billing enabled.

**Not yet verified live:**
- whether `thinking_budget=0` is accepted by gemini-3.8-flash;
- the Vertex path end to end;
- availability of `gemini-3.8-flash` on the Vertex `global` location;
- the media-resolution token comparison.

## 6. Cost estimate (Gemini 3.8 Flash, full VSI-Bench tiny, default resolution, thinking off)

**Token counts.** Measured with `count_tokens` on real videos:

| Video | Tokens | Tokens per second |
|---|---|---|
| 28 s | 1935 | 68.5 |
| 141 s | 9619 | 68.1 |

The rate is about **68 tokens per second of video**. It is the same for
gemini-3.6-flash and gemini-3.8-flash.

**Dataset.** The tiny set has 392 questions over 185 unique videos. Without
context caching, every question re-sends its video, which gives 45,106 s of
video and about **3.1M input tokens**.

**Prices.** Per 1M tokens, the same for AI Studio and Vertex, for 3.6 and 3.8
Flash:

| Period | Input | Output (thinking included) |
|---|---|---|
| Until 2026-12-31 | $0.75 | $3.75 |
| From 2027-01-01 | $1.50 | $7.50 |

**Totals.**

| Period | API cost | Worst case with the `minimal` thinking fallback |
|---|---|---|
| Until 2026-12-31 | ≈ **$2.3** | ≈ $2.8 |
| From 2027-01-01 | ≈ **$4.7** | ≈ $5.6 |

Output without thinking is negligible.

**Other costs.**
- VM (e2-standard-2, about $0.07 per hour, about 3 h): about $0.2.
- Disk (about 30 GB): a few cents per day. The dataset is 17 GB with the zips
  and 11 GB extracted.
- GCS staging: cents.

**Caveats.**
- 503 responses are not billed. Client-side timeouts may still be billed if
  the server processed the request.
- Higher media resolutions cost more. This has not been measured yet.
- There is no hard spend cap. Suggest a billing budget alert, for example
  at $5.

## 7. Open issues and next steps

### 7.1 Open bug: the request timeout does not bound the wall-clock time of real calls

This shows up in the local test run with the AI Studio backend and
gemini-3.8-flash (§7.2):
- attempt 2 started at about 12:01 and only returned a server-side
  `504 DEADLINE_EXCEEDED` at 12:28;
- attempt 6 started at about 12:37 and was still hanging at 13:10.

The httpx timeout appears to be per read or connect phase, not a total
deadline. The offline test with a fully silent server did time out
correctly.

**Proposed fix:** enforce a total deadline per call. For example, run
`generate_content` and the uploads in a `concurrent.futures` worker and use
`future.result(timeout=…)`, treating an expired deadline as retryable.

### 7.2 Local `--limit 5` tests, required before the cloud

**AI Studio backend.** The test is running with PID 24140. It started at
11:55 and has **0/5 answers** so far.
- Upload works.
- Generation gets repeated 504/503 errors because gemini-3.8-flash is
  overloaded on AI Studio.
- Output: `logs/local-test/api-lim5/`. Responses:
  `logs/local-test/api-lim5-responses/`.

```bash
bash evaluate_vsi.sh --model gemini-3.8-flash --benchmark vsi-bench-tiny --backend api \
  --no-thinking --no-context-caching --launcher python --limit 5 \
  --resume_dir logs/local-test/api-lim5-responses --output_path logs/local-test/api-lim5
```

**Vertex backend locally.** This needs the user to do the following on the PC:
1. Install the Google Cloud CLI (`winget install Google.CloudSDK`); it is not
   installed yet.
2. Run `gcloud init`, `gcloud auth application-default login` and
   `gcloud auth application-default set-quota-project PROJECT`.
3. Enable the `aiplatform` and `storage` APIs and create a bucket. The Console
   in the browser works for this.
4. Provide the **project ID** and the **bucket name**, which are still
   unknown.
5. Install the extras: `pip install -e ".[gemini]"`. `google-cloud-storage` is
   not installed in the env yet.

Then run:

```bash
bash evaluate_vsi.sh --model gemini-3.8-flash --benchmark vsi-bench-tiny --backend vertex \
  --gcp_project PROJECT --gcs_bucket BUCKET --no-thinking --no-context-caching \
  --launcher python --limit 5 --output_path logs/local-test/vertex-lim5
```

Check the log for: no empty answers, no thinking warning, and no
failed-questions summary.

### 7.3 Media-resolution comparison

The user asked for a comparison of LOW, MEDIUM, HIGH and ULTRA_HIGH against
the default, for 3.8 Flash. It could not be measured because the model was
not answering.

The script `res.py` sends one `generate_content` call per mode on the 28 s
video and reads `prompt_token_count`. That costs less than $0.01. Re-run it
when the model responds. `count_tokens` on the Gemini API does not accept a
media-resolution config.

### 7.4 VM setup (after the local tests pass)

- **Code.** The repo `origin` is upstream `vision-x-nyu/thinking-in-space`,
  so the user cannot push there. Commit on a new branch and push to the user's
  fork, or copy the files. The user has not yet confirmed a commit.
- **Data.** `data/VSI-Bench` is a local symlink to a Windows folder. On the
  VM, run `hf auth login` (the task yaml uses `token: True`), download
  `nyu-visionx/VSI-Bench`, unzip the videos and set the data dir
  (`--data_dir`).
- **Environment.** Python 3.10, then `pip install -e ".[gemini]"`. The
  submodules (LLaVA-NeXT and others) are not needed for Gemini.
- **IAM and APIs.**
  - VM access scope: `cloud-platform`.
  - Service account roles: Vertex AI User, and Storage Object Admin on the
    bucket.
  - APIs enabled: `aiplatform` and `storage`.
- **Final run.** The same command as §7.2 for Vertex, without `--limit`,
  with `--resume_dir logs/revsi-protocol/gemini-3.8-flash-responses` and
  `--output_path logs/revsi-protocol/gemini-3.8-flash`. If the end-of-run
  summary lists failed questions, re-run the same command.

## 8. Billing and model-lifecycle facts (checked 2026-11-17)

- **Free-trial credits.** The trial was activated on **2026-11-06**.
  - The $300 credit **cannot** pay for Gemini API / AI Studio usage, per the
    official Google Cloud Free Program page.
  - It **can** pay for Vertex AI, because the trial started before the
    2026-11-16 cutoff, after which new trials are excluded.
  - The credit is valid for 90 days.
- **Gemini 3.6 Flash retirement.**
  - On Vertex AI it is retired on **2026-11-19**; the replacement is 3.8
    Flash.
  - On the Gemini API it is listed with **no shutdown date**.
  - Either way, use **gemini-3.8-flash**.
- **gemini-2.5-flash** returns 404 on the Gemini API ("no longer available to
  new users").
- **Capacity on 2026-11-17.** On AI Studio, gemini-3.6-flash returned 503
  ("high demand") and gemini-3.8-flash requests hung or returned 504.

## 9. Sources

- ReVSI paper: https://arxiv.org/pdf/2604.24300 (Appendix, Table 8)
- ReVSI project page: https://3dlg-hcvc.github.io/revsi/
- Gemini API pricing: https://aistudio.google.com/docs/pricing
- Vertex AI pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing
- Google Cloud Free Program: https://docs.cloud.google.com/free/docs/free-cloud-features
- Gemini 3.6 Flash on Vertex AI (retirement): https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-6-flash
- Gemini API deprecations: https://ai.google.dev/gemini-api/docs/deprecations
