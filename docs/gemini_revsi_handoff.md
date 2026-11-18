# Handoff: Gemini Flash evaluation on VSI-Bench-tiny (ReVSI protocol)

Status as of 2026-11-18, about 01:00 CET.

- Branch: `adapt-vsibench`, pushed to `origin`, which is the user's fork
  (`MicheleCazzola/thinking-in-space`).
- Committed and pushed:
  - `7b76cec`: Gemini backends, `vsibench_tiny` and `evaluate_vsi.sh`.
  - `c8fa4bb`: CPU support for LLaVA-OneVision, plus `patches/llava-next-cpu.patch`.
- **Not committed yet** (working tree only):
  - the total per-call deadline (§4.2);
  - `--gcs_prefix` and the no-overwrite GCS reuse (§4.3);
  - the thinking-mode switch and token logging (§4.4);
  - per-question metadata and `details` in the results (§4.5);
  - the README updates;
  - the new file `lmms_eval/api/response_metadata.py`.

  The user does the git operations. A commit is needed before the VM run,
  otherwise `git_hash` in `results.json` does not match the code that ran.

## 1. Goal

Run online inference with **Gemini 3.8 Flash** on **VSI-Bench tiny** using the
**same evaluation setup as the ReVSI paper** (arXiv 2604.24300). Thinking is
**requested off**; §3.1 explains why it cannot be fully disabled. The final run
happens on a **Google Cloud VM**, billed to the Google Cloud free-trial credits
through **Vertex AI**.

Standing rules from the user:
- Keep **both Gemini backends**: Google AI Studio (API key) and Vertex AI. Both
  must be selectable from the CLI (`--backend api|vertex`).
- **Always test locally with `--limit 5` before anything runs on the cloud**,
  for each backend that will be used, and again after code changes.
- Do not interrupt the user's running evaluations.
- Do not run git operations unless the user asks.
- Keep the official VSI-Bench metrics exactly as upstream (see §3.3), so that
  results stay comparable with the paper.

Google Cloud resources:
- Project: `project-136aa544-022b-4530-ac8`.
- Bucket: `spatial-vlm`.
- Dataset already in the bucket, extracted:
  `gs://spatial-vlm/data/VSI-Bench/{arkitscenes,scannet,scannetpp}/<scene>.mp4`.
- Videos are also expected on local disk: `data/VSI-Bench` on the PC, and the
  persistent disk on the VM.

## 2. ReVSI protocol and how this repo matches it

Source: the ReVSI appendix ("For proprietary models…") and Table 8.

| ReVSI protocol | This repo |
|---|---|
| lmms-eval framework, greedy decoding | `temperature=0, top_p=1, top_k=1` (`_build_generation_config`) |
| Official VSI-Bench **tiny** set for proprietary models | task `vsibench_tiny`, using the 400 official ids in `lmms_eval/tasks/vsibench/vsibench_tiny_ids.txt` |
| Gemini 3: whole video, Gemini's default FPS-based sampling (1 FPS), no fixed frame count | whole mp4 passed by GCS URI (Vertex) or Files API upload (AI Studio) |
| VSI-Bench proprietary prompt: `[VIDEO][PRE][QUESTION][POST]`, pre = "These are frames of a video." | `vsibench_doc_to_text` default pre-prompt plus the `gemini_api` post-prompts in the yaml |
| Metrics: MRA (numerical), accuracy (MCA), official aggregation | official values computed exactly as upstream; extras only under `details` (§4.5) |
| Media resolution: not specified, so the default is assumed | do **not** pass `--resolution_mode` |

Known deviations and open choices:
- **392 questions, not 400.** 8 of the official tiny ids were removed from the
  current HF dataset release. The paper does not say which dataset revision it
  used.
- **Numerical post-prompt wording.** The official code says "Do not response
  anything other than a single number!" (with the typo). The ReVSI Table 8
  prints "Do not respond with anything other than a single number!". The repo
  **keeps the official code string**. Changing it is a one-line yaml edit; the
  user has not decided.
- **Thinking.** The user chose to turn it off; the paper does not specify it.
  It cannot be fully disabled on this model (§3.1).
- `max_output_tokens=8192`, the same as the original lmms-eval Gemini code. The
  paper does not specify it.

## 3. Findings

### 3.1 Thinking cannot be fully disabled on gemini-3.8-flash (Vertex)

Measured in local `--limit 5` runs on 2026-11-18, on the same 5 questions:

| Request | Result |
|---|---|
| `thinking_budget=0` (`budget0`, the default) | accepted, 5/5 answers, but the model still thinks: mean 1209 thinking tokens per answer (range 287–2489) in one run, 937 in another; 4–5 of 5 answers used thinking |
| `thinking_level=minimal` (`GEMINI_THINKING_OFF_MODE=minimal`) | rejected on every request: `400 INVALID_ARGUMENT 'Thinking level is unsupported: THINKING_LEVEL_MINIMAL'`, 0/5 answers |

Decision: use `budget0`. Thinking tokens are billed as output (§6). This
should be reported as a deviation: thinking was requested off, but the model
still uses some.

### 3.2 Results are not fully deterministic

Even with greedy decoding, repeated runs on the same questions differ:
- question 182 (object_size_estimation) was answered 212 in two runs and 205
  in a third, so its MRA went from 0.6 to 0.5;
- question 37 used 354, 994 and 840 thinking tokens in three runs.

The likely cause is the residual thinking. A single run on the tiny set
therefore carries some noise. Repeating the run 2–3 times (about $4 each)
would give a spread; the user has not decided.

### 3.3 MRA floating-point edge case (kept on purpose)

The official MRA compares the relative error with `1 - np.linspace(.5, .95, 10)`.
Floating-point rounding makes some thresholds slightly smaller than their
decimal value: `1 - 0.8` gives `0.19999999999999996`. An answer whose relative
error is exactly 20% or 45% therefore scores 0.1 less than with exact
thresholds. For example, ground truth 5 and answer 4 or 6 scores 0.6 instead of
0.7.

Among integer pairs with ground truth 1–20 and answer 0–40, 10 pairs are
affected: 5→4/6, 10→8/12, 15→12/18, 20→16/24 (20%) and 20→11/29 (45%). The user
decided **not** to change it, so the scores stay comparable with the paper. An
independent recomputation from the JSON files matches `results.json` once it
uses the same float thresholds.

### 3.4 Earlier root causes (all fixed)

- **Empty answers.** Old code had `max_output_tokens=32`. Thinking tokens
  count against that cap, so answers came back empty. Now the cap is 8192.
- **`timeout` unused.** Fixed. The httpx timeout only bounds inactivity, so a
  total deadline per call was added (§4.2).
- **Silent upload failures.** Fixed: uploads are retried and reported.
- **CRLF line endings in `evaluate_vsi.sh`.** Fixed, and `.gitattributes`
  forces LF for `*.sh`.

## 4. Code changes

### 4.1 Already committed (`7b76cec`)

- **`google-genai` SDK; args** `thinking`, `max_output_tokens`, `timeout`
  (env `GEMINI_REQUEST_TIMEOUT`, default 300), `backend` (`api` | `vertex`),
  `gcp_project`, `gcp_location` (default `global`), `gcs_bucket` and
  `gcs_prefix`.
- **Retries.** Exponential backoff: 15 s doubling, capped at 300 s, 8 attempts
  (`GEMINI_MAX_ATTEMPTS`, `GEMINI_RETRY_BASE_SECONDS`,
  `GEMINI_RETRY_MAX_SECONDS`). They apply to 429/5xx, timeouts and dropped
  connections, for both generation and uploads.
- **Failure reporting.** Questions with no answer are listed at the end of the
  run.
- **Resume.** `--resume_dir` maps to `continual_mode`; a re-run redoes only the
  missing answers.
- **`evaluate_vsi.sh`.** Flags `--thinking`/`--no-thinking`, `--backend`,
  `--gcp_*`, `--gcs_bucket`, `--resume_dir`. With Vertex, `GOOGLE_API_KEY` is
  unset.

### 4.2 Total per-call deadline (uncommitted)

The httpx timeout resets on every byte received. A real call on AI Studio once
hung for about 27 minutes before the server returned 504.

- Every API call runs in a daemon thread with a total deadline. This covers
  generation, AI Studio upload, file polling, GCS staging and cache creation.
- The deadline is `call_deadline`, or env `GEMINI_CALL_DEADLINE`. The default
  equals `timeout` (300 s); `0` disables it.
- When the deadline expires, the call is abandoned and a retryable
  `_CallTimeoutError` is raised.
- The end of the run logs how many calls were abandoned. An abandoned request
  may still be billed.

### 4.3 GCS reuse, `--gcs_prefix` (uncommitted)

- **`evaluate_vsi.sh`** has a new flag, `--gcs_prefix` (env
  `VSI_BENCH_GCS_PREFIX`, default `vsibench`). For this project use
  `--gcs_prefix data/VSI-Bench`.
- **`_stage_in_gcs`** reuses any existing object and **never overwrites it**.
  - If the size differs from the local file, it logs a warning and uses the
    bucket copy.
  - A missing object is uploaded with `if_generation_match=0`, so an upload
    can never overwrite an object.
- The local video must still exist, because the task checks the path.

### 4.4 Thinking mode and token logging (uncommitted)

- **`GEMINI_THINKING_OFF_MODE`** is `budget0` (default) or `minimal`. Any other
  value is rejected.
- **One log line per question:**
  `Usage doc N: prompt=… output=… thoughts=… (thinking mode '…')`.
- **End-of-run total:** prompt, output and thinking tokens, the mean thinking
  tokens per answer, and how many answers used thinking.

### 4.5 Per-question metadata and `details` in the results (uncommitted)

- **`lmms_eval/api/response_metadata.py` (new).** A module-level registry. The
  model records metadata per question, keyed by the dataset `id`, and the
  task's `process_results` reads it. The registry is needed because lmms-eval
  passes only the generated text from the model to the task.
- **`gemini_api.py`** records, for each question:
  - `status`: `ok`, `api_error` or `upload_error`;
  - `attempts` and `last_error`;
  - `prompt_tokens`, `output_tokens`, `thoughts_tokens`;
  - `finish_reason` and `thinking_mode`.

  With `--resume_dir`, this is also saved to `<model>_metadata.json`. Answers
  taken from the resume cache keep their metadata and get
  `from_resume_cache: true`.
- **`utils.py`, per sample:** two new fields, `parsed_prediction` (the value
  compared with the ground truth; `null` = unparsable number) and
  `response_meta`.
- **`utils.py`, in `results.json`:** the official keys and the `tabulated_*`
  row are unchanged. A new key `details` holds:
  - `n_samples` and `n_per_question_type`;
  - `object_rel_direction_subtypes_accuracy` (easy, medium, hard);
  - `mca_mean_accuracy` and `na_mean_MRA`;
  - `per_dataset` (arkitscenes, scannet, scannetpp);
  - `n_unparsable_numerical_answers`;
  - `api`: status counts, no-answer count, resume count, thinking modes and
    token totals.

  `details` is computed in a `try` block, so the official results are never
  lost because of it.

## 5. Tests done

### 5.1 Offline

All tests ran with fake clients in conda env `vsibench`. The scripts are in
the session scratchpad, not in the repo.

| Suite | Result | Covers |
|---|---|---|
| deadline | 18/18 | defaults/env/disable; hang abandoned then retry OK; all attempts hang → failed question in bounded time; 400 not retried, 503 retried, thinking fallback; upload, polling and GCS hangs recovered; **real SDK against a trickling local server: bug reproduced (alive after 6 s with a 2 s httpx timeout), fix cuts it at 3.0 s** |
| GCS staging | 5/5 | existing object reused with no upload and correct URI; different size not overwritten; missing object uploaded once with `if_generation_match=0`; default prefix |
| thinking/usage | 7/7 | budget0 and minimal configs; invalid value rejected; per-question lines; totals; `thinking=True` sends no config |
| results/details | 21/21 | **old vs new aggregation on all 392 tiny questions (synthetic answers, all 8 question types): official keys, values and tabulated row identical**; counts (392; rel_direction 10/27/13; datasets 140/150/102); subtype mean equals the official value; per-dataset, unparsable and API diagnostics; JSON-serialisable; subset of 5; no metadata → no `api` block; metadata through `generate_until` incl. retry, API error and resume |
| `evaluate_vsi.sh` dry run (fake `python`) | OK | `gcs_prefix=data/VSI-Bench` reaches `model_args` |

### 5.2 Live

| Run (2026-11-17/18) | Result |
|---|---|
| AI Studio, `--limit 5` | **failed**, 0/5: 503/504 overload, then the free-tier quota of 20 requests/day was exhausted (429). Do not use the AI Studio free tier for this. |
| Vertex, `--limit 5`, `budget0` (`logs/local-test/vertex-lim5`, run `1117_1818`) | 5/5, 51.25; video read from the bucket, nothing uploaded; ~26 s per question |
| Vertex, `budget0` vs `minimal` (`logs/local-test/think-budget0`, `think-minimal`) | see §3.1 |
| Vertex, `--limit 5` with all the new code (`logs/local-test/vertex-lim5`, run `1117_1848`) | 5/5, 46.25 (see §3.2); one `Server disconnected` error was retried and recovered (`attempts: 2`); `parsed_prediction`, `response_meta` and `details` present and correct |

The `run.log` of a run is overwritten when another run uses the same
`--output_path`. Use a new output path for the final run.

Still not verified live: the multiple-choice path (`accuracy`, rel_direction
subtypes, `mca_mean_accuracy`), because the first 5 tiny questions are all
numerical. It is covered offline (§5.1).

## 6. Cost estimate (Gemini 3.8 Flash on Vertex, full tiny set, default resolution)

- **Input.** The tiny set has 392 questions over 185 unique videos. Video costs
  about 68 tokens per second. Without context caching every question re-sends
  its video: about **3.1M input tokens**. The live runs match this rate
  (47,098 prompt tokens for 5 questions).
- **Output.** About 1,000 thinking tokens per answer (§3.1) gives about
  **0.4M output tokens**. The answers themselves are negligible.

| Period | Input $/1M | Output $/1M | Estimated total |
|---|---|---|---|
| Until 2026-12-31 | $0.75 | $3.75 | ≈ **$3.9** (2.3 input + 1.6 thinking) |
| From 2027-01-01 | $1.50 | $7.50 | ≈ **$7.8** |

Other costs:
- VM: e2-standard-2, about $0.07/h, and the run takes about 2.8 h
  (≈26 s/question).
- Persistent disk: a few cents per day, also while the VM is stopped.
- Bucket storage: about 11 GB, roughly $0.25/month.

There is no hard spend cap; set a billing budget alert, for example at $5.
Costs show up in Billing → Credits / Reports with a delay of a few hours.

## 7. Next steps

1. **Commit and push** the uncommitted changes on `adapt-vsibench`, including
   the new `lmms_eval/api/response_metadata.py`. The user does this.
2. **VM setup.**
   - Create the VM with access scope `cloud-platform`. The service account
     needs Vertex AI User and Storage Object Viewer on `spatial-vlm`; Viewer
     is enough because nothing is uploaded.
   - Install Python 3.10 and `pip install -e ".[gemini]"` from a clone of
     `adapt-vsibench`. The submodules are not needed.
   - Put the videos on the persistent disk and pass them with `--data_dir`.
     The HF annotations need `hf auth login`, because the task yaml uses
     `token: True`.
3. **Final run** in `tmux`:

   ```bash
   bash evaluate_vsi.sh --model gemini-3.8-flash --benchmark vsi-bench-tiny --backend vertex \
     --gcp_project project-136aa544-022b-4530-ac8 --gcs_bucket spatial-vlm --gcs_prefix data/VSI-Bench \
     --data_dir PATH_TO_VIDEOS --no-thinking --no-context-caching --launcher python \
     --resume_dir logs/revsi-protocol/gemini-3.8-flash-responses \
     --output_path logs/revsi-protocol/gemini-3.8-flash
   ```

   If the end-of-run summary or `details.api.n_no_answer_api_error` reports
   questions with no answer, re-run the same command; only those questions
   are redone. Then copy `logs/` back and stop or delete the VM.
4. **Open choices for the user:** the numerical post-prompt wording (§2) and
   repeated runs for variance (§3.2).
5. **Media-resolution comparison** (requested earlier, not done): one
   `generate_content` call per mode on the 28 s video, reading
   `prompt_token_count`. It costs less than $0.01.

## 8. Billing and model-lifecycle facts (checked 2026-11-17)

- **Free-trial credits.** The trial was activated on **2026-11-06**.
  - The $300 credit **cannot** pay for Gemini API / AI Studio usage, per the
    official Google Cloud Free Program page.
  - It **can** pay for Vertex AI, because the trial started before the
    2026-11-16 cutoff.
  - The credit is valid for 90 days.
  - The project is linked to billing account `01F348-BDEA2B-6255D3` (the only
    one), and billing is enabled.
- **Gemini 3.6 Flash** is retired on Vertex AI on **2026-11-19**; use
  **gemini-3.8-flash**, which is available on the Vertex `global` location
  (verified live).
- **gemini-2.5-flash** returns 404 on the Gemini API ("no longer available to
  new users").

## 9. Sources

- ReVSI paper: https://arxiv.org/pdf/2604.24300 (Appendix, Table 8)
- ReVSI project page: https://3dlg-hcvc.github.io/revsi/
- Gemini API pricing: https://aistudio.google.com/docs/pricing
- Vertex AI pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing
- Google Cloud Free Program: https://docs.cloud.google.com/free/docs/free-cloud-features
- Gemini 3.6 Flash on Vertex AI (retirement): https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-6-flash
- Gemini API deprecations: https://ai.google.dev/gemini-api/docs/deprecations
