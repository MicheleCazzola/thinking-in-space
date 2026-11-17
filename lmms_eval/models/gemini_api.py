import io
import json
import mimetypes
import os
import re
import time
from typing import List, Tuple

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

import hashlib

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
configured_ca_file = os.getenv("SSL_CERT_FILE")
if configured_ca_file and not os.path.isfile(configured_ca_file):
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        os.environ.pop("SSL_CERT_FILE", None)

try:
    from google import genai
    from google.genai import types

except Exception as e:
    eval_logger.error(f"Error importing google-genai: {str(e)}")
    genai = None
    types = None

class _TransientUploadError(RuntimeError):
    """Upload-side condition worth retrying (e.g. file stuck in PROCESSING)."""


@register_model("gemini_api")
class GeminiAPI(lmms):
    @staticmethod
    def _sanitize_error(error):
        message = str(error)
        if GOOGLE_API_KEY:
            message = message.replace(GOOGLE_API_KEY, "<redacted-api-key>")
        return re.sub(r"([?&]key=)[^&\s\"]+", r"\1<redacted-api-key>", message)

    @staticmethod
    def _is_retryable_error(error):
        if re.search(r"\b(?:429|500|502|503|504)\b", str(error)):
            return True
        # Client-side timeouts / dropped connections (httpx raises these without an HTTP code).
        name = type(error).__name__.lower()
        return "timeout" in name or "connect" in name or "remoteprotocol" in name

    def __init__(
        self,
        model_version: str = "gemini-1.5-pro",
        modality: str = "image",
        context_cache: bool = False,
        resolution_mode: str = None,
        thinking: bool = False,
        max_output_tokens: int = 8192,
        timeout: int = int(os.getenv("GEMINI_REQUEST_TIMEOUT", "300")),  # seconds per HTTP request
        backend: str = os.getenv("GEMINI_BACKEND", "api"),  # "api" (Gemini API key) or "vertex" (Vertex AI / ADC)
        gcp_project: str = None,
        gcp_location: str = None,
        gcs_bucket: str = None,
        gcs_prefix: str = "vsibench",
        continual_mode: bool = False,
        response_persistent_folder: str = None,  # We will cache the Gemini API response in this path and use it for future requests
        **kwargs,
    ) -> None:
        super().__init__()
        if genai is None or types is None:
            raise RuntimeError("The google-genai package is required for Gemini evaluation.")
        self.backend = str(backend).lower()
        # Without an explicit timeout a stalled request blocks the whole run forever.
        http_options = types.HttpOptions(timeout=int(timeout) * 1000)  # milliseconds
        if self.backend == "api":
            if not GOOGLE_API_KEY:
                raise RuntimeError("GOOGLE_API_KEY is required for the Gemini API backend.")
            self.client = genai.Client(api_key=GOOGLE_API_KEY, http_options=http_options)
        elif self.backend == "vertex":
            # Vertex AI authenticates with Application Default Credentials (on a GCE VM:
            # the VM service account). Vertex has no Files API, so videos are staged in GCS.
            self.gcp_project = gcp_project or os.getenv("GOOGLE_CLOUD_PROJECT")
            self.gcp_location = gcp_location or os.getenv("GOOGLE_CLOUD_LOCATION") or "global"
            self.gcs_bucket_name = gcs_bucket or os.getenv("VSI_BENCH_GCS_BUCKET")
            self.gcs_prefix = str(gcs_prefix).strip("/")
            if not self.gcp_project:
                raise RuntimeError("Vertex backend requires gcp_project or GOOGLE_CLOUD_PROJECT.")
            if not self.gcs_bucket_name:
                raise RuntimeError("Vertex backend requires gcs_bucket or VSI_BENCH_GCS_BUCKET.")
            try:
                from google.cloud import storage
            except ImportError as exc:
                raise RuntimeError(
                    "The Vertex backend needs google-cloud-storage: pip install -e '.[gemini]'"
                ) from exc
            self.client = genai.Client(
                vertexai=True,
                project=self.gcp_project,
                location=self.gcp_location,
                http_options=http_options,
            )
            self.gcs_bucket = storage.Client(project=self.gcp_project).bucket(self.gcs_bucket_name)
            eval_logger.info(
                f"Gemini backend: Vertex AI (project={self.gcp_project}, location={self.gcp_location}, "
                f"videos staged in gs://{self.gcs_bucket_name}/{self.gcs_prefix}/)"
            )
        else:
            raise ValueError(f"Unknown Gemini backend: {backend} (expected 'api' or 'vertex')")
        self.model_version = model_version
        self.timeout = timeout
        self.context_cache_enabled = context_cache
        self.thinking_enabled = bool(thinking)
        self.max_output_tokens = int(max_output_tokens)
        # When thinking is disabled we first request thinking_budget=0 (the hard
        # "off" switch). Models that refuse a zero budget fall back to the lowest
        # level they accept ("minimal"); see _thinking_fallback.
        self.thinking_mode = None if self.thinking_enabled else "budget0"
        self._thinking_warning_emitted = False
        if resolution_mode is not None:
            resolution_mode = resolution_mode.lower().replace("-", "_")
            valid_resolution_modes = {"low", "medium", "high", "ultra_high"}
            if resolution_mode not in valid_resolution_modes:
                raise ValueError(
                    f"Unsupported Gemini resolution mode: {resolution_mode}. "
                    f"Expected one of: {', '.join(sorted(valid_resolution_modes))}"
                )
            resolution_enum_name = f"MEDIA_RESOLUTION_{resolution_mode.upper()}"
            try:
                self.media_resolution = getattr(types.MediaResolution, resolution_enum_name)
            except AttributeError as exc:
                raise RuntimeError(
                    "The installed google-genai package does not support media resolution."
                ) from exc
        else:
            self.media_resolution = None
        self.file_processing_timeout = int(os.getenv("GEMINI_FILE_PROCESSING_TIMEOUT", "120"))
        self.file_processing_poll_interval = float(os.getenv("GEMINI_FILE_PROCESSING_POLL_INTERVAL", "2"))
        self.max_attempts = max(1, int(os.getenv("GEMINI_MAX_ATTEMPTS", "8")))
        self.retry_base_seconds = max(
            0.0, float(os.getenv("GEMINI_RETRY_BASE_SECONDS", "15"))
        )
        self.retry_max_seconds = max(0.0, float(os.getenv("GEMINI_RETRY_MAX_SECONDS", "300")))
        self._failed_docs = []
        self.generation_config = self._build_generation_config()
        self._uploaded_files = {}
        self._cached_contents = {}
        self._context_cache_warning_emitted = False
        self.continual_mode = continual_mode
        if self.continual_mode and response_persistent_folder is None:
            raise ValueError("Continual mode requires a persistent path for the response. We will cache the Gemini API response in this path and use it for future requests. Please provide a valid path.")
        self.response_persistent_folder = response_persistent_folder
        if self.continual_mode:
            if not os.path.exists(self.response_persistent_folder):
                os.makedirs(self.response_persistent_folder)
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")
            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, "Continual mode is not supported with distributed inference."
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

        self.modality = modality

    def _build_generation_config(self, cached_content=None):
        # Greedy decoding as in the original VSI-Bench / lmms-eval Gemini setup
        # (also the protocol followed by ReVSI).
        config = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.thinking_mode == "budget0":
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        elif self.thinking_mode == "minimal":
            config["thinking_config"] = types.ThinkingConfig(thinking_level="minimal")
        if self.media_resolution is not None:
            config["media_resolution"] = self.media_resolution
        if cached_content is not None:
            config["cached_content"] = cached_content
        return types.GenerateContentConfig(**config)
        
    def _thinking_fallback(self, error):
        """Switch from thinking_budget=0 to thinking_level=minimal if the model rejects a zero budget."""
        if self.thinking_mode != "budget0":
            return False
        message = str(error).lower()
        if "400" not in message or "think" not in message:
            return False
        eval_logger.warning(
            f"{self.model_version} rejected thinking_budget=0 ({self._sanitize_error(error)}). "
            "Falling back to thinking_level='minimal', the lowest thinking level the model accepts."
        )
        self.thinking_mode = "minimal"
        self.generation_config = self._build_generation_config()
        return True

    def _check_response(self, response, doc_id):
        usage = getattr(response, "usage_metadata", None)
        thoughts = getattr(usage, "thoughts_token_count", None) or 0
        if thoughts and not self.thinking_enabled and not self._thinking_warning_emitted:
            eval_logger.warning(
                f"Thinking is disabled but {self.model_version} used {thoughts} thinking tokens "
                f"(doc {doc_id}, thinking mode '{self.thinking_mode}'). This model may not allow thinking to be fully disabled."
            )
            self._thinking_warning_emitted = True
        if not response.text:
            candidates = getattr(response, "candidates", None) or []
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            eval_logger.warning(
                f"Empty Gemini response for doc {doc_id}: finish_reason={finish_reason}, "
                f"thoughts_tokens={thoughts}, prompt_feedback={getattr(response, 'prompt_feedback', None)}"
            )

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_image_size(self, image):
        # Create a BytesIO object to store the image bytes
        img_byte_array = io.BytesIO()

        # Save the image to the BytesIO object
        image.save(img_byte_array, format="PNG")

        # Get the size of the BytesIO object
        img_size = img_byte_array.tell()

        return img_size

    def _retry_delay(self, failed_attempts):
        """Exponential backoff: base, 2*base, 4*base, ... capped at retry_max_seconds."""
        return min(self.retry_max_seconds, self.retry_base_seconds * (2 ** (failed_attempts - 1)))

    def _call_with_retries(self, fn, description):
        """Run fn(), retrying transient errors (429/5xx, timeouts, dropped connections)."""
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except Exception as exc:
                message = self._sanitize_error(exc)
                retryable = self._is_retryable_error(exc) or isinstance(exc, _TransientUploadError)
                if not retryable or attempt == self.max_attempts:
                    eval_logger.error(
                        f"{description} failed after {attempt} attempt(s) "
                        f"({'non-retryable' if not retryable else 'retries exhausted'}): {message}"
                    )
                    raise
                delay = self._retry_delay(attempt)
                eval_logger.info(
                    f"{description}: attempt {attempt}/{self.max_attempts} failed ({message}); "
                    f"retrying in {delay:.1f} seconds."
                )
                time.sleep(delay)

    def _upload_media(self, path):
        if path in self._uploaded_files:
            return self._uploaded_files[path]
        if self.backend == "vertex":
            media = self._call_with_retries(lambda: self._stage_in_gcs(path), f"GCS upload of {path}")
        else:
            media = self._call_with_retries(
                lambda: self.wait_for_file_active(self.client.files.upload(file=path)),
                f"Gemini upload of {path}",
            )
        self._uploaded_files[path] = media
        return media

    def _stage_in_gcs(self, path):
        """Upload a local file to GCS once (skipped if an object with the same size exists) and
        return a Part referencing it. Gemini reads the whole video, as with the Files API."""
        parts = os.path.normpath(path).split(os.sep)
        blob_name = "/".join([self.gcs_prefix] + parts[-2:]) if self.gcs_prefix else "/".join(parts[-2:])
        blob = self.gcs_bucket.blob(blob_name)
        local_size = os.path.getsize(path)
        if blob.exists():
            blob.reload()
        if not blob.size or int(blob.size) != local_size:
            blob.upload_from_filename(path, timeout=self.timeout)
        mime_type = mimetypes.guess_type(path)[0] or "video/mp4"
        return types.Part.from_uri(file_uri=f"gs://{self.gcs_bucket_name}/{blob_name}", mime_type=mime_type)

    def encode_video(self, video_path):
        return self._upload_media(video_path)

    def encode_image(self, image_path):
        return self._upload_media(image_path)

    def wait_for_file_active(self, uploaded_file):
        file_name = getattr(uploaded_file, "name", None)
        if not file_name:
            raise RuntimeError("Gemini upload did not return a file name.")

        deadline = time.monotonic() + self.file_processing_timeout
        while True:
            try:
                current_file = self.client.files.get(name=file_name)
            except Exception as exc:
                # A transient error while polling must not discard a successful upload.
                if not self._is_retryable_error(exc) or time.monotonic() >= deadline:
                    raise
                eval_logger.info(f"Polling {file_name} failed ({self._sanitize_error(exc)}); polling again.")
                time.sleep(self.file_processing_poll_interval)
                continue
            state = getattr(current_file, "state", None)
            state_name = getattr(state, "name", str(state)).upper()
            if state_name == "ACTIVE":
                return current_file
            if state_name in {"FAILED", "ERROR"}:
                error = getattr(current_file, "error", None)
                raise RuntimeError(
                    f"Gemini file processing failed for {file_name}: "
                    f"{error or state_name}"
                )
            if time.monotonic() >= deadline:
                # Processing stuck: treated as transient, the upload is retried.
                raise _TransientUploadError(
                    f"Timed out waiting for Gemini file {file_name} to become ACTIVE "
                    f"(last state: {state_name})"
                )
            time.sleep(self.file_processing_poll_interval)

    def get_cached_content(self, video_path, uploaded_video):
        if video_path in self._cached_contents:
            return self._cached_contents[video_path]
        try:
            cached_content = self.client.caches.create(
                model=self.model_version,
                config=types.CreateCachedContentConfig(
                    display_name=f"vsibench-{hashlib.sha256(video_path.encode()).hexdigest()[:16]}",
                    contents=[uploaded_video],
                    ttl="3600s",
                ),
            )
            self._cached_contents[video_path] = cached_content.name
            eval_logger.info(f"Created Gemini context cache for {video_path}")
            return cached_content.name
        except Exception as exc:
            if not self._context_cache_warning_emitted:
                eval_logger.warning(
                    "Gemini context caching is unavailable for this SDK/model; "
                    f"continuing with upload caching only: {self._sanitize_error(exc)}"
                )
                self._context_cache_warning_emitted = True
            return None

    def convert_video(self, images):
        for idx, img in enumerate(images):
            if self.modality == "video" and isinstance(img, str):
                try:
                    images[idx] = self.encode_video(img)
                except Exception as e:
                    raise RuntimeError(f"Error converting video: {self._sanitize_error(e)}") from e
            elif self.modality == 'image' and isinstance(img, str):
                try:
                    images[idx] = self.encode_image(img)
                except Exception as e:
                    raise RuntimeError(f"Error converting image: {self._sanitize_error(e)}") from e
        return images

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            visual_paths = doc_to_visual(self.task_dict[task][split][doc_id])
            visuals = [visual_paths]
            request_contents = [contexts]
            cached_content_name = None
            if visuals != [None]:
                visuals = self.flatten(visuals)
                try:
                    visuals = self.convert_video(visuals)
                except RuntimeError as exc:
                    eval_logger.error(f"Skipping request for {doc_id}: {exc}")
                    self._failed_docs.append((doc_id, "video upload"))
                    res.append("")
                    pbar.update(1)
                    continue

            if visuals != [None]: # always follow gemini's suggestion that take video first
                if self.context_cache_enabled and self.modality == "video":
                    cached_content_name = self.get_cached_content(visual_paths[0], visuals[0])
                if cached_content_name is None:
                    request_contents.insert(0, visuals[0])

            content = ""
            succeeded = False
            attempt = 0
            while attempt < self.max_attempts:
                request_config = (
                    self._build_generation_config(cached_content_name)
                    if cached_content_name is not None
                    else self.generation_config
                )
                try:
                    response = self.client.models.generate_content(
                        model=self.model_version,
                        contents=request_contents,
                        config=request_config,
                    )
                    self._check_response(response, doc_id)
                    content = response.text or ""
                    succeeded = True
                    break
                except Exception as e:
                    error_message = self._sanitize_error(e)
                    eval_logger.info(
                        f"Attempt {attempt + 1}/{self.max_attempts} failed with error: "
                        f"{error_message}"
                    )
                    if self._thinking_fallback(e):
                        continue  # retry immediately with the fallback; does not consume an attempt
                    attempt += 1
                    if isinstance(e, ValueError):
                        content = ""
                        break
                    if not self._is_retryable_error(e):
                        eval_logger.error(
                            f"Non-retryable Gemini error: {error_message}"
                        )
                        break
                    if attempt < self.max_attempts:
                        delay = self._retry_delay(attempt)
                        eval_logger.info(
                            f"Retrying Gemini request in {delay:.1f} seconds."
                        )
                        time.sleep(delay)
                    else:
                        eval_logger.error(
                            f"All {self.max_attempts} attempts failed. "
                            f"Last error message: {error_message}"
                        )
            if not succeeded:
                self._failed_docs.append((doc_id, "generation"))
            res.append(content)
            pbar.update(1)

            if self.continual_mode is True:  # Cache the response
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = content
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        if self._failed_docs:
            eval_logger.error(
                f"{len(self._failed_docs)}/{len(requests)} requests got NO model answer because of API errors "
                f"and are scored as wrong: {self._failed_docs}. The score is NOT comparable to the paper; "
                "re-run (with continual_mode=True to only redo the failed ones)."
            )
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini API not support"
