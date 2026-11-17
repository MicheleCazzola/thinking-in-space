#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash evaluate_vsi.sh [options]

Options:
  --benchmark NAME       vsi-bench, vsi-bench-tiny, or vsi-bench-debiased (default: vsi-bench)
  --model NAME           One model name (default: llava_one_vision_qwen2_0p5b_ov)
  --num_frames N         Number of video frames for local video models (default: 2)
  --num_processes N      Accelerate process count (default: 1)
  --launcher NAME        python or accelerate (default: python)
  --limit N              Limit the number of questions per task
  --resolution_mode NAME Gemini media resolution: low, medium, high, ultra_high
  --output_path PATH     Output directory (default: logs/YYYYMMDD)
  --data_dir PATH        Local VSI-Bench video/annotation directory (default: data/VSI-Bench)
  --context-caching      Enable Gemini context caching (default: disabled)
  --no-context-caching   Disable Gemini context caching
  --thinking             Enable Gemini thinking (default: disabled)
  --no-thinking          Disable Gemini thinking (thinking_budget=0)
  --backend NAME         Gemini backend: api (key in .google_api_key) or vertex (default: api)
  --gcp_project ID       Vertex: Google Cloud project (default: $GOOGLE_CLOUD_PROJECT)
  --gcp_location NAME    Vertex: location (default: $GOOGLE_CLOUD_LOCATION or global)
  --gcs_bucket NAME      Vertex: GCS bucket used to stage the videos (default: $VSI_BENCH_GCS_BUCKET)
  --resume_dir PATH      Gemini: persist answers in PATH; re-running only redoes failed questions
  --help                 Show this help

Local model:
  llava_one_vision_qwen2_0p5b_ov

Gemini models:
  gemini_1p5_flash
  gemini_1p5_pro_002
  gemini_2p0_flash_exp
  Any Gemini API model ID from any generation, including 3 and 4:
  gemini-2.5-flash
  gemini-2.5-pro
  gemini-3-flash-preview
  gemini-3.1-pro-preview
  gemini-3.5-flash-lite
  gemini-3.6-flash
  gemini-3.8-flash
  gemini-4-pro
  gemini-2.0-flash
  models/gemini-2.5-flash
EOF
}

benchmark="vsi-bench"
model_names="llava_one_vision_qwen2_0p5b_ov"
num_frames=2
num_processes=1
launcher="python"
output_path="logs/$(TZ="America/New_York" date "+%Y%m%d")"
data_dir="data/VSI-Bench"
google_api_key_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.google_api_key"
limit=""
resolution_mode=""
context_cache="false"
thinking="false"
backend="api"
gcp_project="${GOOGLE_CLOUD_PROJECT:-}"
gcp_location="${GOOGLE_CLOUD_LOCATION:-global}"
gcs_bucket="${VSI_BENCH_GCS_BUCKET:-}"
resume_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
    --benchmark)
        [[ $# -ge 2 ]] || { echo "Missing value for --benchmark" >&2; exit 2; }
        benchmark="$2"
        shift 2
        ;;
    --model)
        [[ $# -ge 2 ]] || { echo "Missing value for --model" >&2; exit 2; }
        model_names="$2"
        shift 2
        ;;
    --num_frames)
        [[ $# -ge 2 ]] || { echo "Missing value for --num_frames" >&2; exit 2; }
        num_frames="$2"
        shift 2
        ;;
    --num_processes)
        [[ $# -ge 2 ]] || { echo "Missing value for --num_processes" >&2; exit 2; }
        num_processes="$2"
        shift 2
        ;;
    --launcher)
        [[ $# -ge 2 ]] || { echo "Missing value for --launcher" >&2; exit 2; }
        launcher="$2"
        shift 2
        ;;
    --limit)
        [[ $# -ge 2 ]] || { echo "Missing value for --limit" >&2; exit 2; }
        limit="$2"
        shift 2
        ;;
    --resolution_mode)
        [[ $# -ge 2 ]] || { echo "Missing value for --resolution_mode" >&2; exit 2; }
        resolution_mode="${2//-/_}"
        shift 2
        ;;
    --output_path)
        [[ $# -ge 2 ]] || { echo "Missing value for --output_path" >&2; exit 2; }
        output_path="$2"
        shift 2
        ;;
    --data_dir)
        [[ $# -ge 2 ]] || { echo "Missing value for --data_dir" >&2; exit 2; }
        data_dir="$2"
        shift 2
        ;;
    --context-caching)
        context_cache="true"
        shift
        ;;
    --no-context-caching)
        context_cache="false"
        shift
        ;;
    --thinking)
        thinking="true"
        shift
        ;;
    --no-thinking)
        thinking="false"
        shift
        ;;
    --backend)
        [[ $# -ge 2 ]] || { echo "Missing value for --backend" >&2; exit 2; }
        backend="$2"
        shift 2
        ;;
    --gcp_project)
        [[ $# -ge 2 ]] || { echo "Missing value for --gcp_project" >&2; exit 2; }
        gcp_project="$2"
        shift 2
        ;;
    --gcp_location)
        [[ $# -ge 2 ]] || { echo "Missing value for --gcp_location" >&2; exit 2; }
        gcp_location="$2"
        shift 2
        ;;
    --gcs_bucket)
        [[ $# -ge 2 ]] || { echo "Missing value for --gcs_bucket" >&2; exit 2; }
        gcs_bucket="${2#gs://}"
        gcs_bucket="${gcs_bucket%/}"
        shift 2
        ;;
    --resume_dir)
        [[ $# -ge 2 ]] || { echo "Missing value for --resume_dir" >&2; exit 2; }
        resume_dir="$2"
        shift 2
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

case "$benchmark" in
    vsi-bench)
        tasks=("vsibench")
        ;;
    vsi-bench-tiny)
        tasks=("vsibench_tiny")
        ;;
    vsi-bench-debiased)
        tasks=("vsibench_debiased")
        ;;
    *)
        echo "Invalid --benchmark: $benchmark (expected vsi-bench, vsi-bench-tiny, or vsi-bench-debiased)" >&2
        exit 2
        ;;
esac

case "$launcher" in
    python|accelerate)
        ;;
    *)
        echo "Invalid --launcher: $launcher (expected python or accelerate)" >&2
        exit 2
        ;;
esac

case "$resolution_mode" in
    ""|low|medium|high|ultra_high)
        ;;
    *)
        echo "Invalid --resolution_mode: $resolution_mode (expected low, medium, high, or ultra_high)" >&2
        exit 2
        ;;
esac

export LMMS_EVAL_LAUNCHER="$launcher"
export VSI_BENCH_DATA_DIR="$data_dir"
export PYTHONIOENCODING="utf-8"
export NO_COLOR="1"
export CLICOLOR="0"
export PY_COLORS="0"
export PYTHONUNBUFFERED="1"

if [[ "$model_names" == *,* ]]; then
    echo "Only one model may be selected per run; do not use comma-separated model names." >&2
    exit 2
fi

requested_model="$model_names"
model_family=""
model_args=""
model_suffix="$requested_model"

case "$requested_model" in
gemini_1p5_flash)
    model_family="gemini_api"
    model_args="model_version=gemini-1.5-flash,modality=video,context_cache=${context_cache}"
    ;;
gemini_1p5_pro_002)
    model_family="gemini_api"
    model_args="model_version=gemini-1.5-pro,modality=video,context_cache=${context_cache}"
    ;;
gemini_2p0_flash_exp)
    model_family="gemini_api"
    model_args="model_version=gemini-2.0-flash-exp,modality=video,context_cache=${context_cache}"
    ;;
gemini-*|gemini:gemini-*|models/gemini-*|gemini:models/gemini-*)
    gemini_model_version="${requested_model#gemini:}"
    if [[ ! "$gemini_model_version" =~ ^(gemini-[A-Za-z0-9._-]+|models/gemini-[A-Za-z0-9._-]+)$ ]]; then
        echo "Invalid Gemini model ID: $gemini_model_version" >&2
        exit 2
    fi
    model_family="gemini_api"
    model_suffix="${gemini_model_version//[^A-Za-z0-9._-]/_}"
    model_args="model_version=${gemini_model_version},modality=video,context_cache=${context_cache}"
    ;;
llava_one_vision_qwen2_0p5b_ov)
    model_family="llava_onevision"
    model_suffix="${requested_model}_${num_frames}f"
    model_args="pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=${num_frames},device=cpu,device_map=auto"
    ;;
*)
    echo "Unknown model: $requested_model" >&2
    exit 2
    ;;
esac

if [[ "$model_family" == "gemini_api" ]]; then
    model_args+=",thinking=${thinking},backend=${backend}"
fi

if [[ "$model_family" == "gemini_api" && -n "$resolution_mode" ]]; then
    model_args+=",resolution_mode=${resolution_mode}"
fi

if [[ "$model_family" == "gemini_api" && -n "$resume_dir" ]]; then
    model_args+=",continual_mode=true,response_persistent_folder=${resume_dir}"
fi

case "$backend" in
    api|vertex)
        ;;
    *)
        echo "Invalid --backend: $backend (expected api or vertex)" >&2
        exit 2
        ;;
esac

for value in "$gcp_project" "$gcp_location" "$gcs_bucket" "$resume_dir"; do
    if [[ "$value" == *,* || "$value" == *=* ]]; then
        echo "Values passed to Gemini may not contain ',' or '=': $value" >&2
        exit 2
    fi
done

if [[ "$model_family" == "gemini_api" && "$backend" == "vertex" ]]; then
    [[ -n "$gcp_project" ]] || { echo "Vertex backend: set --gcp_project or GOOGLE_CLOUD_PROJECT" >&2; exit 2; }
    [[ -n "$gcs_bucket" ]] || { echo "Vertex backend: set --gcs_bucket or VSI_BENCH_GCS_BUCKET" >&2; exit 2; }
    model_args+=",gcp_project=${gcp_project},gcp_location=${gcp_location},gcs_bucket=${gcs_bucket}"
    # Vertex uses Application Default Credentials (the VM service account), not an API key.
    unset GOOGLE_API_KEY
elif [[ "$model_family" == "gemini_api" ]]; then
    if [[ ! -f "$google_api_key_file" ]]; then
        echo "Google API key file not found: $google_api_key_file" >&2
        echo "Create it with the key on a single line." >&2
        exit 2
    fi
    google_api_key="$(<"$google_api_key_file")"
    google_api_key="${google_api_key//$'\r'/}"
    google_api_key="${google_api_key//$'\n'/}"
    if [[ -z "$google_api_key" ]]; then
        echo "Google API key file is empty: $google_api_key_file" >&2
        exit 2
    fi
    if [[ "$google_api_key" == gen-lang-client-* ]]; then
        echo "The Google credential file contains a project/client identifier, not an API key." >&2
        echo "Create an API key in Google AI Studio and replace the contents of $google_api_key_file." >&2
        exit 2
    fi
    if [[ "$google_api_key" == YOUR_GOOGLE_API_KEY ]]; then
        echo "Replace the placeholder in $google_api_key_file with a real Google AI Studio API key." >&2
        exit 2
    fi
    export GOOGLE_API_KEY="$google_api_key"
else
    unset GOOGLE_API_KEY
fi

for task in "${tasks[@]}"; do
    command=(python -u -m lmms_eval
        --model "$model_family"
        --model_args "$model_args"
        --tasks "$task"
        --batch_size 1
        --log_samples
        --log_samples_suffix "$model_suffix"
        --output_path "$output_path/$task")

    if [[ "$launcher" == "accelerate" ]]; then
        command=(accelerate launch "--num_processes=$num_processes" "${command[@]}")
    fi

    if [[ -n "$limit" ]]; then
        command+=(--limit "$limit")
    fi

    task_output_path="$output_path/$task"
    run_log="$task_output_path/run.log"
    mkdir -p "$task_output_path"
    {
        printf 'Starting model=%s task=%s\n' "$requested_model" "$task"
        printf 'Output: %s\n' "$task_output_path"
        "${command[@]}"
    } 2>&1 | sed -u -E $'s/\x1B\\[[0-9;]*[[:alpha:]]//g' | tee "$run_log"
done
