
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
from collections import OrderedDict

import datasets

from lmms_eval.api import response_metadata

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
local_data_dir = os.getenv("VSI_BENCH_DATA_DIR")
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def vsibench_doc_to_visual(doc):
    cache_dir = os.path.expanduser(local_data_dir) if local_data_dir else os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    if doc['question_type'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['question_type'] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

def process_docs_tiny(dataset: datasets.Dataset) -> datasets.Dataset:
    tiny_ids_path = Path(__file__).with_name("vsibench_tiny_ids.txt")
    with tiny_ids_path.open() as file:
        tiny_ids = {int(line.strip()) for line in file if line.strip()}
    dataset = dataset.filter(lambda doc: doc["id"] in tiny_ids)
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        dataset = dataset.shuffle(seed=42)
    eval_logger.info(f"VSI-Bench tiny subset contains {len(dataset)} available questions.")
    return dataset

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def vsibench_process_results(doc, results):

    doc['prediction'] = results[0]
    if doc['question_type'] in MCA_QUESTION_TYPES:
        # The value actually compared with the ground truth.
        doc['parsed_prediction'] = fuzzy_matching(doc['prediction'])
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        doc['parsed_prediction'] = to_float(fuzzy_matching(doc['prediction']))  # None = unparsable
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    # Per-sample API status / attempts / token usage recorded by API models (None otherwise).
    doc['response_meta'] = response_metadata.lookup(doc.get('id'))

    return {"vsibench_score": doc}


OFFICIAL_QUESTION_TYPE_ORDER = [
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction",
    "route_planning",
    "obj_appearance_order",
]
REL_DIRECTION_SUBTYPES = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"]


def _metric_column(question_type):
    return "accuracy" if question_type in MCA_QUESTION_TYPES else "MRA:.5:.95:.05"


def _macro_scores(df):
    """Official-style scores (x100) on any subset of samples: per question type, with
    rel_direction = mean of the subtypes present, and overall = mean of the question types.
    Robust to missing question types (unlike the official function, which needs all three
    rel_direction subtypes as soon as one is present)."""
    per_type = {}
    for question_type, group in df.groupby("question_type"):
        per_type[question_type] = float(group[_metric_column(question_type)].mean()) * 100.0
    rel = [per_type.pop(t) for t in REL_DIRECTION_SUBTYPES if t in per_type]
    if rel:
        per_type["object_rel_direction"] = sum(rel) / len(rel)
    ordered = OrderedDict((t, per_type[t]) for t in OFFICIAL_QUESTION_TYPE_ORDER if t in per_type)
    overall = sum(ordered.values()) / len(ordered) if ordered else None
    return overall, ordered


def _details(df):
    """Extra breakdowns and run diagnostics. The official keys are not affected."""
    details = OrderedDict()
    details["n_samples"] = int(len(df))
    counts = df["question_type"].value_counts()
    n_per_type = OrderedDict()
    for t in OFFICIAL_QUESTION_TYPE_ORDER:
        if t == "object_rel_direction":
            n = int(sum(counts.get(s, 0) for s in REL_DIRECTION_SUBTYPES))
        else:
            n = int(counts.get(t, 0))
        if n:
            n_per_type[t] = n
    for s in REL_DIRECTION_SUBTYPES:
        if counts.get(s, 0):
            n_per_type[s] = int(counts[s])
    details["n_per_question_type"] = n_per_type

    # rel_direction subtypes, which the official aggregation averages into one value.
    details["object_rel_direction_subtypes_accuracy"] = OrderedDict(
        (s, float(df.loc[df["question_type"] == s, "accuracy"].mean()) * 100.0)
        for s in REL_DIRECTION_SUBTYPES
        if counts.get(s, 0)
    )

    # Means over the multiple-choice and the numerical question types (macro, as overall).
    _, per_type = _macro_scores(df)
    mca = [v for t, v in per_type.items() if t not in NA_QUESTION_TYPES]
    na = [v for t, v in per_type.items() if t in NA_QUESTION_TYPES]
    details["mca_mean_accuracy"] = sum(mca) / len(mca) if mca else None
    details["na_mean_MRA"] = sum(na) / len(na) if na else None

    # Same scores per source dataset (arkitscenes / scannet / scannetpp).
    per_dataset = OrderedDict()
    for dataset_name, group in df.groupby("dataset"):
        overall, scores = _macro_scores(group)
        per_dataset[dataset_name] = OrderedDict(
            [("n_samples", int(len(group))), ("overall", overall), *scores.items()]
        )
    details["per_dataset"] = per_dataset

    # Answer parsing: numerical answers that could not be read as a number score 0.
    na_mask = df["question_type"].isin(NA_QUESTION_TYPES)
    if "parsed_prediction" in df:
        details["n_unparsable_numerical_answers"] = int(df.loc[na_mask, "parsed_prediction"].isna().sum())

    # API diagnostics recorded by API models (e.g. gemini_api).
    metas = [m for m in df.get("response_meta", pd.Series(dtype=object)).tolist() if isinstance(m, dict)]
    if metas:
        status = pd.Series([m.get("status", "unknown") for m in metas]).value_counts()
        tokens = {k: sum(int(m.get(k) or 0) for m in metas) for k in ("prompt_tokens", "output_tokens", "thoughts_tokens")}
        answered = int(status.get("ok", 0))
        details["api"] = OrderedDict(
            [
                ("n_with_metadata", len(metas)),
                ("status_counts", {k: int(v) for k, v in status.items()}),
                ("n_no_answer_api_error", int(len(metas) - answered)),
                ("n_from_resume_cache", int(sum(1 for m in metas if m.get("from_resume_cache")))),
                ("thinking_modes", sorted({str(m.get("thinking_mode")) for m in metas})),
                ("tokens", tokens),
                ("mean_thoughts_tokens_per_answer", tokens["thoughts_tokens"] / answered if answered else None),
                ("n_answers_with_thinking", int(sum(1 for m in metas if (m.get("thoughts_tokens") or 0) > 0))),
            ]
        )
    return details

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    results_df = results

    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")

    if 'object_rel_direction_easy_accuracy' in output:
        output['object_rel_direction_accuracy'] = sum([
            output.pop('object_rel_direction_easy_accuracy'),
            output.pop('object_rel_direction_medium_accuracy'),
            output.pop('object_rel_direction_hard_accuracy'),
        ]) / 3.

    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")

    results = OrderedDict()
    results["overall"] = output["overall"].item() * 100.
    
    for question_type in [
        "object_counting",
        "object_abs_distance",
        "object_size_estimation",
        "room_size_estimation",
        "object_rel_distance",
        "object_rel_direction",
        "route_planning",
        "obj_appearance_order",
    ]:
        for metric in [
            "accuracy",
            "MRA:.5:.95:.05",
        ]:
            key = f"{question_type}_{metric}"
            if key in output:
                results[key] = output[key].item() * 100.

    tabulated_keys = ", ".join([_ for _ in results.keys()])
    tabulated_results = ", ".join([f"{_:.3f}" for _ in results.values()])
    eval_logger.info(f"Tabulated results: {tabulated_keys}")
    eval_logger.info(f"Tabulated results: {tabulated_results}")

    results["tabulated_keys"] = tabulated_keys
    results["tabulated_results"] = tabulated_results

    # Extra breakdowns (counts, rel_direction subtypes, per dataset, API diagnostics).
    # Kept under one key so the official values and the tabulated row are unchanged.
    try:
        results["details"] = _details(results_df)
        eval_logger.info(f"Details: {results['details']}")
    except Exception as exc:  # never lose the official results because of the extras
        eval_logger.warning(f"Could not compute VSI-Bench details: {exc}")

    return results
