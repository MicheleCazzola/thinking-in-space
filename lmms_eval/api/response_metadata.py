"""Per-sample response metadata shared between a model wrapper and a task's process_results.

lmms-eval only passes the generated text from the model to the task. API-based models can
record extra per-sample information here (API status, attempts, token usage, ...), keyed by
the dataset's own question id, and the task can attach it to the logged sample.
Generation finishes before process_results runs, in the same process, so a module-level
registry is enough.
"""

_RECORDS = {}


def record(sample_key, metadata):
    _RECORDS[str(sample_key)] = dict(metadata)


def lookup(sample_key):
    return _RECORDS.get(str(sample_key))
