"""Small, dependency-free helpers used by the shared evaluator."""

import json


def check(ok, message):
    if not ok:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_bytes())
