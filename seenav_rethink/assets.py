"""Check the same full Qwen2.5-VL export as the method evaluator."""

from pathlib import Path

from .data import check, load_json


def model_files(model):
    config = load_json(model / "config.json")
    check(config["model_type"] == "qwen2_5_vl", "expected Qwen2.5-VL")
    required = {"config.json", "preprocessor_config.json", "tokenizer_config.json"}
    index = model / "model.safetensors.index.json"
    if index.exists():
        required.add(index.name)
        weights = set(load_json(index)["weight_map"].values())
    else:
        weights = {"model.safetensors"}
    check(weights and all(Path(n).name == n and n.endswith(".safetensors") for n in weights),
          "missing or unsafe weight shards")
    check(weights == {p.name for p in model.glob("*.safetensors")}, "missing or mixed weight shards")
    check(required <= {p.name for p in model.iterdir() if p.is_file()}, "processor/model files missing")
    return weights
