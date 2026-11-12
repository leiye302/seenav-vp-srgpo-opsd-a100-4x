"""Re-score the original sampled tokens with a static navigation prefix."""

from pathlib import Path

import torch


SKILL_FILE = Path(__file__).with_name("navigation_skill.txt")


def prefix_ids(tokenizer):
    skill = SKILL_FILE.read_text(encoding="utf-8").strip()
    ids = tokenizer.encode(f"[Privileged Skill Information]\n{skill}\n\n", add_special_tokens=False)
    if not ids or len(ids) > 256 or set(ids) & set(tokenizer.all_special_ids):
        raise ValueError("Expected a short text-only navigation prefix.")
    return ids


def pack_teacher_tensors(student, prefix, max_model_len):
    """Insert text after the left pad, retaining all native tokens and images.

    Text preceding the existing prompt shifts its RoPE positions by the
    prefix length. Keep the native position layout (1D or 3-axis MRoPE);
    neither re-render images nor re-tokenize the original prompt/response.
    """
    ids = student["input_ids"]
    mask = student["attention_mask"]
    pos = student["position_ids"]
    responses = student["responses"]
    if ids.ndim != 2 or mask.shape != ids.shape or pos.shape[0] != ids.shape[0] or pos.shape[-1] != ids.shape[-1]:
        raise ValueError("Misaligned native token, attention or position tensors.")
    if pos.ndim not in (2, 3) or (pos.ndim == 3 and pos.shape[1] != 3):
        raise ValueError("Expected the native scalar or 3-axis position layout.")
    if responses.ndim != 2 or responses.shape[0] != ids.shape[0]:
        raise ValueError("Misaligned native response batch.")
    r = responses.shape[-1]
    p = ids.shape[-1] - r
    if r < 1 or p < 1 or not torch.equal(ids[:, -r:], responses):
        raise ValueError("The native response must be the unchanged input suffix.")
    left_pads = p - mask[:, :p].sum(-1).long()
    expected = torch.arange(p, device=ids.device)[None, :] >= left_pads[:, None]
    if not torch.equal(mask[:, :p].bool(), expected) or bool((left_pads >= p).any()):
        raise ValueError("Expected nonempty left-padded native prompts.")
    k = len(prefix)
    if k < 1 or ids.shape[-1] + k > max_model_len:
        raise ValueError("Teacher prefix exceeds the native context limit; no truncation allowed.")
    expanded = ids.shape[-1] + k
    teacher_ids = ids.new_empty((ids.shape[0], expanded))
    teacher_mask = mask.new_empty((ids.shape[0], expanded))
    teacher_pos = pos.new_empty((*pos.shape[:-1], expanded))
    prefix_tensor = ids.new_tensor(prefix)
    prefix_pos = torch.arange(k, device=pos.device, dtype=pos.dtype)
    shifted = pos + k * (mask if pos.ndim == 2 else mask.unsqueeze(1)).to(pos.dtype)
    for i, left in enumerate(left_pads.tolist()):
        start = left + k
        teacher_ids[i, :left] = ids[i, :left]
        teacher_ids[i, left:start] = prefix_tensor
        teacher_ids[i, start:] = ids[i, left:]
        teacher_mask[i, :left] = mask[i, :left]
        teacher_mask[i, left:start] = 1
        teacher_mask[i, start:] = mask[i, left:]
        teacher_pos[i, ..., :left] = pos[i, ..., :left]
        teacher_pos[i, ..., left:start] = prefix_pos
        teacher_pos[i, ..., start:] = shifted[i, ..., left:]
    return {"input_ids": teacher_ids, "attention_mask": teacher_mask,
            "position_ids": teacher_pos, "responses": responses}


def attach_teacher_scores(trainer, batch):
    """Driver-only hook, after rollout/advantages and before the Actor update."""
    from verl import DataProto

    images = batch.non_tensor_batch.get("multi_modal_inputs")
    if images is None or len(images) != len(batch.batch["responses"]):
        raise ValueError("Native VP image inputs must be present for Teacher scoring.")
    actor = trainer.config.actor_rollout_ref
    if actor.rollout.multi_turn.enable:
        raise ValueError("This adapter expects the native per-decision EB-Nav rollout.")
    prefix = prefix_ids(trainer.tokenizer)
    tensors = pack_teacher_tensors(batch.batch, prefix, int(actor.rollout.max_model_len))
    # Do not copy rollout metadata (including any reference/adapter mode).
    # Same Actor worker, same image tensors, no separate model, no generation.
    teacher_batch = DataProto.from_dict(tensors=tensors, non_tensors={"multi_modal_inputs": images})
    scored = trainer.actor_rollout_wg.compute_log_prob(teacher_batch)
    teacher = scored.batch["old_log_probs"].detach().to(batch.batch["old_log_probs"].device)
    r = batch.batch["responses"].shape[-1]
    mask = batch.batch["attention_mask"][:, -r:].bool()
    if teacher.shape != mask.shape or not bool(mask.any()) or not bool(torch.isfinite(teacher[mask]).all()):
        raise ValueError("Teacher scores do not cover the native response mask.")
    batch.batch["teacher_log_probs"] = teacher
    return {"sdar/teacher_rows": int(teacher.shape[0]), "sdar/teacher_prefix_tokens": len(prefix)}
