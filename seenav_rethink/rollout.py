"""Small driver-side bridge; neither observations nor rewards are edited."""

from .protocol import decode_decision, for_execution, marker_token_id


def execution_texts(tokenizer, responses, attention_mask):
    """Give the native parser only well-formed decisions; invalid means no-op."""
    marker = marker_token_id(tokenizer)
    active = attention_mask[:, -responses.shape[-1]:].bool()
    result = []
    for ids, mask in zip(responses, active, strict=True):
        decoded = decode_decision(tokenizer, ids[mask].tolist(), marker)
        result.append(for_execution(decoded.text) if decoded.parsed is not None else "")
    return result
