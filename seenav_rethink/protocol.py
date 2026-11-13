"""One reasoning span, one optional pre-action marker, one native move."""

from dataclasses import dataclass
import re

MARKER = "<RETHINK>"
EOS = "<|im_end|>"
_BOUNDARY = r"[\s\u00ad]*"
_PATTERN = re.compile(
    rf"\s*<think>(?P<reasoning>.*?)</think>{_BOUNDARY}"
    rf"(?:(?P<marker>{MARKER}){_BOUNDARY})?"
    r"<action>\s*(?P<action>[0-7])\s*</action>\s*\Z",
    re.S,
)
_CONTROL = re.compile(r"</?think>|</?action>|<RETHINK>|<\|[^<>]*\|>", re.I)
_BETWEEN_TAGS = re.compile(r"(</think>|<RETHINK>)([\s\u00ad]+)(?=<RETHINK>|<action>)")
_NATIVE_INVALID_CHARS = re.compile(r"[\u4e00-\u9fff]")


def parse_decision(text):
    """Do not infer a marker from reasoning text, or route malformed responses."""
    text = re.sub(r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>))+\s*\Z", "", text)
    match = _PATTERN.fullmatch(text)
    if not match:
        return None
    reason = match["reasoning"].strip()
    if not reason or _CONTROL.search(reason):
        return None
    return reason, bool(match["marker"]), int(match["action"])


def for_execution(text):
    """Treat SHY only at the tag boundaries as transparent; do not repair moves."""
    return _BETWEEN_TAGS.sub(lambda m: m[1] + m[2].replace("\u00ad", ""), text)


def marker_token_id(tokenizer):
    ids = tokenizer.encode(MARKER, add_special_tokens=False)
    return ids[0] if len(ids) == 1 and ids[0] in tokenizer.all_special_ids else None


@dataclass(frozen=True)
class DecodedDecision:
    text: str
    marker_count: int
    parsed: tuple[str, bool, int] | None

    @property
    def action(self):
        return self.parsed[2] if self.parsed is not None else -1

    @property
    def routed(self):
        return self.parsed is not None and self.parsed[1]


def decode_decision(tokenizer, ids, marker):
    """Classify the actual response IDs, never text with specials stripped."""
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    count = ids.count(marker) if marker is not None else 0
    parsed = parse_decision(text)
    # Keep the original SeeNav tag parser's character restriction. A marker
    # spelled by normal text tokens is not the registered control token.
    if parsed is not None and (count != int(parsed[1]) or _NATIVE_INVALID_CHARS.search(text)):
        parsed = None
    return DecodedDecision(text, count, parsed)
