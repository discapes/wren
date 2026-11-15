"""Label normalization and the room category vocabulary (vocab.yaml)."""
import re
import unicodedata
from pathlib import Path

import yaml


def norm(label):
    """Casefolded, accent-free label without digits, parentheses or punctuation other than / and '."""
    s = unicodedata.normalize("NFKD", re.sub(r"\(.*?\)", "", label).casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w/' ]|\d|_", " ", s)
    return " ".join(s.split())


_V = yaml.safe_load((Path(__file__).parent / "vocab.yaml").read_text())
ROOMS = {norm(label): cat for cat, labels in _V["rooms"].items() for label in labels}
OUTDOOR = set(_V["outdoor"])


def room_cat(label):
    return ROOMS.get(norm(label or ""), "unknown")
