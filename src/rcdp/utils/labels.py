from __future__ import annotations


def canonicalize_classname(name: str) -> str:
    # dataset folder names -> readable class label for prompt
    s = name.replace("_", " ").replace("-", " ").strip()
    s = " ".join(s.split())
    return s.lower()
