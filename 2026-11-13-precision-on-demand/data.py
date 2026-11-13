"""Tiny, deterministic, offline text corpus: docstrings from the Python stdlib.

No downloads needed. The corpus is filtered to printable ASCII so the
character vocabulary stays small.
"""
import importlib
import inspect
import string

MODULES = [
    "collections", "functools", "itertools", "heapq", "bisect", "json", "csv",
    "argparse", "logging", "threading", "queue", "asyncio", "socket", "http.client",
    "urllib.parse", "email.message", "pathlib", "os", "shutil", "subprocess",
    "textwrap", "string", "re", "random", "statistics", "decimal", "fractions",
    "datetime", "calendar", "unittest", "doctest", "pickle", "sqlite3", "zipfile",
    "tarfile", "gzip", "hashlib", "hmac", "secrets", "base64", "typing", "dataclasses",
    "enum", "contextlib", "abc", "copy", "pprint", "inspect", "ast", "dis", "tokenize",
    "difflib", "configparser", "tempfile", "glob", "fnmatch", "selectors", "ssl",
]

ALLOWED = set(string.printable) - set("\x0b\x0c\r\t")


def _docstrings(obj, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 2:
        return []
    seen.add(id(obj))
    out = []
    doc = inspect.getdoc(obj)
    if doc:
        out.append(doc)
    if inspect.ismodule(obj) or inspect.isclass(obj):
        for name, member in sorted(vars(obj).items()):
            if name.startswith("_"):
                continue
            if inspect.isfunction(member) or inspect.isclass(member) or inspect.ismethod(member):
                out.extend(_docstrings(member, depth + 1, seen))
    return out


def build_corpus(max_chars=600_000):
    chunks = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        chunks.extend(_docstrings(mod))
    text = "\n\n".join(chunks)
    text = "".join(c for c in text if c in ALLOWED)
    return text[:max_chars]


class CharVocab:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)
