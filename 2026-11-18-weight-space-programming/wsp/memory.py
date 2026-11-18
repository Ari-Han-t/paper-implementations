"""Hand-written ("weight-space programmed") models for multiple-choice QA.

SparseTableMemory  -- the structure the paper reports frontier LLMs converge to
                      when free to design the model: plain-text facts compiled
                      into a sparse lookup table + a fixed query procedure.
DenseAttentionMemory -- a dense, Transformer-style single attention layer whose
                      weights are also written by hand from the same facts
                      (random token embeddings, keys = summed embeddings,
                      values = one-hot answers). Used as the dense baseline.

No gradient descent is used anywhere.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

_TOK = re.compile(r"[A-Za-z0-9']+")
STOP = {"a", "an", "the", "of", "to", "does", "do", "is", "what", "which",
        "where", "in", "kind", "for", "was"}


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text)


# --------------------------------------------------------------------------
# Sparse table
# --------------------------------------------------------------------------
@dataclass
class Row:
    subject: tuple[str, ...]      # key part 1: entity tokens (case-sensitive)
    predicate: tuple[str, ...]    # key part 2: predicate tokens (lowercased)
    value: str                    # the stored answer
    source: str                   # the sentence this row was compiled from


class SparseTableMemory:
    """Facts -> rows. Each row is ~6 non-zero 'weights'."""

    def __init__(self):
        self.rows: list[Row | None] = []
        self.index: dict[str, set[int]] = defaultdict(set)   # subject token -> row ids
        # cue weights learned from the labeled examples: (question word, predicate) -> count
        self.cue: dict[tuple[str, tuple[str, ...]], float] = defaultdict(float)

    # ---- compiler: plain text -> table ---------------------------------
    @staticmethod
    def parse(sentence: str) -> Row:
        toks = tokenize(sentence)
        i = 0
        while i < len(toks) and toks[i][0].isupper():
            i += 1
        subject = tuple(toks[:i])
        obj = toks[-1]
        pred = tuple(t.lower() for t in toks[i:-1] if t.lower() not in {"a", "an", "the"})
        return Row(subject, pred, obj, sentence)

    def compile(self, sentences: list[str]) -> "SparseTableMemory":
        for s in sentences:
            self.add_row(self.parse(s))
        return self

    def add_row(self, row: Row) -> int:
        rid = len(self.rows)
        self.rows.append(row)
        for t in row.subject:
            self.index[t].add(rid)
        return rid

    # ---- use the 10 labeled examples to learn question-word -> predicate cues
    def fit_cues(self, examples) -> None:
        for q in examples:
            gold = q.options[q.answer]
            for rid in self._subject_rows(q.text):
                row = self.rows[rid]
                if row.value == gold:
                    for w in self._content_words(q.text, row.subject):
                        self.cue[(w, row.predicate)] += 1.0
                    # also tie predicate words that appear in the question itself
        # predicates never seen with a cue get a weak lexical-overlap fallback (see score)

    # ---- fixed query procedure ------------------------------------------
    @staticmethod
    def _content_words(text, subject):
        return [t.lower() for t in tokenize(text)
                if t not in subject and t.lower() not in STOP]

    def _subject_rows(self, text: str) -> list[int]:
        toks = set(tokenize(text))
        cand = set()
        for t in toks:
            cand |= self.index.get(t, set())
        out = []
        for rid in cand:
            row = self.rows[rid]
            if row is not None and set(row.subject) <= toks:   # full-name match
                out.append(rid)
        return sorted(out)

    def _score(self, text, row):
        words = self._content_words(text, row.subject)
        s = sum(self.cue.get((w, row.predicate), 0.0) for w in words)
        s += 0.1 * len(set(words) & set(row.predicate))      # lexical fallback
        return s

    def explain(self, text: str, options: list[str]):
        """Return (chosen option index, row id used or None)."""
        cands = [rid for rid in self._subject_rows(text) if self.rows[rid].value in options]
        if not cands:
            return 0, None                       # abstain -> first option
        best = max(cands, key=lambda rid: (self._score(text, self.rows[rid]), -rid))
        return options.index(self.rows[best].value), best

    def predict(self, text: str, options: list[str]) -> int:
        return self.explain(text, options)[0]

    # ---- editing -----------------------------------------------------------
    def delete_row(self, rid: int) -> Row:
        row = self.rows[rid]
        self.rows[rid] = None
        for t in row.subject:
            self.index[t].discard(rid)
        return row

    def restore_row(self, rid: int, row: Row) -> None:
        self.rows[rid] = row
        for t in row.subject:
            self.index[t].add(rid)

    def edit_row(self, rid: int, new_value: str) -> None:
        self.rows[rid].value = new_value

    # ---- size --------------------------------------------------------------
    def n_params(self) -> int:
        """Non-zero entries: subject tokens + predicate tokens + 1 value per row,
        plus the learned cue weights."""
        return sum(len(r.subject) + len(r.predicate) + 1 for r in self.rows if r) + len(self.cue)


# --------------------------------------------------------------------------
# Dense attention baseline
# --------------------------------------------------------------------------
class DenseAttentionMemory:
    """One softmax-attention layer with hand-written dense weights.

    E : vocab x d random token embeddings (fixed, written not trained)
    K : rows x d keys = normalised sum of subject+predicate embeddings
    V : rows x A one-hot answer values
    y = softmax(q K^T / tau) V,  q = normalised sum of question embeddings
    """

    def __init__(self, d: int = 128, tau: float = 0.05, seed: int = 0):
        self.d, self.tau = d, tau
        self.rng = np.random.default_rng(seed)
        self.vocab: dict[str, int] = {}
        self.answers: dict[str, int] = {}
        self._E = []

    def _emb(self, tok: str) -> np.ndarray:
        tok = tok.lower()
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab)
            self._E.append(self.rng.standard_normal(self.d) / np.sqrt(self.d))
        return self._E[self.vocab[tok]]

    def _enc(self, toks) -> np.ndarray:
        toks = [t for t in toks if t.lower() not in STOP]
        v = np.sum([self._emb(t) for t in toks], axis=0) if toks else np.zeros(self.d)
        return v / (np.linalg.norm(v) + 1e-9)

    def compile(self, sentences: list[str]) -> "DenseAttentionMemory":
        keys, vals = [], []
        for s in sentences:
            row = SparseTableMemory.parse(s)
            keys.append(self._enc(list(row.subject) + list(row.predicate)))
            vals.append(self.answers.setdefault(row.value, len(self.answers)))
        self.K = np.stack(keys)
        self.V = np.zeros((len(vals), len(self.answers)))
        self.V[np.arange(len(vals)), vals] = 1.0
        return self

    def predict(self, text: str, options: list[str]) -> int:
        q = self._enc(tokenize(text))
        s = self.K @ q / self.tau
        a = np.exp(s - s.max())
        a /= a.sum()
        y = a @ self.V
        scores = [y[self.answers[o]] if o in self.answers else -1 for o in options]
        return int(np.argmax(scores))

    def n_params(self) -> int:
        return len(self.vocab) * self.d + self.K.size + self.V.size
