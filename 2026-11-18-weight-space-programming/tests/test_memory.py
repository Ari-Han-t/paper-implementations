import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsp.data import make_world, Question  # noqa: E402
from wsp.memory import DenseAttentionMemory, SparseTableMemory  # noqa: E402

FACTS = ["Asha Rao lives in Pune.", "Asha Iyer lives in Oslo.",
         "Asha Rao was born in Lyon.", "Ravi Das works as a chef."]


def test_parse_splits_subject_predicate_object():
    row = SparseTableMemory.parse("Asha Rao works as a chef.")
    assert row.subject == ("Asha", "Rao")
    assert row.predicate == ("works", "as")
    assert row.value == "chef"


def test_full_name_match_disambiguates_shared_first_name():
    m = SparseTableMemory().compile(FACTS)
    opts = ["Pune", "Oslo", "Kyoto", "Perth"]
    assert opts[m.predict("Where does Asha Iyer live?", opts)] == "Oslo"


def test_cues_from_examples_disambiguate_same_answer_pool():
    m = SparseTableMemory().compile(FACTS)
    opts = ["Pune", "Lyon", "Oslo", "Perth"]
    train = [Question("Where was Asha Rao born?", opts, 1, "Asha Rao", "born"),
             Question("Where does Asha Rao live?", opts, 0, "Asha Rao", "city")]
    m.fit_cues(train)
    assert opts[m.predict("Where was Asha Rao born?", opts)] == "Lyon"
    assert opts[m.predict("Where does Asha Rao live?", opts)] == "Pune"


def test_explain_returns_the_row_used_and_abstains_without_row():
    m = SparseTableMemory().compile(FACTS)
    _, rid = m.explain("What does Ravi Das do?", ["chef", "pilot", "nurse", "baker"])
    assert m.rows[rid].source == "Ravi Das works as a chef."
    _, rid = m.explain("What does Unknown Person do?", ["chef", "pilot"])
    assert rid is None


def test_edit_and_delete_are_local():
    m = SparseTableMemory().compile(FACTS)
    opts = ["chef", "pilot", "nurse", "baker"]
    _, rid = m.explain("What does Ravi Das do?", opts)
    m.edit_row(rid, "pilot")
    assert opts[m.predict("What does Ravi Das do?", opts)] == "pilot"
    city_opts = ["Pune", "Oslo", "Kyoto", "Perth"]
    assert city_opts[m.predict("Where does Asha Iyer live?", city_opts)] == "Oslo"
    row = m.delete_row(rid)
    assert m.explain("What does Ravi Das do?", opts)[1] is None
    m.restore_row(rid, row)
    assert m.explain("What does Ravi Das do?", opts)[1] == rid


def test_world_is_deterministic_and_train_is_clean():
    a, b = make_world(seed=3), make_world(seed=3)
    assert a.facts_text == b.facts_text
    assert len(a.train) == 10
    for q in a.train:
        assert (q.subject, q.relation) not in a.corrupted | a.missing


def test_sparse_beats_dense_and_is_much_smaller():
    w = make_world(n_people=60, seed=0)
    s = SparseTableMemory().compile(w.facts_text)
    s.fit_cues(w.train)
    d = DenseAttentionMemory().compile(w.facts_text)
    acc = lambda m: sum(m.predict(q.text, q.options) == q.answer for q in w.test) / len(w.test)
    assert acc(s) > 0.75
    assert acc(s) > acc(d)
    assert d.n_params() > 10 * s.n_params()
