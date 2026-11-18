"""Synthetic multiple-choice QA world.

A small fictional knowledge base is rendered as plain-text sentences (the
"facts" a weight-programmer reads). A fraction of the sentences are
deliberately corrupted (wrong object) so that some answers are wrong in a way
that can be traced back to a single table row, and a fraction of entities have
no fact at all (unanswerable from the text).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

FIRST = ["Asha", "Ravi", "Meera", "Kabir", "Nina", "Omar", "Lena", "Arjun",
         "Tara", "Felix", "Zoya", "Ivan", "Priya", "Hugo", "Sana", "Dev"]
LAST = ["Rao", "Iyer", "Khan", "Silva", "Novak", "Mehta", "Berg", "Das",
        "Okafor", "Lund", "Ito", "Costa"]
CITIES = ["Vellore", "Pune", "Oslo", "Lisbon", "Nairobi", "Kyoto", "Quito",
          "Tallinn", "Hanoi", "Perth", "Austin", "Lyon"]
JOBS = ["engineer", "pilot", "chef", "doctor", "painter", "lawyer", "farmer",
        "teacher", "nurse", "architect", "baker", "writer"]
PETS = ["cat", "dog", "parrot", "rabbit", "turtle", "hamster", "goldfish",
        "ferret"]

# relation -> (fact templates, question templates, answer pool)
RELATIONS = {
    "city": (["{s} lives in {o}.", "{s} has a home in {o}."],
             ["Where does {s} live?", "Which city is home to {s}?",
              "In which city does {s} reside?"], CITIES),
    # same answer pool as "city": options alone cannot tell these two apart,
    # so the query procedure must learn question-word -> predicate cues.
    "born": (["{s} was born in {o}.", "{s} is a native of {o}."],
             ["Where was {s} born?", "What is the birthplace of {s}?",
              "Which city was {s} born in?"], CITIES),
    "job": (["{s} works as a {o}.", "{s} is employed as a {o}."],
            ["What is the profession of {s}?", "What does {s} do for work?",
             "Which job does {s} have?"], JOBS),
    "pet": (["{s} owns a {o}.", "{s} keeps a pet {o}."],
            ["What pet does {s} own?", "Which animal does {s} keep?",
             "What kind of pet belongs to {s}?"], PETS),
}


@dataclass
class Question:
    text: str
    options: list[str]
    answer: int          # index of the gold option
    subject: str
    relation: str


@dataclass
class World:
    facts_text: list[str]                 # plain-text facts given to the compiler
    truth: dict = field(default_factory=dict)       # (s, r) -> true object
    corrupted: set = field(default_factory=set)     # (s, r) whose sentence is wrong
    missing: set = field(default_factory=set)       # (s, r) with no sentence
    train: list[Question] = field(default_factory=list)
    test: list[Question] = field(default_factory=list)


def make_world(n_people: int = 60, p_corrupt: float = 0.08,
               p_missing: float = 0.08, n_train: int = 10,
               seed: int = 0) -> World:
    rng = random.Random(seed)
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(names)
    people = names[:n_people]

    world = World(facts_text=[])
    for s in people:
        for r, (ftpl, _, pool) in RELATIONS.items():
            o = rng.choice(pool)
            world.truth[(s, r)] = o
            u = rng.random()
            if u < p_missing:
                world.missing.add((s, r))
                continue
            shown = o
            if u < p_missing + p_corrupt:
                shown = rng.choice([x for x in pool if x != o])
                world.corrupted.add((s, r))
            world.facts_text.append(rng.choice(ftpl).format(s=s, o=shown))
    rng.shuffle(world.facts_text)

    qs = []
    for (s, r), o in world.truth.items():
        _, qtpl, pool = RELATIONS[r]
        distract = rng.sample([x for x in pool if x != o], 3)
        opts = distract + [o]
        rng.shuffle(opts)
        qs.append(Question(rng.choice(qtpl).format(s=s), opts, opts.index(o), s, r))
    rng.shuffle(qs)
    # the 10 labeled examples come only from clean, present facts
    clean = [q for q in qs if (q.subject, q.relation) not in world.corrupted
             and (q.subject, q.relation) not in world.missing]
    world.train = clean[:n_train]
    train_ids = {id(q) for q in world.train}
    world.test = [q for q in qs if id(q) not in train_ids]
    return world
