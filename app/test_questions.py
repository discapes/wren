"""Acceptance tests: does the app answer the way a site engineer or a Stoa judge would ask?

The dataset templates are scored by score.py. These cases ask about the same drawings in other words, with a scope,
in other languages, or about things the graph already holds, and check that a wrong kind of answer is never given.
Expected values are Stoa's reference answers or were checked by hand on the drawing.

uv run pytest test_questions.py            # ask.py cases, no server needed
uv run pytest test_questions.py -k web     # also the running web UI on 127.0.0.1:8300
"""
import json
import random
import re
import time
import urllib.parse
import urllib.request

import pytest

import ask
import graph
import pdfplan
import web

SWISS = "data/swiss_dwellings/site3534_bldg5416_floor23763.pdf"  # Whg. A (1.01-1.07) and Whg. B (1.08-1.15)
CUBI = "data/cubicasa5k/colorful-12002.png"  # held-out test plan
DUPLEX = "data/ifc_models/bsc-duplex_duplex-a-20110907.ifc"
FZK = "data/ifc_bench/fzk_house_arc.ifc"
A801 = "data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf"
SCAN = "data/wikimedia/100075564.jpg"

NOT_COUNT = "not a count"  # the question asks for something else; a count is a wrong answer, declining is fine


def contains(s):
    return ("contains", s)


CASES = [
    # Same facts as the dataset templates, asked in other words.
    (SWISS, "How many doors are there?", 22),
    (SWISS, "How many apartments are on this floor?", 2),
    (SWISS, "How many bedrooms are on this floor?", 3),
    (SWISS, "What's the floor area of room 1.03?", 23.77),
    (SWISS, "How big is room 1.17?", 11.05),
    (SWISS, "How large is apartment B?", 100.93),
    (SWISS, "What is the total area of all bedrooms on this floor?", 60.04),
    (SWISS, "How many windws are drawn on this floor plan?", 13),
    (SWISS, "¿Cuántas ventanas hay en este plano?", 13),
    (SWISS, "Hoeveel ramen zijn er op deze plattegrond?", 13),
    (CUBI, "How many bedrooms are there?", 1),
    (CUBI, "How many toilets are there?", 1),
    (CUBI, "Is there a sauna?", contains("no")),
    (DUPLEX, "How many doors are in this model?", 14),
    (DUPLEX, "How many doors are on Level 1?", 6),
    (DUPLEX, "How many doors are on 'Level 1'?", 6),
    (DUPLEX, "What is the total floor area of Level 2?", 114.25),
    # Scoped questions: the answer must respect the apartment or room named in the question.
    (SWISS, "How many bedrooms are in apartment B?", 2),
    (SWISS, "How many bedrooms does apartment A have?", 1),
    (SWISS, "How many toilets are in apartment B?", 1),
    (SWISS, "How many windows does apartment B have?", 9),
    (SWISS, "What is the area of the kitchen in apartment A?", 9.86),
    (SWISS, "How many rooms labelled 'Bad' are there in apartment A?", 1),
    (SWISS, "Is there a bathtub in apartment A?", contains("no")),
    (SWISS, "How many windows does kitchen 1.05 have?", 1),
    (CUBI, "How many doors are in the bathroom?", ("not", 7)),
    # Questions a count template must not grab.
    (SWISS, "How wide are the doors?", NOT_COUNT),
    (SWISS, "What is the total area of the bedrooms?", NOT_COUNT),
    (CUBI, "What is the total area of the bedrooms?", NOT_COUNT),
    (SWISS, "Which room is the largest?", contains("1.12")),
    # Open questions the graph already holds the data for.
    (FZK, "How many doors are in the building?", 5),
    (FZK, "Which room in the building is the largest?", contains("Galerie")),
    (FZK, "Which door types include fire rating data?", contains("no")),
    (FZK, "What is the building's total net floor area?", 173.3),
    # Schedule sheets: the reader extracts the door schedule (38 rows) and the finish room schedule (29 rows).
    (A801, "How many doors are listed in the door schedule?", 38),
    (A801, "How many rooms are listed in the finish room schedule?", 29),
    (A801, "What is the area of room 128 (Fieldhouse) in the finish room schedule?", 29690),
    (A801, "How many doors in the door schedule are door type A1?", 14),
    (A801, "What door type is door 121.A?", contains("B3")),
    # A drawing from the half of the dataset that has no questions: scans and historic plans.
    (SCAN, "How many rooms are on this plan?", "any answer"),
]


@pytest.mark.parametrize("path, question, expected", CASES, ids=[f"{p.split('/')[1]}: {q}" for p, q, _ in CASES])
def test_question(path, question, expected):
    a, _, _ = ask.answer(path, question)
    if expected == NOT_COUNT:
        assert a is None or a.unit != "count", f"answered a count: {a.value} ({a.via})"
        return
    assert a is not None, "no answer"
    if isinstance(expected, tuple) and expected[0] == "not":
        assert a.value != expected[1], f"answered the whole-plan count {a.value} ({a.via})"
        return
    if expected == "any answer":
        return
    if isinstance(expected, tuple):
        assert expected[1].casefold() in f"{a.value} {a.text}".casefold(), f"got {a.value!r}: {a.text}"
    else:
        assert web.matches(expected, a.value), f"got {a.value} via {a.via}, expected {expected}"


def sample(source, n=150):
    qs = [json.loads(line) for line in open(graph.DATA / "questions.jsonl")]
    random.seed(0)
    return random.sample([q for q in qs if f"/{source}/" in q["path"]], n)


@pytest.mark.parametrize("source", ["swiss_dwellings", "ifc_models", "cubicasa5k"])
def test_answers_are_drawn(source):
    """Every node an answer counts or sums is outlined on a plan the UI shows, so a human can check it. An answer over
    several storeys is outlined across their plans; the view tabs count the items on each."""
    bad = []
    for q in sample(source):
        a, g, _ = ask.answer(q["path"], q["question"])
        if not a or not a.ids:
            continue
        views = {v.id for v in web.shown_views(g)}
        drawn = {i for i in a.ids if any(an.shape and an.view in views for an in g.anchors.get(i, ()))}
        if len(drawn) < len(set(a.ids)):
            bad.append(f"{q['path'].split('/')[-1]}: {len(drawn)}/{len(set(a.ids))} drawn: {q['question'][:70]}")
    assert not bad, f"{len(bad)} answers not fully drawn:\n" + "\n".join(bad[:10])


def test_vector_pdf_plan():
    """A floor plan read from its vector PDF alone (pdfplan: text, paths and the legend, no DXF) answers the dataset
    questions about it, and each room, door and fixture it counts is outlined on the PDF page."""
    g = pdfplan.read(SWISS.split("/", 2)[2])
    wrong = []
    for q in (q for q in map(json.loads, open(graph.DATA / "questions.jsonl")) if q["path"] == SWISS):
        a = ask.match(g, q["question"])
        if a is None or not web.matches(q["answer"], a.value) or not all(g.anchors[i] for i in a.ids):
            wrong.append(f"{q['question'][:70]}: {a and a.value}, expected {q['answer']}")
    assert not wrong, "\n".join(wrong)


def test_zero_answers_show_where_it_looked():
    """An answer of 0 still shows the scope searched (the plan or storey), so 'none' can be verified too."""
    a, g, _ = ask.answer(CUBI, "How many saunas are on this floor plan?")
    assert a.value == 0
    assert a.ids, "a zero answer draws nothing on the plan"


# ---- the running web UI ----

BASE = "http://127.0.0.1:8300"


def get(path, data=None, headers=None):
    req = urllib.request.Request(BASE + path, data and urllib.parse.urlencode(data).encode(), headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read().decode(errors="replace")


def asked(path, data):
    """POST a question and follow the progress view until the answer arrives, as the browser does."""
    _, html = get(path, data, {"HX-Request": "true"})
    while m := re.search(r'hx-get="(/d/[^"]+/ask/\w+)"', html):
        time.sleep(0.4)
        _, html = get(m.group(1), headers={"HX-Request": "true"})
    return html


def drawings():
    con = graph.connect()
    random.seed(1)
    return [random.choice(con.execute(f"select id from drawing where {web.GROUP} = ?", (s,)).fetchall())[0]
            for s in web.SOURCES] + [graph.resolve(p)[0] for p in (SWISS, CUBI, DUPLEX, FZK, A801)]


@pytest.mark.parametrize("drawing", drawings())
def test_web_drawing_page(drawing):
    """Every drawing opens with a picture to check answers against."""
    status, html = get(f"/d/{urllib.parse.quote(drawing, safe='')}")
    assert status == 200
    assert 'id="plan"' in html, "page has no drawing to outline answers on"


def test_web_answer_is_drawn():
    html = asked(f"/d/{graph.resolve(SWISS)[0]}/ask", {"question": "How many bathtubs are drawn on this floor plan?"})
    assert "<path" in html and "Stoa's answer: 1" in html


def test_web_free_question_gets_an_answer():
    """A question outside the templates is answered (Whg. B has 2 bedrooms, Whg. A has 1)."""
    html = asked(f"/d/{graph.resolve(SWISS)[0]}/ask", {"question": "Which apartment has more bedrooms?"})
    assert "No template covers" not in html
    assert "Whg. B" in html or "apartment B" in html, "answer does not name apartment B"


def test_web_follow_up_refers_to_earlier_answer():
    """A follow-up's "its" means the apartment named in the earlier answer (Whg. A's living room 3.10 is 22.09 m²)."""
    earlier = [{"q": "Which apartment has more bedrooms?", "value": "Whg. A", "text": "Whg. A has 5 bedrooms, Whg. B has 4.",
                "ids": []}]
    html = asked("/d/site1002_bldg2084_floor9393/ask", {"question": "How big is its living room?",
                                                        "history": json.dumps(earlier)})
    assert "22.09" in html, "follow-up did not answer for Whg. A's living room"
    assert html.count('class="q"') == 2 and 'name="history"' in html
