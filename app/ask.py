"""Answer a question about a drawing: template matcher over the graph's query functions, the model for the rest."""
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import agent
import graph
from graph import Answer


def has(pattern, q):
    return re.search(pattern, q, re.I)


def quoted(q):
    for p in (r"«(.+?)»", r"„(.+?)“", r"'(.+)'", r'"(.+?)"'):
        if m := re.search(p, q):
            return m.group(1)


AREA = r"\b(area|fläche|nettofläche|nettowohnfläche|nettogeschossfläche|surface|superficie)\b"
LABELLED = r"labelled|Bezeichnung|désignation|designées|désignées|denominazione|denominati"
NUMBERED = r"numbered|nummerierten|numérotées|numerati"
ROOM_NO = r"\b(?:room|Raum|pièce|locale) ([A-Z]?\d+(?:\.\d+)*) \("
APARTMENT = r"\b(?:apartment|Wohnung|appartement|appartamento|l'appartement|dell'appartamento) ([A-Z]{1,2})\b"
APARTMENTS = r"\b(apartments|Wohnungen|appartements|logements|appartamenti)\b"
COUNTS = [  # (categories summed, words that ask for them); combined and room forms before single fixtures
    # before "wc": the bathroom question says "not counting separate WC rooms"
    (("bathroom",), r"\bbathrooms\b|kylpy- tai pesuhuone"),
    (("wc",), r"separate WC rooms|erillistä WC-tilaa"),
    (("kitchen",), r"\bkitchens\b|keittiötä"),
    (("bedroom",), r"\bbedrooms\b|makuuhuonetta"),
    (("balcony",), r"\bbalconies\b|parveketta"),
    (("sauna",), r"\bsaunas\b|saunaa"),
    (("storey",), r"\bfloors \(storeys\)|kerrosta"),
    (("sink", "washbasin"), r"sinks and wash basins|pesu- ja tiskiallasta"),
    (("shower", "bathtub"), r"showers and bathtubs|suihkua ja kylpyammetta"),
    (("door",), r"\b(doors|Türen|portes|porte|ovea)\b"),
    (("window",), r"\b(windows|Fenster|fenêtres|finestre|ikkunaa)\b"),
    (("toilet",), r"\b(toilets|toilet bowls|WCs?)\b|WC-istuinta"),
    (("washbasin",), r"\b(washbasins|Lavabos|lavabi)\b"),
    (("bathtub",), r"\b(bathtubs|Badewannen|baignoires|vasche da bagno)\b"),
    (("shower",), r"\b(showers|Duschen|douches|docce)\b"),
]
EXCLUDED = {"shaft", "void"}


# The IFC questions quote room/model/storey labels; apostrophes inside names are literal.
IFC_QUOTES = r"'(.+?)'(?=\s+(?:on|på|op|i\b)|[,?]|$)|„(.*?)[“”]|«(.*?)»|\"(.*?)\""


def guillemets(q):
    """Each label of the question written in «», the quoting the router uses."""
    return re.sub(IFC_QUOTES, lambda m: "«" + next(v for v in m.groups() if v) + "»", q)


def masked(q):
    """The question with its labels, numbers and apartment letters replaced by placeholders."""
    q = re.sub(r"«.*?»|„.*?[“”]|\".*?\"|'.*'", "‹label›", q)
    q = re.sub(APARTMENT, lambda m: m.group(0)[:-len(m.group(1))] + "‹unit›", q, flags=re.I)
    return " ".join(re.sub(r"\d+(?:[.,]\d+)*", "‹n›", q).split())


def form(q):
    """The question's template form."""
    return masked(q).casefold()


# Templates answer the question forms of the dataset sources they were written for. Other wordings are rewritten
# into one of these forms by a fast model (`route`); other scopes and kinds of question go to the agent.
QUESTIONS = [q for q in map(json.loads, open(graph.DATA / "questions.jsonl"))
             if q["path"].split("/")[1] in ("swiss_dwellings", "cubicasa5k", "ifc_models")]
FORMS = {form(q["question"]) for q in QUESTIONS} | {form(guillemets(q["question"])) for q in QUESTIONS}
ENGLISH = {}  # source → the English question forms, written with placeholders, most asked first
for q in QUESTIONS:
    if re.match(r"(How|What) ", q["question"]):
        ENGLISH.setdefault(q["path"].split("/")[1], Counter())[masked(guillemets(q["question"]))] += 1
ENGLISH = {s: [f for f, _ in c.most_common()] for s, c in ENGLISH.items()}
ROUTER = "deepseek-4-1-flash-1017"
ROUTE = """You map a question about one construction drawing to a fixed query, so that a program can answer it exactly.

Queries (‹label› is a name written on the drawing, ‹n› a room number, ‹unit› an apartment letter):
{forms}

Names on this drawing: {names}

Question: {question}

If one query asks for exactly the same number, reply with it, every placeholder filled in and each ‹label› written \
in «» exactly as on the drawing: {{"query": "..."}}. Wording and language may differ; a question that names no \
apartment, room or storey is about the whole drawing. Reply {{"query": null}} if the question asks for anything \
else, for example a scope the query does not have (rooms of one apartment when the query counts the whole floor), \
a comparison, a yes/no, which/where/why, or a condition the query lacks."""

SAME = """Two questions about the same construction drawing:
A: {question}
B: {query}

Write what number each asks for, naming the quantity, the things counted or measured, and the part of the drawing \
(a question that names no apartment, room or storey means the whole drawing). A yes/no, comparison or which/where \
question asks for no number. Then say if the two numbers are the same.
Reply with JSON: {{"a": "...", "b": "...", "same": true or false}}"""


PLURAL = {"balcony": "balconies", "storey": "storeys", "wc": "WC rooms", "sanitary": "sanitary fixtures",
          "light": "light fixtures", "sauna_bench": "sauna benches"}


def noun(what, n):
    """A category name as counted: "1 door", "8 windows", "2 sinks + washbasins"."""
    if n == 1 or " " in what.replace(" + ", ""):  # phrases like "rooms labelled 'Bad'" are already plural
        return what
    return " + ".join(PLURAL.get(w, w if w.endswith("s") else w + "s") for w in what.split(" + "))


def count(g, nodes, what, via):
    """A count; when nothing is found the answer points at the storeys searched, so 'none' can be checked too.
    Uncertain detections (drawn in orange) are counted and turn the answer into a range."""
    ids = [n.id for n in nodes] or [n.id for n in g.find(kind="storey")]
    n, low = len(nodes), sum(x.conf < 0.7 for x in nodes)
    text = f"{n} {noun(what, n)}"
    if low:
        text += f", {low} of them uncertain (orange), so {n - low}–{n}"
    return Answer(n, "count", ids, text, via)


def area(g, nodes, what, via):
    nodes = [n for n in nodes if g.area(n) is not None]
    parts = " + ".join(f"{g.area(n):g}" for n in nodes)
    return Answer(g.measure(nodes), "m²", [n.id for n in nodes], f"{what}: {parts}", via)


def indoor(g, **filters):
    return [n for n in g.find(kind="space", **filters)
            if n.number and not n.attrs.get("outdoor") and n.cat not in EXCLUDED]


def match(g, q):
    """Known question forms → Answer, or None when no template fits."""
    if form(q) not in FORMS or not countable(g):
        return None  # templates count and measure rooms and elements; schedule sheets have neither
    if g.source in ("ifc_models", "ifc_bench"):
        return match_ifc(g, q)
    label = quoted(q)
    if label and has(LABELLED, q):
        rooms = g.find(kind="space", name=label)
        if has(AREA, q):
            return area(g, rooms, f"rooms labelled {label!r}", "label-area")
        return count(g, rooms, f"rooms labelled {label!r}", "label-count")
    if has(AREA, q):
        if m := has(ROOM_NO, q):
            return area(g, g.find(kind="space", number=m.group(1)), f"room {m.group(1)}", "room-area")
        if m := has(APARTMENT, q):
            unit = g.find(kind="unit", number=m.group(1))
            rooms = indoor(g, within=unit[0].id) if unit else []
            return area(g, rooms, f"indoor rooms of apartment {m.group(1)}", "apartment-area")
        if has(NUMBERED, q):
            return area(g, indoor(g), "numbered indoor rooms", "indoor-area")
    if has(APARTMENTS, q):
        return count(g, g.find(kind="unit", cat="apartment"), "apartments", "apartments")
    for cats, words in COUNTS:
        if has(words, q):
            return count(g, [n for c in cats for n in g.find(cat=c)], " + ".join(cats), f"count-{'+'.join(cats)}")


IFC_ELEMENTS = [
    ("door", r"doors|Türen|ust|døre|deuren|porte|puertas|drzwi"),
    ("window", r"windows|Fenster|akent|vinduer|ramen|finestre|ventanas|ferestre|okien"),
    ("wall", r"walls|Wände|seina|vægge|wanden|muri|muros|pereți|ścian"),
    ("column", r"columns|Stützen|posti|søjler|kolommen|pilares|stâlpi"),
    ("beam", r"beams|Träger|tala|bjælker|balken|vigas|grinzi|belek"),
    ("stair", r"stairs|Treppen|treppi|trapper|escaleras|scări|schodów"),
    ("sanitary", r"sanitary fixtures|Sanitärobjekte|sanitaire toestellen"),
    ("light", r"light fixtures|corpuri de iluminat"),
    ("slab", r"slabs|Decken|Bodenplatten|dæk|plader|vloeren"),
    ("space", r"rooms|spaces|Räume|ruumi|rum|ruimtes|locales|pomieszczeń"),
]


def match_ifc(g, q):
    labels = [next(v for v in m if v) for m in re.findall(IFC_QUOTES, q)]
    if not labels:
        return None
    # Ignore labels when identifying the operation: room names can contain category words.
    text = re.sub(IFC_QUOTES, "", q)
    storeys = g.find(kind="storey", name=labels[-1])
    scope = storeys[0].id if storeys else None
    model_count = has(r"\b(model|Modell|mudelis|modellen|modello|modelo|modelul|modelu)\b", text)
    if scope is None and not model_count:
        return None
    cat = next((c for c, words in IFC_ELEMENTS if has(r"\b(?:" + words + r")\b", text)), None)
    if has(r"m²|m2", text) and len(labels) == 2:
        label = labels[0].casefold()
        rooms = [n for n in g.find(kind="space", within=scope)
                 if label in {(n.name or "").casefold(), (n.attrs.get("ifc_name") or "").casefold(),
                              (n.attrs.get("long_name") or "").casefold()}]
        return area(g, rooms, labels[0], "ifc-room-area")
    if cat is None and has(r"m²|m2", text):
        cat = "space"
    if cat is None:
        return None
    nodes = g.find(kind="space", within=scope) if cat == "space" else [
        n for n in g.find(cat=cat, within=scope) if n.kind in ("opening", "element")]
    if has(r"m³|m3", text):
        attr, units, via = "volume_m3", "m³", "ifc-volume-" + cat
    elif has(r"length|længde|lengte|Wie lang", text):
        attr, units, via = "length_m", "m", "ifc-length-" + cat
    elif has(r"m²|m2", text):
        if cat == "space":
            return area(g, nodes, labels[-1], "ifc-storey-area")
        attr, units, via = "side_area_m2", "m²", "ifc-side-area-" + cat
    else:
        return count(g, nodes, cat, "ifc-count-" + cat)
    nodes = [n for n in nodes if attr in n.attrs]
    return Answer(round(sum(n.attrs[attr] for n in nodes), 3), units, [n.id for n in nodes],
                  f"Sum of {attr} on {labels[-1]}", via)


def countable(g):
    return any(n.kind in ("space", "opening", "element") for n in g.nodes.values())


YES_NO = r"^\s*(is|are|does|do|did|has|have|can|could|was|were|will|would)\b"
ROUTE_SECONDS = 5  # after this the agent answers instead
routing = ThreadPoolExecutor(4)


def route(g, question):
    """route_once within ROUTE_SECONDS, or None: a stalled or failed router call hands over to the agent."""
    if has(YES_NO, question):
        return None  # the count templates answer "how many", not "is there"
    try:
        return routing.submit(route_once, g, question).result(timeout=ROUTE_SECONDS)
    except Exception:
        return None


def scope(q):
    """The room numbers, apartment letters and labels a question names: its scope, which a rewrite must keep."""
    return ([*re.findall(r"\d+(?:[.,]\d+)*", q), *(m.group(1) for m in re.finditer(APARTMENT, q, re.I))]
            + ([quoted(q)] if quoted(q) else []))


def route_once(g, question):
    """The question rewritten by the fast model as one of the template forms, answered by that template; or None."""
    source = g.source if g.source in ENGLISH else "ifc_models" if g.source == "ifc_bench" or any(
        v.file.endswith(".ifc") for v in g.views.values()) else "cubicasa5k"
    names = {"storeys": [n.name for n in g.find(kind="storey") if n.name],
             "apartments": [n.number for n in g.find(kind="unit") if n.number],
             "rooms (number and label)": sorted({f"{n.number or ''} {n.name}".strip() for n in g.find(kind="space")
                                                 if n.name})[:150],
             "model": g.title}
    r = agent.chat(ROUTER, [{"role": "user", "content": ROUTE.format(
        forms="\n".join(ENGLISH[source]), names=json.dumps(names, ensure_ascii=False), question=question)}])
    query = json.loads(re.search(r"\{.*\}", r["choices"][0]["message"]["content"], re.S).group()).get("query")
    for u in names["apartments"]:  # the model sometimes quotes apartment letters like labels
        query = query and query.replace(f"«{u}»", u)
    if query and (m := re.search(r"room (\S+) \(‹label›\)", query)):  # the label of a room the number names
        rooms = g.find(kind="space", number=m.group(1))
        query = rooms and rooms[0].name and query.replace("(‹label›)", f"(«{rooms[0].name}»)")
    if query and any(not re.search(rf"(?<![\w.]){re.escape(t)}(?![\w]|\.\d)", query, re.I) for t in scope(question)):
        return None  # the rewrite dropped part of the question's scope ("in apartment A", "kitchen 1.05")
    known = {(v or "").casefold() for n in g.nodes.values() for v in (n.name, n.attrs.get("ifc_name"), n.attrs.get("long_name"))}
    known.add((g.title or "").casefold())
    labels = re.findall(r"«(.*?)»", query or "")
    if not query or "‹" in query or form(query) not in FORMS or any(l.casefold() not in known for l in labels):
        return None  # labels must be on the drawing: a guessed one would answer a confident 0
    # the model picks less carefully than it compares: a second call checks the pick asks for the same number
    check = agent.chat(ROUTER, [{"role": "user", "content": SAME.format(question=question, query=query)}])
    if not json.loads(re.search(r"\{.*\}", check["choices"][0]["message"]["content"], re.S).group()).get("same"):
        return None
    a = match(g, query)
    if a:
        a.read_as = query
        usage = [r["usage"], check["usage"]]
        a.usage = (f"{ROUTER}, 2 calls, {sum(u['total_tokens'] for u in usage)} tokens, "
                   f"${sum(agent.cost(ROUTER, u) for u in usage):.4f}")
    return a


def reply(g, question, history=(), progress=None):
    """A template's answer when one fits the question (as asked, or as rewritten by the router), else the agent's.
    Follow-ups (with history) go to the agent, which knows what "those" refers to."""
    if history:
        return agent.answer(g, question, history, progress)
    return (match(g, question) or (countable(g) and route(g, question))
            or agent.answer(g, question, progress=progress))


def answer(path, question, llm=True):
    """(Answer or None, graph, view id) for a question about the file at `path`; templates only unless llm."""
    hit = graph.resolve(path)
    if not hit:
        return None, None, None
    g = graph.load(hit[0])
    return reply(g, question) if llm else match(g, question) or route(g, question), g, hit[1]
