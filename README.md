# Wren

Ask a question about a construction drawing and get the answer drawn on the drawing: every counted door, measured room or schedule row is outlined and numbered, so a person can check the answer against the sheet in seconds. Built for Stoa's "Make Blueprints Talk" challenge at Junction 2026.

Live at https://wren.jc.iffe.eu. Drawings can be uploaded as PDF, DXF, IFC or a picture.

## How a question is answered

Each drawing is read once, when it is ingested or uploaded, into a **drawing graph**: storeys, apartments, rooms, doors, windows, fixtures, schedule tables and rows, with their measures and an outline anchored in every view of the drawing (PDF page, picture, DXF, IFC plan). Questions are then answered from the graph, cheapest step first:

| step | what it does | model calls | cost per question |
|---|---|---|---|
| 1. Template | The question has a known form ("How many doors are drawn on this floor plan?"); a fixed query over the graph answers it | none | $0 |
| 2. Router | A reworded question ("how many bathrooms are there") is rewritten into a known form by `deepseek-4-1-flash-1017`. A second call checks that both ask for the same number; the room numbers, apartment letters and labels in the question must survive the rewrite, and every label must exist on the drawing | 2 × DeepSeek, ~900 tokens, ~0.7 s | ~$0.0001 |
| 3. Agent | Everything else: `deepseek-4-1-flash-1017` with graph tools (find, tally, details, total, plan geometry, PDF text search, reading the picture). `kimi-k3-1-pro` answers afresh when DeepSeek errs, is rate-limited, gives no value, or runs out of its 8 steps or 15 s, and reads pictures for the `look` tool. Kimi answers with what it has at 35 s; a call still running at 50 s is cut off | 2–4 calls | $0.0015 on average for typed questions, $0.025 for open IFC questions |

All models run on Verda's inference API, which allows 60 requests per minute for DeepSeek and 40 for Kimi per account. Prompt caching works on both models (72–86% of DeepSeek's prompt tokens are cached in the evaluations).

## Using the app

- **Drawing list:** search by name (every typed word must match; spaces, underscores and hyphens count as the same), tabs per source with counts, your uploads first and newest first, thumbnails, infinite scroll; the header slides away while scrolling down and returns on any scroll up.
- **Upload:** PDF, DXF, IFC or any picture format. A PDF gets its schedule tables and its floor-plan content (rooms, apartments, doors, fixtures and windows from the sheet's legend). A DXF with Swiss-style semantic layers is read exactly; any other DXF is drawn to a PDF sheet and read like a PDF. An IFC model gets its full semantics and generated per-storey plans. A picture gets the symbol and room detector (a single tiled yolo26m on CPU, 12–24 s). Upload errors are shown next to the button.
- **Drawing page:** the drawing with zoom (wheel), pan (drag), Fit, and tabs per view or storey that show how many of the answer's items each holds. "In this drawing" lists what was extracted, by category and schedule table; clicking one outlines all of them.
- **Asking:** answers ready within 1.5 s appear at once; slower ones show the agent's steps live ("Looking up…", "Reading the picture…") until the answer replaces them. After an answer the next question is a follow-up; "New question" starts over.
- **Dataset questions:** a folded list of Stoa's questions for the drawing; asking one shows Stoa's reference answer with ✓ or ✗, which follows any recount.

## Checking an answer

- Every item the answer counted or summed is outlined on the drawing and numbered (numbers stay 12 px at any zoom); clicking its row zooms to it, hovering highlights it, and the list folds above 10 items. The row says where it is ("Korridor 3.08 ↔ Reduit 3.06", "in Bad 3.05", "Whg. B").
- Unchecking a row leaves that item out: the count or area sum is recomputed and its outline dashed.
- Detections below 70% confidence are drawn in orange, and the count states a range ("8 windows, 1 of them uncertain (orange), so 7–8").
- Rooms whose printed area differs from their outline by more than 2% are marked.
- A reworded question shows how it was read ("Read as …"); an agent answer lists its lookups, each highlighting the items it found, and its model calls, tokens and cost.
- A count of zero still outlines the storeys it searched.
- Outlines from the vision model's reading of the picture (`look`) are drawn in orange and marked approximate.
- Follow-up questions carry the conversation, so "how big is its living room?" refers to the apartment named before.

## Accuracy

The templates are built from the question forms in Stoa's dataset, so the dataset score shows the readers' accuracy on those forms. The reworded-question score is the one that applies to new wording.

| questions | result | answered by |
|---|---|---|
| Swiss dwellings, dataset questions | 37,400 / 37,584 = 99.5% | DXF reader + templates |
| Swiss dwellings, read from the PDF alone (DXF ignored) | 37,074 / 37,584 = 98.6% | vector PDF reader + templates |
| Swiss dwellings, DXF read without its layer names (100 drawings) | 2,790 / 3,610 = 77.3% (rooms, apartments, areas, doors; no fixtures or windows without a legend) | DXF → PDF sheet + vector PDF reader |
| IFC models, dataset questions | 3,062 / 3,062 = 100% | IFC quantities + templates |
| CubiCasa plans, 957 held-out test plans | 14,930 / 17,322 = 86.2% of English and Finnish questions (windows 54.5%, doors 68.8%), from a 4-model symbol ensemble and a 2-model room ensemble, none of whose models trained on these plans; every other plan is answered by 5-fold cross-trained models that never saw it (85.6% on the train and val plans). | YOLO detectors + templates |
| Opening Design schedule sheets | 63 / 64 = 98.4% (measured with Kimi as the only agent, before tiering) | schedule table reader + agent |
| 120 dataset questions reworded as typed queries (`app/paraphrases.jsonl`) | 116 / 120 = 96.7%, median 2.2 s | router + tiered agent |
| Open IFC questions (ifc_bench, 100-question sample, judged by DeepSeek) | 66 / 100 = 66%, median 15.8 s, $0.025 per question (Kimi alone: 54%, median 41 s, $0.078) | tiered agent |

CubiCasa windows and doors are limited by the annotation convention: annotators split some windows at the mullion and leave others whole.

## Cost per drawing (ingestion, paid once)

| drawing | how it is read | time |
|---|---|---|
| Swiss DXF or vector PDF | `swiss.py` / `pdfplan.py`, CPU | under 1 s per page |
| IFC model | `ifc.py` + `ifcplans.py` (per-storey plans), CPU; very large models on a rented Verda CPU instance | seconds to minutes |
| Picture (CubiCasa style) | dataset plans: 4-model symbol ensemble + 2-model room ensemble (out-of-fold models for the train plans), predicted in batch on Verda GPUs; uploads: a single tiled yolo26m (82.7% exact counts against the ensemble's 86.2% on the test plans) + room detector on CPU, pictures scaled to at most 2000 px | 12–24 s per upload |
| Schedule sheet | `sheets.py` ruled-table reader, CPU | ~1.5 s per page |

No model call happens at ingestion; the readers use geometry, text and the trained detector only.

## What stays useful as models improve

General-purpose models still miscount symbols on technical drawings and cannot show where an answer came from. The parts of this system that do not depend on which model sits on top:

- **The drawing graph with source anchors.** Every node keeps its outline in every view of the drawing (DXF entities fitted to the PDF render by a similarity transform, IFC GlobalIds to the generated plan SVGs). Any model, now or next year, that answers through the graph produces an answer a person can check on the sheet.
- **Readers that use the drawing's own structure.** Vector PDFs and DXFs carry exact text, arcs and polygons: room labels, printed areas, door swings, legend symbols. Reading them costs nothing per question and is exact where the drawing is.
- **A checked router.** A cheap model turns free wording into a deterministic query, a second call checks the rewrite asks the same number, and code checks that the question's scope survived. The answer then costs a graph lookup.
- **Held-out detector predictions.** Every CubiCasa plan is predicted by models that never trained on it (five folds), so the measured accuracy is the accuracy on unseen plans.
- **Known coverage.** Each drawing's graph records which extractors ran and which categories and attributes it holds, so the agent can say what the drawing lacks instead of guessing.

## Limits

- Fixtures and windows in a vector PDF or a non-Swiss DXF are found only when the sheet has a legend; rotated fixtures and sliding doors are missed.
- Floor plans embedded in a PDF as images are read only for their text.
- The CubiCasa detector was trained on CubiCasa plans; on other drawing styles its counts are approximate.
- The agent can answer open breakdown questions confidently and wrongly; nothing yet detects that.

## Running it

```sh
cd app
uv run uvicorn web:app --port 8300          # web UI
uv run ingest.py <source>                    # build graphs for a dataset source into ~/junction-data/graph.db
uv run score.py [source | file.jsonl] [n] [--llm]   # accuracy on the dataset or reworded questions
uv run score_pdfplan.py                      # the vector PDF reader on the Swiss PDFs
uv run pytest -q test_questions.py           # paraphrased, scoped, open questions and the web UI
```

