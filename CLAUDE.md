# find-me-a-lipstick

Hybrid text + photo search over ~15k beauty products, built so Taras can defend every part
with his own numbers in a technical interview.

- `GLOSSARY.md`: the project's words. Use them in code, docs and commits.
- `docs/PLAN.md`: the plan, the agreed decisions and the small steps of each part. Read it when starting a step.
- `docs/journal.md`: what each finished step did, why, and each research verdict.
- `docs/adr/`: decisions that are hard to undo (created when the first one appears).
- `~/Documents/Projects/ask-bro/find-me-a-lipstick/`: private context (who this is for, dates,
  the assistant it imitates, sources of test queries). Read it when planning or writing test
  queries. This repo is public: never copy a name, date or detail from there into any file,
  commit message or code here.

## Who does what

Taras is learning computer vision search, hybrid search and reranking. Claude writes the code;
Taras owns the ideas and must be able to explain every piece.

Breadth over depth (20/80): teach only the ideas a Team Lead would ask about, the mechanisms
and trade-offs of search, ranking, evaluation and the assistant. Plumbing (downloads, parsing,
encodings) is just done, with no quiz.

Teach each part with the scheme in `docs/PLAN.md` ("How Taras learns each part"): a ~5-line toy
example, Taras guesses its result, a trap example, one question the Team Lead would ask. Explain
each piece through its mechanism: the idea, how the code works step by step, why it behaves so.
Before any new number in the ablation table, Taras guesses it first; record his guess next to
the measured value in the journal.

Never write the trap's answer in a committed file before Taras has guessed it.

A long session opens with its map: the blocks, the time for each and the total. After each block,
say how far along you are and what is left.

## Downloads

The sandbox blocks `m.media-amazon.com` (product images) and may block model hosts. When a
download is blocked, give Taras one command to paste into his terminal, and check its result
afterwards.

## Finishing a step

- `README.md` has a copy-paste command for every way to run the project.
- The step's check in `docs/PLAN.md` passes, and the journal has Taras's guess and the measured value.
