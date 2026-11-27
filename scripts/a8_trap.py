"""A8, one-off: for test queries where BM25 has the source item in its top 10 but the hybrid does not,
where is the source in the hybrid top 50, and does the reranker bring it back?

Usage: .venv/bin/python scripts/a8_trap.py   (~3 min)
"""
import json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import rerank, hybrid
from bm25 import BM25
items = [json.loads(l) for l in open("data/items.jsonl")]
qs = [json.loads(l) for l in open("eval/queries.jsonl") if json.loads(l)["split"] == "test"]
rr = rerank.Rerank.from_items(items)
bm = rr.first.engines[0]
lost = in50 = back = 0
back_any = 0  # rerank brings source into top 10 on queries where hybrid lacked it
for q in qs:
    s = q["source_id"]
    h50 = rr.first.search(q["query"], 50)
    top = [d for d, _ in rr.order(q["query"], h50)][:10]
    if s not in h50[:10] and s in top:
        back_any += 1
    if s in bm.search(q["query"], 10) and s not in h50[:10]:
        lost += 1
        if s in h50:
            in50 += 1
            back += s in top
            print(q["qid"], "hybrid rank", h50.index(s) + 1, "reranked rank", [d for d, _ in rr.order(q["query"], h50)].index(s) + 1)
        else:
            print(q["qid"], "not in hybrid top 50")
print(f"lost {lost}, in hybrid top 50 {in50}, back in top 10 after rerank {back}; rerank brought source into top 10 on {back_any} queries overall")
