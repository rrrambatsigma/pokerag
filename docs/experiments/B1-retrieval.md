# B1 — Retrieval baseline

- queries: 30
- qrels: `data/benchmark/qrels/dev.jsonl`
- strategy: `heading`
- index_version: `fa5a8dafad1e`
- embedder: `BAAI/bge-m3`

Relevance dinilai di level dokumen; `span_*` menuntut chunk juga memuat
kutipan `evidence`. Metrik dihitung pada cut-off yang sama untuk semua mode.

| Mode | recall@5 | recall@10 | mrr@5 | mrr@10 | ndcg@5 | ndcg@10 | span_hit@5 | span_hit@10 | span_mrr@5 | span_mrr@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `dense` | 1.000 | 1.000 | 0.907 | 0.907 | 0.930 | 0.930 | 0.933 | 1.000 | 0.794 | 0.803 |
| `bm25` | 0.967 | 1.000 | 0.758 | 0.763 | 0.810 | 0.821 | 0.633 | 0.867 | 0.427 | 0.456 |
| `hybrid` | 1.000 | 1.000 | 0.893 | 0.893 | 0.919 | 0.919 | 0.933 | 1.000 | 0.733 | 0.744 |
