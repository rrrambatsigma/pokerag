# A1 — perbandingan strategi chunking (sisi index)

> Dibuat otomatis oleh `pokerag-index build`; angka berasal dari manifest
> yang hash-nya tercatat di tabel. Artefak besar ada di HF (v2), bukan di git.

- snapshot: `pokerag-corpus-v2` — `5eda7c8efc03f6fe`
- embedder: `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181`, 1024 dim, float16, jendela 1024 token
- database: satu Qdrant, 3 koleksi (chunks_fixed, chunks_heading, chunks_hierarchical)
- vektor: L2-normalised, cosine, BM25 terpisah per koleksi

## Hasil

| Strategi | Chunk | Titik | token p50 / p95 | max | ctx@k=5 | index size | build | cakupan teks | index_version |
|---|---|---|---|---|---|---|---|---|---|
| `fixed` | 11642 | 11642 | 598 / 784 | 1005 | 2990 | 125.2 MB | 223.5 s | 100.0% | 75f6330e7f01 |
| `heading` | 27902 | 27902 | 302 / 448 | 627 | 1510 | 286.7 MB | 329.9 s | 99.8% | fa5a8dafad1e |
| `hierarchical` | 28786 | 28786 | 241 / 365 | 512 | 1205 | 296.4 MB | 351.3 s | 99.9% | 6df736f91d34 |

## Definisi

| Metrik | Definisi |
|---|---|
| Chunk | unit retrieval yang di-embed (baris di `chunks/<strategi>.parquet` dengan `role != parent`) |
| Titik | vektor di koleksi Qdrant (satu per unit retrieval) |
| token p50/p95 | tokenizer model embedding, `add_special_tokens=false` |
| max | chunk terpanjang; harus ≤ jendela embedder |
| ctx@k=5 | 5 × token p50 — perkiraan konteks yang masuk generator |
| index size | byte di disk: direktori koleksi Qdrant + BM25 |
| build | waktu build nyata terakhir per koleksi, dari riwayat `provenance.json` (run cache tidak dihitung) |
| cakupan teks | proporsi dokumen yang tercakup minimal satu chunk (gabungan span) |
| index_version | hash snapshot, chunk, strategi, identitas embedder, metrik jarak, BM25 |

`hierarchical` juga menyimpan 15,803 baris parent untuk context expansion (tidak di-embed).

## Tertunda (belum diukur, bukan nol)

| Metrik | Kenapa belum ada | Cara mengukurnya nanti |
|---|---|---|
| passage recall@k | butuh qrels; belum dibuat | qrels menyimpan `doc_id` + rentang `char_start/char_end`; hit dihitung bila chunk top-k berasal dari dokumen yang sama dan tumpang-tindih dengan rentang gold (dua varian: ≥1 char, gold tertutup penuh) |
| groundedness | butuh generator (B0–B6) | jawaban dinilai terhadap chunk yang dikutip, dengan rubrik yang dicatat di laporan tahap generator |

Kedua metrik di atas **tidak** boleh dilaporkan sebagai 0 atau "tidak ada";
selama belum diukur, selnya tetap kosong dengan alasan ini.

## Reproduksi

```bash
pokerag-chunk build
pokerag-index build
pokerag-index stats
```
