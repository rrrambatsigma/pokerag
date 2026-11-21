# ADR-0009 — Hybrid retrieval di Python: dua index, satu ranking

**Status:** diterima
**Tanggal:** 2026-11-21

## Konteks

Tahap index sudah menghasilkan dua jenis index yang **terpisah secara sengaja**
(ADR-0008): vektor dense di Qdrant lokal (`chunks_fixed`, `chunks_heading`,
`chunks_hierarchical`) dan BM25 per strategi di `artifacts/indexes/bm25/`
(format `bm25s`). Yang belum ada adalah lapisan yang menjawab satu pertanyaan
dengan menggabungkan keduanya.

Tiga hal menuntut keputusan:

1. **Skala skor tidak sebanding.** Cosine ada di `[0, 1]`, BM25 tidak dibatasi.
   Menjumlahkan skor mentah membuat salah satu sumber menang hanya karena
   skalanya lebih besar.
2. **BM25 bisa dipindah ke Qdrant sebagai sparse vector**, dan Qdrant 1.20
   mendukung `prefetch` + fusion di server. Apakah itu jalan yang lebih baik?
3. **Evaluasi harus jujur.** Klaim "hybrid lebih baik" hanya boleh muncul kalau
   ada qrels dan angka, bukan karena hybrid terdengar lebih canggih.

## Keputusan

1. **Fusion di Python dengan reciprocal-rank fusion (RRF).** Setiap retriever
   menominasikan `top_dense` / `top_bm25` kandidat, lalu:

   ```
   skor(chunk) = Σ weight_sumber / (rrf_k + rank_sumber),  rank mulai 1
   ```

   Yang dipakai adalah **peringkat**, bukan skor. Ini menghilangkan masalah
   perbedaan skala tanpa normalisasi yang harus disetel per korpus.
2. **BM25 tetap di `bm25s`, tidak dipindah ke Qdrant.** Pindah ke sparse vector
   akan mengubah tokenizer/skor (harus menulis ulang bobot BM25 atau memakai
   FastEmbed dengan stemmer berbeda), menghapus pemisahan bersih untuk A2, dan
   tidak memberi keuntungan terukur pada 24 ribu chunk. Opsi itu dicatat sebagai
   jalan yang tersedia kalau nanti pindah ke server dengan banyak request.
3. **Relevansi dinilai di level dokumen**, dengan metrik tambahan `span_*` yang
   menuntut chunk memuat kutipan `evidence`. Level chunk dipakai hanya untuk
   `span_*`, karena batas chunk berbeda antar strategi (A1) sehingga judgement
   chunk akan membuat strategi tidak sebanding.
4. **Guard sebelum query, bukan setelah hasil aneh.** Manifest harus cocok
   dengan identitas embedder dan hash chunk; jumlah titik Qdrant harus sama
   dengan baris parquet; urutan `bm25/chunk_ids.json` harus identik dengan
   urutan `chunk_id` di parquet (BM25 memetakan hasil lewat nomor baris).
   Semua pelanggaran melempar error, bukan menghasilkan ranking yang terdengar
   masuk akal.
5. **Tidak ada yang dibangun ulang.** Retrieval tidak menulis index. Qdrant dan
   BM25 dibaca apa adanya; mengganti `rrf_k`, bobot, atau `top_k` hanya
   mengubah config.

## Hasil baseline (B1)

30 query dev, satu dokumen relevan per query, strategi `heading`, embedder
`BAAI/bge-m3` (index `a89abec5b5e5`). Angka lengkap: `docs/experiments/B1-retrieval.md`.

| Mode | recall@5 | MRR@5 | nDCG@5 | span_hit@5 |
|---|---|---|---|---|
| dense | 1,000 | 0,907 | 0,930 | 0,933 |
| BM25 | 0,967 | 0,741 | 0,797 | 0,667 |
| hybrid (`rrf_k=10`, bobot dense 2 : BM25 1) | 1,000 | 0,893 | 0,919 | 0,933 |

Pembacaan yang benar: **hybrid menyamakan recall dense dan mengalahkan BM25,
tetapi belum mengalahkan dense pada MRR/nDCG.** Dengan satu dokumen relevan per
query dan dense yang sudah kuat, fusion terutama menambah noise peringkat.
`rrf_k=60` dengan bobot 1:1 bahkan menurunkan recall@5 menjadi 0,967; karena itu
default memakai `rrf_k=10` dan bobot 2:1, dipilih di dev dan dicatat sebagai
baseline, bukan sebagai klaim kemenangan.

## Konsekuensi

* Satu query hybrid membuka Qdrant + file BM25. Di skala ini biayanya puluhan
  milidetik; embedder tetap komponen termahal (±10 detik sekali muat).
* Filter metadata (mis. `doc_id`) belum didukung: Qdrant punya `query_filter`,
  BM25 harus disaring manual. Ditunda sampai ada kebutuhan nyata.
* Reranker cross-encoder belum dipakai; ia berbagi VRAM dengan generator dan
  lebih tepat menjadi eksperimen B4 setelah baseline ini tetap.
* qrels dev berisi 30 query/30 dokumen. Itu cukup untuk menemukan bug dan
  memilih default, **tidak** cukup untuk klaim signifikan. Set uji terpisah
  harus ditulis sebelum angka masuk laporan akhir.
* Kalau nanti hybrid tetap tidak mengalahkan dense, hasil itu dilaporkan apa
  adanya; alternatif yang masuk akal adalah menaikkan `top_dense`, menambah
  query multi-bukti, atau memakai BM25 hanya sebagai filter kandidat.
