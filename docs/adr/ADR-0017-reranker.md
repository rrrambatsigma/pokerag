# ADR-0017 — Reranker cross-encoder untuk B3 dan A2

**Status:** diterima
**Tanggal:** 2026-11-22

## Konteks

Pilot B2 menunjukkan kegagalan retrieval yang konkret. Pada q0005 ("Are Slowbro
and Golduck both in the Water 1 egg group?"), chunk egg group Slowbro berada di
rank 11, sedangkan B2 hanya mengambil top-5. Model hanya melihat Golduck, lalu
menjawab "No" padahal jawabannya "Yes".

A2 juga menuntut tingkat keempat, hybrid + reranking, dan B3 mendefinisikan
reranker sebagai bagian dari paket retrieval.

## Keputusan

1. **Cross-encoder `BAAI/bge-reranker-v2-m3`**, revisi
   `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, fp16.
2. **Dua puluh kandidat, lalu potong ke `top_k`.** Reranker tidak boleh menilai
   seluruh korpus; 20 cukup untuk menaikkan chunk yang relevan tanpa membuat
   latensi meledak.
3. **`max_length` 1024 dan batch 4.** Chunk fixed maksimum 1.005 token; memotong
   di 512 akan membuang bukti di bagian akhir chunk.
4. **Reranker hidup hanya di tahap retrieval.** Setelah semua query direrank,
   reranker dan embedder dilepas sebelum generator dimuat.
5. **Override `--mode` dan `--rerank/--no-rerank`** supaya A2 bisa dijalankan
   pada satu varian tanpa mengedit config.

## Konsekuensi

- VRAM tahap retrieval naik ke sekitar 2,27 GB (bge-m3 + reranker), masih aman
  di 6 GB karena generator belum dimuat.
- q0005 membaik secara retrieval: sebelum rerank, chunk egg group Slowbro ada di
  rank 11; sesudahnya, Golduck dan Slowbro masing-masing di rank 1 dan 2, dan
  keduanya memuat baris egg group.
- Namun B3 masih menjawab "No" pada q0005 meskipun kedua bukti sudah ada. Jadi
  reranker menyelesaikan masalah retrieval, bukan masalah penalaran. Perbaikan
  berikutnya ada di B4, bukan di retrieval.
- Reranker menambah latensi retrieval. Angka p50 B3 lima query sekitar 1,7 detik
  termasuk retrieval, rerank, dan generation.

## Alternatif yang ditolak

- **Reranker yang lebih kecil (`jina-reranker-v2-base`).** Belum dicoba; kalau
  VRAM atau latensi menjadi masalah, ini kandidat pertama.
- **Memperbesar top-k tanpa reranker.** Lebih murah, tetapi menambah konteks dan
  biaya token tanpa menjamin urutan yang benar.
- **Rerank dengan LLM generator.** Dua peran bercampur dan biayanya jauh lebih
  tinggi untuk 20 kandidat per query.
