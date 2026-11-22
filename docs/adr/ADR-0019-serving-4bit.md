# ADR-0019 — Serving fp16: generator di GPU, embedder di CPU

**Status:** diterima
**Tanggal:** 2026-11-22

## Konteks

UI Meiva perlu jawaban dan sitasi nyata. Anggaran VRAM 6 GB tidak cukup untuk
bge-m3, reranker, dan Qwen fp16 sekaligus: percobaan pertama OOM di 5,27 GB.

Percobaan 4-bit (NF4) berhasil secara memori, tetapi menghasilkan kesalahan
entitas: untuk "What is the evolution line of charmander" model menjawab
"starts with Bulbasaur". Retrieval benar (top-5 hanya Charmander/Charizard),
dan B4 fp16 pada evidence yang sama menjawab benar. Jadi kuantisasi 4-bit
menurunkan kesetiaan entitas, dan verifier yang memakai model sama menerimanya.

## Keputusan

1. **Generator memakai Qwen fp16**, sama dengan eksperimen. 4-bit tetap tersedia
   lewat `--load-in-4bit`, tetapi bukan default.
2. **bge-m3 untuk query berjalan di CPU.** Hanya satu query per permintaan,
   sehingga biayanya kecil. Model dimuat fp32 di CPU agar kompatibel; `dtype:
   float16` tetap di config supaya `identity_hash` dan `verify_manifest` cocok
   dengan index yang dibangun.
3. **Reranker tetap di GPU.** Ini kebalikan dari percobaan pertama; reranker CPU
   membuat chat 17,5 detik karena ia menilai 20 pasangan per query.
4. **Warmup saat startup**: embedder CPU, reranker, dan generator dimuat sebelum
   server menerima permintaan.
5. **Citation fallback tetap ada, tetapi bukan bukti grounding.** Jika model
   tidak menyitir, API memasang chunk teratas. Sumber itu dokumen yang diambil,
   bukan klaim yang diverifikasi.

## Bukti

| Konfigurasi | Charmander | VRAM | Latensi chat |
| --- | --- | --- | --- |
| 4-bit, embedder GPU, reranker GPU | salah (Bulbasaur) | 3,43 GB | ~4 s |
| fp16, embedder GPU, reranker CPU | benar | ~4,4 GB | 17,5 s |
| fp16, embedder CPU, reranker GPU | benar | 4,41 GB | 3,8 s |

## Konsekuensi

- Demo dan eksperimen sekarang sama-sama fp16, sehingga angka laporan dan demo
  merujuk konfigurasi yang sama. Ini memperbaiki pelanggaran di versi 4-bit.
- Query encoding di CPU menambah sekitar 0,5 sampai 1 detik per pertanyaan.
  Retrieval tetap memakai index yang sama; tidak ada vektor yang dibangun ulang.
- Reranker GPU memakai sekitar 1,1 GB, Qwen fp16 sekitar 3,1 GB. Total 4,41 GB,
  menyisakan ruang untuk KV cache.
- Qdrant local mode tetap satu proses; API tidak boleh berjalan bersamaan dengan
  eval yang membuka database yang sama.

## Alternatif yang ditolak

- **4-bit untuk demo.** Cepat dan hemat, tetapi terbukti menukar entitas.
- **Reranker CPU.** Benar, tetapi 17,5 detik per chat.
- **8-bit.** Belum diuji; kandidat kalau kualitas fp16 terlalu mahal di mesin
  lain.
- **Dua proses terpisah.** Lebih bersih untuk produksi, tetapi menambah
  kompleksitas demo.
