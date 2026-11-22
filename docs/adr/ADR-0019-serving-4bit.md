# ADR-0019 — Serving: B4 dengan Qwen 4-bit dan UI nyata

**Status:** diterima
**Tanggal:** 2026-11-22

## Konteks

UI Meiva masih memakai `answerFor` dari mock, dan API masih memakai
`StubGenerator`. Untuk demo, jawaban dan sitasi harus berasal dari sistem nyata.
Kendalanya VRAM: bge-m3 (1,1 GB), reranker (1,1 GB), dan Qwen fp16 (3,1 GB)
tidak muat bersama di 6 GB.

## Keputusan

1. **Serving memakai Qwen 4-bit NF4.** Model turun ke sekitar 1,15 GB, sehingga
   bge-m3 + reranker + generator tinggal di sekitar 3,4 GB.
2. **Varian serving adalah B4**, dimuat dari `configs/ladder.yaml`: planner,
   reranker, verifier, dan satu repair. API bisa dijalankan dengan
   `--no-generator` untuk kembali ke stub tanpa GPU.
3. **Warmup retrieval saat startup.** bge-m3 dimuat sebelum server menerima
   permintaan, supaya chat pertama tidak menunggu 30 detik.
4. **UI memanggil `/api/chat`.** Jika API gagal, store kembali ke mock dan
   menulis peringatan di konsol. Fallback ini untuk kenyamanan pengembangan,
   bukan untuk produksi.
5. **Proxy Vite bisa diarahkan lewat `VITE_API_TARGET`**, default tetap
   `http://127.0.0.1:8000`. Di mesin ini port 8000 dipakai layanan lain, jadi
   demo berjalan di 8010.
6. **Fallback sitasi di API.** Kalau model tidak menyebut satupun chunk tetapi
   jawabannya tidak abstain, API menyitir chunk teratas. Ini hanya untuk serving;
   metrik eksperimen tetap memakai sitasi model.

## Konsekuensi

- **Angka laporan dan demo memakai kuantisasi berbeda.** Eksperimen B0-B4
  memakai fp16; demo memakai 4-bit. Brief meminta keduanya merujuk benda yang
  sama. Pilihannya: jalankan ulang eksperimen dalam 4-bit, atau demo memakai
  fp16 dengan embedder di CPU atau proses terpisah. Ini harus diputuskan sebelum
  laporan final, bukan setelah.
- Chat pertama setelah startup sekarang cepat karena embedder sudah dimuat.
- Fallback mock bisa menyembunyikan API yang mati. Peringatan konsol adalah
  satu-satunya sinyal; jangan diabaikan saat demo.
- Qdrant local mode tetap satu proses. API tidak boleh berjalan bersamaan dengan
  eval yang membuka database yang sama.

## Alternatif yang ditolak

- **fp16 untuk demo.** OOM saat embedder ikut dimuat.
- **Embedder CPU.** VRAM aman, tetapi latensi retrieval naik dan belum diukur.
- **Dua proses (retrieval dan generation).** Lebih bersih untuk produksi, tetapi
  menambah kompleksitas demo.
