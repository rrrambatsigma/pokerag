# ADR-0016 — Backend Qwen dan runner dua tahap

**Status:** diterima
**Tanggal:** 2026-11-22

## Konteks

Generator perlu dijalankan untuk B0 sampai B6. GPU yang tersedia 6 GB, sedangkan
`bge-m3` untuk query dan Qwen2.5-1.5B harus dipakai di mesin yang sama. Pada uji
pertama, keduanya dimuat bersamaan dan proses kehabisan memori: 5,27 GB terpakai
dan alokasi 490 MB gagal.

## Keputusan

1. **Model dipin** ke `Qwen/Qwen2.5-1.5B-Instruct` revisi
   `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, dimuat fp16. `require_pinned()`
   menolak run bila revisi kosong.
2. **Runner dua tahap.** Semua retrieval diselesaikan lebih dulu, embedder
   dilepas, `torch.cuda.empty_cache()` dipanggil, baru generator dimuat. Puncak
   VRAM menjadi maksimum dari keduanya (sekitar 3,1 GB), bukan jumlahnya.
3. **Warmup tidak dihitung sebagai latensi.** Pemuatan `bge-m3` memakan sekitar
   30 detik di mesin ini; warmup dilakukan sebelum timer, sehingga latensi
   permintaan mencerminkan retrieval dan generation, bukan pemuatan model.
4. **Parser menerima JSON prompt**, lalu jatuh ke sitasi `[n]` bila model
   mengabaikan skema. Nomor sitasi dipetakan ke `chunk_id` berdasarkan urutan
   evidence; nomor di luar rentang diabaikan.
5. **Seed per run** diambil dari `sampling.seeds[run_index]`. Tiga seed sudah
   disiapkan, jadi tiga run memakai seed berbeda tanpa mengubah config.
6. **Serving belum memakai backend ini.** API tetap memakai stub karena
   `bge-m3` dan Qwen tidak muat bersamaan di 6 GB. Pilihan berikutnya: embedder
   di CPU, Qwen 4-bit, atau retrieval di proses terpisah.

## Konsekuensi

- Latensi retrieval dan generation digabung per query. Pemuatan model tidak
  termasuk; angka bisa dibandingkan antar run pada mesin yang sama.
- Dua tahap berarti urutan query tidak memengaruhi hasil, tetapi seluruh hasil
  retrieval disimpan di memori sampai generation selesai.
- Parser yang toleran menyelamatkan output model kecil, tetapi kualitas sitasi
  tetap harus dinilai lewat metrik generation, bukan diasumsikan dari format.

## Alternatif yang ditolak

- **Memuat keduanya sekaligus.** Terbukti OOM pada 6 GB.
- **Embedder di CPU sekarang.** Lebih lambat untuk 100 query dan belum perlu
  selama eksperimen offline.
- **Memaksa JSON ketat tanpa fallback.** Model 1,5B kadang menjawab teks biasa;
  membuang jawaban itu akan menghukum model karena format, bukan isi.
