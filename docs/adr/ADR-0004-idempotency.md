# ADR-0004 — Idempotensi artefak ingestion

**Status:** diterima
**Tanggal:** 2026-11-14

## Konteks

Tahap ingestion dijalankan berkali-kali: saat aturan cleaning diperbaiki, saat
fetch terputus, dan saat laporan perlu dibuat ulang. Tanpa idempotensi, setiap
run menghasilkan hash berbeda dan `snapshot_id` tidak bisa dipakai sebagai bukti
"data yang dipakai di laporan adalah data ini".

Tiga sumber ketidakstabilan ditemukan pada implementasi awal:

1. `pages_index.jsonl` dan `cleaned.jsonl` memakai append, sehingga run ulang
   (atau crash setelah append) bisa meninggalkan baris ganda.
2. `fetched_at` (jam dinding) ikut tersimpan di artefak kanonik.
3. Urutan baris mengikuti urutan kedatangan, bukan kunci stabil.

## Keputusan

* **Tulis atomik**: semua file kanonik ditulis ke `<nama>.tmp` lalu `os.replace`.
  Tidak ada file parsial.
* **Upsert, bukan append**: register dan hasil cleaning dimuat sebagai `dict`
  ber-key `doc_id`, diperbarui, lalu ditulis ulang dalam urutan `doc_id`.
* **Tanpa jam dinding**: `created_at` dan `git_commit` pindah ke `provenance.json`,
  yang tidak ikut hash konten.
* **Versi di dalam data**: setiap baris `cleaned.jsonl` menyimpan `cleaner_version`
  dan `clean_config_sha`; perubahan kode/konfigurasi memaksa rebuild, bukan
  mencampur dua generasi hasil.
* **Resume berbasis isi**: "sudah diambil" berarti hash file tersimpan cocok,
  bukan sekadar file ada.
* **Hash konten eksplisit**: `manifest.json` memuat `content_files`, sementara
  `config_snapshot.json` menyimpan konfigurasi yang dipakai sehingga
  `snapshot_id` bisa dihitung ulang tanpa mempercayai file konfigurasi saat ini.

## Konsekuensi

* Run kedua pada korpus yang tidak berubah: 0 request jaringan, 0 dokumen ditulis
  ulang, hash semua artefak identik.
* `fetch_log.jsonl` tetap append-only dan **tidak** masuk hash konten; ia mencatat
  apa yang terjadi, bukan apa yang benar. Ini ditandai di manifest sebagai audit.
* Migrasi skema tidak perlu fetch ulang: register lama dibersihkan dari field yang
  tidak dikenal dan hash-nya di-backfill sekali.
* Perintah `pokerag-ingest verify` menghitung ulang seluruh hash dan `snapshot_id`;
  ia gagal bila satu byte berubah.
