# ADR-0013 — Run manifest dan arti "tiga run independen"

**Status:** diterima
**Tanggal:** 2026-11-22

## Konteks

Brief menuntut minimal tiga run independen per konfigurasi, manifest yang
mencatat versi kode, prompt, model, index, seed, parameter sampling, token, dan
latensi, lalu variance dilaporkan. Tanpa manifest, angka di laporan tidak bisa
dibedakan dari kebetulan.

Yang belum ada: satu tempat yang mengikat satu angka ke keadaan yang
menghasilkannya.

## Keputusan

1. **Satu manifest per run**, disimpan di `artifacts/runs/<run_id>/manifest.json`,
   bersama `requests.jsonl` berisi satu baris per permintaan.
2. **Identitas yang dicatat:** commit git dan status dirty, hash config, prompt,
   qrels, dan query set, `snapshot`, `index_version`, strategi, mode, spesifikasi
   model beserta revisinya, dan parameter sampling termasuk seed.
3. **Nilai yang tidak diukur tetap `None`.** Run tanpa pencatatan token tidak
   boleh terlihat seperti run dengan token nol.
4. **`run_set` dan `run_index`** memisahkan kelompok run dari ulangannya. Variance
   dihitung lintas `run_index` dengan setelan yang sama.
5. **Independensi.** Untuk retrieval murni, index bersifat deterministik, jadi
   membangun ulang index tidak wajib; yang wajib adalah `index_version` identik.
   Untuk generation, seed dan sampling harus benar-benar berbeda tiap run.
   Kalau index dibangun ulang, manifest lama dan baru hanya sah dibandingkan bila
   `index_version` sama.
6. **Trace per permintaan** menyimpan latensi, token masuk dan keluar, status
   abstain, sitasi, dan error. Rangkuman p50/p95 dihitung dari trace, bukan
   diketik manual.

## Konsekuensi

- Path absolut di manifest bisa membocorkan struktur direktori. Sebelum
  publikasi, ganti dengan path relatif atau hash saja.
- Token hanya sebanding bila memakai tokenizer model yang sama. Manifest mencatat
  model dan revisinya, jadi perbandingan lintas model harus hati-hati.
- Latensi bergantung mesin. Yang boleh diklaim adalah perbandingan relatif pada
  mesin yang sama, bukan angka absolut.
- Manifest bukan bagian dari hash konten, jadi waktu pembuatan tidak menggeser
  `index_version`.

## Alternatif yang ditolak

- **Log konsol saja.** Tidak bisa dihitung ulang setelah sesi selesai.
- **Satu manifest untuk semua run.** Menghapus justru hal yang ingin dibuktikan:
  variasi antar run.
- **Mengisi field yang tidak diukur dengan nol.** Itu membuat klaim token dan
  latensi tidak jujur.
