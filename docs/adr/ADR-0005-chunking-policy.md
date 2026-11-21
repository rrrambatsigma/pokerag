# ADR-0005 — Chunking policy: fixed untuk B2–B6, heading untuk A1

**Status:** diterima
**Tanggal:** 2026-11-14

## Konteks

Brief menetapkan B2 (Naive RAG) memakai *fixed chunking*. B3–B6 tidak
menyebut perubahan chunking. Di sisi lain, data korpus sangat tidak rata:
12.981 section, median 355 karakter, 45,4% di bawah 300 karakter, 17,1% di atas
2.048 karakter (maksimum 57.710).

## Keputusan

1. **Jalur utama B2–B6 memakai `fixed`** (2.048 karakter ≈ 512 token, overlap 256).
   Alasannya metodologis: kalau chunking berubah di tengah B-series, delta
   B3−B2 mencampur efek retrieval dengan efek chunking dan tidak bisa
   diatribusikan.
2. **A1 membandingkan tiga strategi** dengan qrels identik: `fixed`,
   `heading`, `hierarchical`.
3. **Kandidat sistem terbaik: `heading`** — batas `h2`, split rekursif 1.200
   karakter, overlap 120, padding section pendek di dalam blok yang sama.
   Chunk di bawah 80 karakter dibuang karena hanya berisi tautan
   (mis. "Main article: Mew (TCG)"), bukan bukti.
4. Kalau A1 memilih `heading`, B2–B6 **dijalankan ulang** dengan chunking itu
   dan dilaporkan sebagai sistem v2 — bukan menukar chunking di satu konfigurasi.

## Konsekuensi

| | fixed | heading | hierarchical |
|---|---|---|---|
| chunk | 9.887 | 23.767 | 37.854 (24.444 child) |
| token untuk embedding | 6,03 jt | 5,86 jt | 5,66 jt (child saja) |
| ukuran vektor 1024-dim | 38,6 MB | 92,8 MB | ~95 MB |
| cakupan teks | 100% | 99,8% | 99,9% |

Jumlah chunk `heading` 2,4× lebih besar, tetapi token untuk embedding hampir
sama — biaya embedding ditentukan token, bukan jumlah chunk. Yang membesar
adalah ukuran index dan jumlah kandidat rerank.

Semua chunk menyimpan `char_start`/`char_end` sehingga bisa dipetakan kembali ke
dokumen dan dipakai untuk qrels berbasis span. Invariant `chunk.text ==
doc.text[start:end]` diuji; build gagal bila dilanggar.

## Adendum (2026-11-17): parent `hierarchical` sekarang per blok h2

Audit chunking (`pokerag-chunk report`, `docs/experiments/A1-chunking.md`)
menemukan bahwa implementasi `hierarchical` **melanggar batas h2**: 2.117 dari
23.399 child menyeberang heading, 58% dimulai di tengah kata, dan hanya 1,5%
dimulai tepat di heading — praktis identik dengan `fixed`. Penyebabnya,
`top_level_blocks(sections)` dihitung tetapi tidak dipakai; parent dibangun
sebagai jendela 1.500 karakter dari awal sampai akhir dokumen.

Perbaikan: parent dibangun **di dalam** blok h2 yang dipaksa menutup seluruh teks
(tiap blok diperpanjang sampai awal blok berikutnya), lalu child tetap jendela
1.000 karakter di dalam parent. Hasil terukur: 0 pelanggaran batas h2, mulai di
heading 1,5% → 13%, section utuh 44,8% → 50,3%, cakupan 100% → 99,9% karena
1.296 child di bawah 80 karakter (stub tautan pada blok sangat pendek) dibuang
sesuai `min_chars_keep`, sama seperti `heading` membuang 512 stub.

Konsekuensi: chunk `hierarchical` berubah, sehingga vektor dan `index_version`
koleksi itu harus dibangun ulang; `fixed` dan `heading` tidak tersentuh. Angka
token/vektor di tabel atas sudah memperhitungkan versi baru.

