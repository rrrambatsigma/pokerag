# ADR-0002 — Cleaning dari HTML, bukan bedah wikitext

**Status:** diterima
**Tanggal:** 2026-11-14

## Konteks

Dua jalur tersedia:

1. **Wikitext** (`prop=revisions&rvprop=content`): satu request per batch, tetapi
   penuh template (`{{ndex}}`, `{{Infobox Pokémon}}`, `{{learnlist/...}}`), tabel
   `{| |}`, dan parser buatan sendiri yang rapuh.
2. **HTML hasil parse** (`action=parse&oldid=<revid>`): satu request per halaman,
   tetapi struktur sudah jadi — heading, tabel, infobox — dan bisa dibersihkan
   dengan selector CSS.

## Keputusan

Pakai keduanya: wikitext untuk identitas revisi dan deteksi redirect/disambiguasi,
HTML untuk teks bersih. HTML dipin ke `oldid=<revid>` agar selalu menggambarkan
revisi yang sama.

## Konsekuensi

* Biaya: 320 request HTML tambahan → total ~384 request, ~32 menit pada crawl-delay 5 s.
  Diterima karena hanya sekali, dan semuanya di-cache ke `artifacts/`.
* Selector harus diuji terhadap markup nyata. Probe menemukan bahwa sebagian besar
  halaman dibungkus `table.roundy` dan navigasi bawah berupa rantai `div` tanpa class,
  sehingga filter berbasis class saja tidak cukup: ditambahkan `is_link_farm`
  (rasio tautan terhadap teks) dan `strip_junk_lines`.
* Invariant `text[char_start:char_end]` diuji per dokumen, sehingga kesalahan
  pemetaan offset gagal di CI, bukan diam-diam masuk ke chunking.
