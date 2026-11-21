# ADR-0001 — Seleksi korpus: National Pokédex 1–320

**Status:** diterima
**Tanggal:** 2026-11-14

## Konteks

Brief mewajibkan minimal 300 dokumen unik dari sumber nyata. Pilihan yang mudah
adalah mengambil 300 judul pertama dari `Category:Pokémon`, tetapi urutannya
alfabetis (`Abomasnow`, `Abra`, `Absol`, ...). Itu menghasilkan korpus yang
condong ke nama A–C dan memutus keluarga evolusi, sehingga pertanyaan multi-hop
seperti "apa yang dibutuhkan untuk mengevolusi X" kehilangan bukti.

## Keputusan

Ambil rentang **National Pokédex 1–320**, dibaca dari halaman
`List of Pokémon by National Pokédex number` melalui template `{{ndex}}`.
Ambil 320 (bukan 300) sebagai buffer untuk halaman disambiguasi, stub, dan duplikat.

## Konsekuensi

* Seleksi deterministik dan bisa direproduksi: aturannya satu baris di YAML.
* Korpus mencakup Gen I–III lengkap dengan garis evolusi utuh.
* Cakupan bias ke tiga generasi pertama; ini keterbatasan yang harus disebut di laporan,
  bukan disembunyikan.
* Jika gate 300 gagal, yang diperluas adalah rentang dex (mis. 1–350), bukan ambang gate.
