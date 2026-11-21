# ADR-0015 — Tangga B0-B6 dan kebijakan prompt

**Status:** diterima
**Tanggal:** 2026-11-22

## Konteks

Brief meminta tujuh konfigurasi di atas satu base generator, dengan B6
dibandingkan ke parent B4. Yang mudah salah bukan daftar tingkatnya, melainkan
prompt: kalau B0 memakai prompt grounded yang sama dengan B2, B0 akan selalu
abstain karena tidak ada evidence, dan tangga kehilangan baseline.

## Keputusan

1. **Satu `configs/ladder.yaml`** berisi B0 sampai B6, divalidasi: urutan, mode
   retrieval per tingkat, flag planner/verifier, adapter, dan tool.
2. **B0 dan B1 tidak memakai evidence.** Keduanya memakai `plain_user.txt`.
   B0 memakai prompt polos; B1 menambahkan few-shot, constraint, skema output,
   dan policy abstain. Jadi selisih B1 terhadap B0 hanya efek cara bertanya.
3. **B2 sampai B6 memakai prompt grounded.** Evidence diberi nomor `[n]`, dan
   prompt menyatakan evidence adalah data, bukan instruksi. Ini juga garis
   pertahanan yang diuji suite injection.
4. **Skema output JSON** untuk semua konfigurasi yang memakai prompt terstruktur:
   `{"answer": ..., "citations": [n], "abstain": ...}`. Nomor sitasi memetakan
   langsung ke urutan evidence.
5. **Tiga seed** (`7, 17, 27`) untuk tiga run independen. Base model harus
   di-pin revisinya sebelum run; `require_pinned()` menolak run dengan
   `revision: null`.
6. **B5 dinyatakan nonaktif** karena `allow_ft_training=false` untuk seluruh 386
   dokumen. Alasannya tercatat di config. Keputusan akhir, dataset pengganti
   atau tidak dijalankan, masih terbuka.
7. **B6 menyimpan nama tool** (`document_version`, `generation_check`) tanpa
   implementasi. Tool nyata dan trace-nya menyusul; suite tool misuse baru
   bermakna penuh setelah itu.

## Konsekuensi

- B0 sengaja boleh menjawab dari parameter. Itu yang membuat
  `parametric_leak` bisa diukur, bukan dihilangkan.
- Prompt version adalah hash isi; run manifest mencatatnya, sehingga perubahan
  satu kata pada prompt terlihat.
- Validasi menolak `mode: off` untuk B2 ke atas, dan menolak B2 yang bukan dense.
- YAML 1.1 membaca `off` sebagai boolean. Nilainya harus ditulis `"off"`.

## Alternatif yang ditolak

- **Satu prompt untuk semua tingkat.** B0 akan abstain selalu; baseline menjadi
  tidak bermakna.
- **Prompt terpisah per file config.** Tujuh file dengan isi hampir sama lebih
  mudah menyimpang daripada satu tangga yang divalidasi.
- **Mengaktifkan B5 diam-diam.** Itu melanggar flag lisensi dan membuat klaim
  adaptation tidak sah.
