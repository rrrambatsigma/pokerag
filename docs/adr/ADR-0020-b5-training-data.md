# ADR-0020 — Data training B5 dari fakta, bukan dari teks korpus

**Status:** diterima
**Tanggal:** 2026-11-22

## Konteks

Brief mewajibkan B5 sebagai adapter LoRA/QLoRA pada base generator yang sama,
dan A4 membandingkannya dengan parent B4 yang identik. Korpus proyek ini
Bulbapedia, CC BY-NC-SA 2.5, dan `docs/legal/data-license.md` sudah menetapkan
`allow_ft_training = false` untuk seluruh 386 dokumen.

Artinya data training tidak boleh memuat teks korpus. Yang tidak dilarang adalah
fakta: tipe, catch rate, egg group, dan level evolusi bukan ekspresi yang
dilindungi, dan tidak ada cara lain menuliskannya. ADR-0015 mencatat B5 nonaktif
sambil menunggu keputusan; ADR ini mengisi keputusan itu.

## Keputusan

1. **Data training dibangun dari nilai, bukan kalimat.** `pokerag.sft.facts`
   mengekstrak nilai dari snapshot, `pokerag.sft.render` menulis pertanyaan,
   evidence, dan jawaban sendiri. Tidak ada satu kalimat pun yang disalin.
2. **Ukuran: 100 butir unik, 300 contoh.** Empat jenis fakta dengan kuota
   40 type, 25 catch rate, 20 egg group, 15 evolution. Tiap butir menghasilkan
   varian `clean`, `hard`, dan `abstain`.
3. **Entitas training tidak boleh muncul di test atau dev.** Daftar terlarang
   diambil dari kolom `entity` dan `expected_doc` query set, ditambah `doc_id`
   dev qrels dan nama entitas yang muncul di pertanyaan dev.
4. **Gate mekanis sebelum training:** 0 irisan entitas, 0 pertanyaan duplikat,
   0 temuan 8-gram kata, dan longest common substring di bawah 50 karakter
   terhadap dokumen asal.
5. **Prompt training identik dengan prompt serving**, yaitu
   `configs/prompts/grounded_system.txt` dan `grounded_user.txt`, supaya
   distribusi training dan inference tidak berbeda.
6. **Split per butir, bukan per contoh**, supaya entitas val tidak bocor ke
   training lewat varian lain.

## Konsekuensi

- Dataset boleh di-commit karena seluruh isinya tulisan sendiri. Yang tidak
  masuk repo adalah daftar entitas terlarang (`artifacts/sft/excluded_entities.txt`),
  karena itu metadata test set.
- Base Qwen2.5-1.5B-Instruct berlisensi Apache-2.0, jadi adapter boleh dilatih
  dan disebar dengan notice yang dipertahankan. Karena data training tidak
  mengandung teks CC BY-NC-SA, tidak ada kewajiban ShareAlike yang menempel
  pada bobot adapter.
- Evidence di dataset ini **semi-grounded**: kalimatnya pendek dan bersih,
  berbeda dari chunk retrieval sungguhan yang panjang dan berisik. Gain yang
  muncul sebagian berasal dari kemudahan format. Ini batasan, bukan temuan.
- Varian `abstain` sengaja 100 contoh, sepertiga dataset, karena perilaku
  abstain adalah yang paling mudah rusak oleh fine-tuning.

## Angka

| Metrik | Nilai |
| --- | --- |
| Butir | 100 |
| Contoh train / val | 258 / 42 |
| Kuota | type 40, catch_rate 25, egg_group 20, evolution 15 |
| Irisan entitas dengan test/dev | 0 |
| Duplikat pertanyaan test | 0 |
| 8-gram terhadap dokumen asal | 0 |
| Substring terpanjang | 35 karakter (ambang 50) |
| Versi prompt | `064e878946e1` |

## Alternatif yang ditolak

- **Training pada teks korpus.** Paling mudah dan paling sesuai "grounded",
  tetapi melanggar lisensi dan berkasnya harus ditarik lagi nanti.
- **Fakta dari data Showdown (MIT) saja.** Legalitasnya lebih ringan, tetapi
  memisahkan sumber fakta dari sumber retrieval, sehingga klaim B5 menjadi
  "adapter dengan sumber data lain", bukan "adapter pada sistem yang sama".
  Jalur ini tetap tersedia kalau ada keberatan atas jalur korpus.
- **Melewati B5 dan hanya menulis batasan.** Sempat direkomendasikan, tetapi
  100 butir hanya perlu beberapa jam dan tidak memblokir pekerjaan lain.

## Reproduksi

```bash
uv run python -m pokerag.sft.cli build
uv run python -m pokerag.sft.cli check
```

`build` keluar non-nol kalau gate gagal. `check` memverifikasi ulang nilai tiap
butir terhadap snapshot dan menjalankan gate yang sama, sehingga korpus yang
bergeser tidak bisa lolos diam-diam.
