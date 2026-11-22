# B5 supervised dataset

Data training untuk adapter B5. Isinya **fakta**, bukan teks korpus.

Korpus Bulbapedia berlisensi CC BY-NC-SA 2.5 dengan `allow_ft_training = false`,
jadi tidak satu pun kalimat di direktori ini boleh berasal dari korpus. Nilai
(tipe, catch rate, egg group, level evolusi) diekstrak dari snapshot, lalu
kalimat pertanyaan, evidence, dan jawabannya ditulis ulang sendiri oleh
`pokerag.sft.render`. Gate di `pokerag.sft.leak` membuktikannya secara mekanis.

## Berkas

| Berkas | Isi |
| --- | --- |
| `items.jsonl` | 100 butir fakta unik, satu per Pokémon |
| `train.jsonl` | 258 contoh chat (86 butir × 3 varian) |
| `val.jsonl` | 42 contoh chat (14 butir × 3 varian) |
| `manifest.json` | hash korpus, hash query set, versi prompt, hasil gate |

Total 300 contoh. Split dilakukan per butir, bukan per contoh, supaya entitas
val tidak bocor ke training lewat varian lain.

## Skema satu butir

```json
{
  "item_id": "sft0001",
  "entity": "Hoothoot",
  "kind": "catch_rate",
  "question": "What is the catch rate of Hoothoot?",
  "evidence": "Hoothoot has a catch rate of 255, which equals 43.9%.",
  "answer": "Hoothoot has a catch rate of 255 (43.9%)",
  "provenance": {
    "doc_id": "bulba:0002506",
    "revid": 4651726,
    "locator": "Infobox",
    "values": ["255", "43.9"]
  }
}
```

`provenance` ada supaya nilainya bisa dicek ulang terhadap snapshot. `values`
adalah satu-satunya hal yang diambil dari dokumen.

## Varian

Setiap butir menghasilkan tiga contoh, semuanya memakai prompt serving yang sama
(`configs/prompts/grounded_system.txt` dan `grounded_user.txt`):

| Varian | Evidence | Target |
| --- | --- | --- |
| `clean` | fakta yang benar + dua distraktor beda jenis | jawaban + sitasi |
| `hard` | fakta yang benar + dua near miss sejenis (entitas lain) | jawaban + sitasi |
| `abstain` | hanya near miss, tanpa fakta yang benar | JSON abstain |

Tujuannya bukan menambah pengetahuan, tapi menggeser perilaku: memilih evidence
yang benar di antara yang mirip, menyitir nomor yang tepat, dan berhenti saat
evidence tidak memuat jawaban.

## Gate

| Gate | Ambang |
| --- | --- |
| Entitas training vs test/dev | 0 irisan |
| Pertanyaan training vs pertanyaan test | 0 duplikat |
| 8-gram kata vs dokumen asal | 0 temuan |
| Longest common substring vs dokumen asal | < 50 karakter |

Hasil terakhir: 0 irisan entitas, 0 duplikat pertanyaan, 0 temuan 8-gram,
substring terpanjang 35 karakter.

## Membangun ulang

```bash
uv run python -m pokerag.sft.cli build
uv run python -m pokerag.sft.cli check
```

`build` menulis ulang `items.jsonl`, `train.jsonl`, `val.jsonl`, dan
`manifest.json`, lalu menjalankan gate. `check` memverifikasi ulang setiap butir
terhadap snapshot dan menjalankan gate yang sama; keduanya keluar dengan status
non-nol kalau ada pelanggaran.

Daftar entitas yang dilarang ditulis ke `artifacts/sft/excluded_entities.txt`
(gitignored), karena berasal dari test set.

## Batasan yang harus disebut di laporan

Evidence di dataset ini **semi-grounded**: kalimatnya kita tulis sendiri dan
lebih pendek serta lebih bersih daripada chunk hasil retrieval sungguhan. Karena
itu gain yang muncul sebagian berasal dari format yang lebih mudah, bukan dari
pengetahuan domain. Evaluasi tetap harus memakai chunk asli lewat B4, dan
perbandingannya harus disebut apa adanya.
