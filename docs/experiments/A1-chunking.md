# A1 — audit chunking (tanpa retrieval)

> Dibuat otomatis oleh `pokerag-chunk report`; tidak ada chunk yang dibangun
> ulang dan tidak ada vektor yang disentuh. Angka berasal dari
> `chunks/<strategi>.parquet` dan `documents.parquet` pada snapshot ini.

- snapshot: `pokerag-corpus-v1` — `5f898e4f98813909`
- tokenizer: `hf:BAAI/bge-m3`, budget 1024 token, `min_chars_keep` 80 karakter

## Ukuran dan biaya

| Strategi | Unit | Parent | char p50 / p95 / max | token p50 / p95 / max | total token | dibuang | < min | cakupan |
|---|---|---|---|---|---|---|---|---|
| `fixed` | 9887 | 0 | 2048 / 2048 / 2048 | 597 / 781 / 1005 | 6,031,126 | 0 | 0 | 100.0% |
| `heading` | 23767 | 0 | 1112 / 1200 / 1200 | 302 / 445 / 627 | 5,864,406 | 512 | 680 | 99.8% |
| `hierarchical` | 24444 | 13410 | 600 / 1000 / 1000 | 241 / 363 / 512 | 5,630,591 | 1296 | 894 | 99.9% |

## Integritas batas potongan

| Strategi | mulai bersih | berhenti bersih | mulai di heading | mulai di tengah kata | berhenti di tengah kata | memotong batas h2 | mirip tabel |
|---|---|---|---|---|---|---|---|
| `fixed` | 6.6% | 3.4% | 3.3% | 57.3% | 55.8% | 2130 | 6.8% |
| `heading` | 38.2% | 26.7% | 35.8% | 36.9% | 35.7% | 0 | 7.4% |
| `hierarchical` | 15.9% | 11.1% | 13.0% | 51.4% | 50.9% | 0 | 7.9% |

## Fragmentasi section

| Strategi | section | utuh dalam 1 chunk | pecah ≥3 chunk | p50 | p95 | maks | tidak tersentuh |
|---|---|---|---|---|---|---|---|
| `fixed` | 12981 | 54.3% | 15.3% | 1 | 5 | 34 | 0 |
| `heading` | 12981 | 40.0% | 29.0% | 2 | 7 | 55 | 512 |
| `hierarchical` | 12981 | 50.3% | 24.4% | 1 | 10 | 77 | 511 |

## Redundansi overlap

| Strategi | karakter chunk | karakter unik | duplikat | redundansi | token duplikat (≈) |
|---|---|---|---|---|---|
| `fixed` | 19,953,878 | 17,504,726 | 2,449,152 | 12.3% | 612,288 |
| `heading` | 19,365,170 | 17,471,279 | 1,893,891 | 9.8% | 473,473 |
| `hierarchical` | 18,588,371 | 17,479,254 | 1,109,117 | 6.0% | 277,279 |

## Distribusi ukuran (token, unit saja)

| Strategi | <200 | 200-400 | 400-600 | 600-800 | 800-1024 | >1024 |
|---|---|---|---|---|---|---|
| `fixed` | 91 | 95 | 4919 | 4390 | 392 | 0 |
| `heading` | 9522 | 12238 | 2006 | 1 | 0 | 0 |
| `hierarchical` | 10677 | 13113 | 654 | 0 | 0 | 0 |

## Struktur parent/child (`hierarchical`)

- parent: 13,410 (tidak di-embed), child: 24,444
- child per parent: p50 2, p95 2, maks 2
- parent tanpa child: 0
- token di parent: 5,286,436 (tidak masuk index)
- pelanggaran invariant child ⊂ parent: **0**

## Definisi

| Metrik | Definisi |
|---|---|
| Unit | chunk yang di-embed (`role != parent`) |
| Parent | baris `hierarchical` untuk context expansion, tidak di-embed |
| mulai bersih | chunk mulai di `char_start` sebuah section, atau karakter non-spasi sebelumnya adalah `.` `!` `?` `:` `;` atau tanda kutip/penutup |
| berhenti bersih | karakter non-spasi sebelum `char_end` adalah akhir kalimat, atau chunk berakhir di akhir dokumen |
| mulai tepat di heading | `char_start` sama dengan `char_start` sebuah section |
| tengah kata | karakter sebelum dan sesudah batas keduanya huruf/angka |
| memotong batas h2 | ada section level 2 yang mulai di dalam rentang chunk |
| mirip tabel | lebih dari 30% karakter chunk bukan huruf/spasi |
| fragmentasi section | jumlah chunk yang beririsan dengan satu section |
| utuh dalam 1 chunk | section hanya disentuh satu chunk — syarat satu chunk bisa dikutip untuk satu fakta |
| redundansi overlap | (jumlah karakter chunk − gabungan span) / jumlah karakter chunk |
| token duplikat | perkiraan dari redundansi karakter (÷4), bukan hitungan tokenizer |
| cakupan | proporsi teks dokumen yang tercakup minimal satu chunk |

## Ancaman validitas

* Deteksi batas kalimat berbasis karakter, bukan parser kalimat; singkatan
  seperti "vs." dapat dihitung sebagai akhir kalimat.
* `mirip tabel` adalah proksi; tabel dengan banyak kata tetap terhitung
  sebagai prosa.
* Token duplikat adalah perkiraan; laporan retrieval nanti memakai token
  nyata dari konteks top-k.
* Section kecil (mis. heading kosong) tetap dihitung; jumlah section tidak
  sama dengan jumlah fakta.

## Reproduksi

```bash
pokerag-chunk build
pokerag-chunk report
```
