# M0: impor dan verifikasi artefak

Tanggal: 21 November 2026

## Sumber

| Item | Nilai |
| --- | --- |
| Dataset | `meivawijaya/pokerag-corpus-v3` |
| Snapshot | `5eda7c8efc03f6fe1e892289093e04912854aa8addc28ead50cac700dc583e1c` |
| Arsip index | sha256 `044b4887be885188da4dcc09187f9623d923df1e6ae09af79eb1d981cc3aa2b8` |
| `INTEGRITY.sha256` | sha256 `72a4db6a4a9f7237916f1f702d26148af84364a34f15a79711a8a19f1633c299` |
| File diunduh | 28 dari 804, semuanya cocok hash |

Artefak mentah 324 MB, hasil ekstraksi 709 MB. Keduanya tidak masuk git.

## Hasil verifikasi

| Strategi | Koleksi | Poin | Baris parquet | Span rusak |
| --- | --- | --- | --- | --- |
| fixed | `chunks_fixed` | 11.642 | 11.642 | 0 |
| heading | `chunks_heading` | 27.902 | 27.902 | 0 |
| hierarchical | `chunks_hierarchical` | 28.786 | 44.589 (28.786 child + 15.803 parent) | 0 |

BM25 untuk ketiga strategi memuat dengan jumlah dokumen yang sama. Span
diperiksa untuk seluruh chunk, bukan sampel: `documents.text[char_start:char_end]`
harus sama dengan `chunks.text`, dan hasilnya nol pelanggaran.

`index_version` fixed: `75f6330e7f014650e616eef24e54d29141d2a7cbac8cfe635a255fd381f90775`.

## Lingkungan

| Paket | Versi |
| --- | --- |
| Python | 3.12.13 |
| qdrant-client | 1.20.0 |
| bm25s | 0.3.16 |
| pandas | 3.1.1 |
| pyarrow | 26.0.1 |
| numpy | 2.5.5 |

## Temuan lisensi

| Flag | Nilai | Akibat |
| --- | --- | --- |
| `allow_retrieval` | true untuk 386 dokumen | retrieval aman |
| `allow_redistribute` | false untuk 386 dokumen | teks tidak boleh didistribusikan, termasuk ke repo publik |
| `allow_ft_training` | false untuk 386 dokumen | B5 tidak boleh dilatih pada teks ini |

`allow_ft_training=false` berarti adapter LoRA/QLoRA di B5 memerlukan dataset
berlisensi terpisah, atau B5 dinyatakan tidak dijalankan sesuai kelonggaran di
brief. Keputusan ini belum diambil dan perlu dicatat sebagai ADR sebelum M5.

## Reproduksi

```bash
uv sync
uv run python scripts/fetch_artifacts.py --extract
uv run python scripts/verify_index.py
```

Keluaran terakhir: `M0 index verification passed`.
