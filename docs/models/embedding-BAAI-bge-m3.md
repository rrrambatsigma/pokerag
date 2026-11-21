# Identitas model embedding — BAAI/bge-m3

> Dibuat otomatis oleh `pokerag-index build`. Jangan diedit tangan:
> sumbernya `artifacts/models/embedder_identity.json`, yang sha256-nya
> tercatat di manifest setiap koleksi.

| Field | Nilai |
|---|---|
| Nama dan tag | `BAAI/bge-m3` (`main` @ `5617a9f61b028005a4858fdac845db406aefb181`) |
| Digest | `sha256:b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38` (pytorch_model.bin) |
| Arsitektur | XLMRobertaModel |
| Jumlah parameter | 567,754,752 |
| Kuantisasi | bobot float32, inferensi float16 |
| Dimensi embedding | 1024 |
| Panjang konteks | model 8194 token (config); dipakai 1024 token |
| Format file | PyTorch bin |
| Lisensi | mit |
| Pooling | cls, L2-normalised |
| Tokenizer | `BAAI/bge-m3` |
| Perangkat | NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GiB) |
| Runtime | cuda 12.4, sentence_transformers 6.3.0, torch 2.6.0+cu124 |
| Chunk terpotong | 0 dari 68330 (maksimum 1005 token, batas 1024) |

## Cara membacanya

* **Digest** adalah SHA-256 berkas bobot pada revisi yang dipin, bukan
  hash direktori unduhan. Revisi dipin di config (`embedder.revision`),
  sehingga unduhan ulang tidak bisa menukar bobot di balik nama model.
* **Kuantisasi**: bobot di repositori disimpan pada presisi tertentu
  (dihitung dari byte/parameter), lalu dimuat ke VRAM sebagai
  `dtype_inference`. Keduanya dicatat karena keduanya memengaruhi vektor.
* **Panjang konteks**: `model_max_tokens` diambil dari
  `config.max_position_embeddings` model (kartu model biasanya menyebut
  8192; selisih 2 posisi disediakan untuk penanda khusus). Build hanya
  memakai `used_max_tokens`, dan chunk yang melewati batas itu akan
  dipotong sebelum di-embed, karena itu jumlah chunk terpotong wajib 0.

## Batas yang tidak diklaim

Tabel ini menyatakan identitas model dan konfigurasi embedding. Ia tidak
menyatakan kualitas retrieval; itu diukur terpisah di laporan A1.
