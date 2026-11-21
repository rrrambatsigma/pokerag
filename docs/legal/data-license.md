# Data licence register — PokéRAG corpus

## Sumber

| Field | Nilai |
|---|---|
| Situs | Bulbapedia (https://bulbapedia.bulbagarden.net) |
| Lisensi konten | CC BY-NC-SA 2.5 |
| URL lisensi | http://creativecommons.org/licenses/by-nc-sa/2.5/ |
| Diverifikasi pada | 2026-11-14, `action=query&meta=siteinfo&siprop=rightsinfo` |
| `robots.txt` | `Allow: /wiki/`, `Crawl-delay: 5` |

## Apa yang boleh

* Digunakan untuk riset akademik dan evaluasi retrieval (non-komersial).
* Disimpan lokal sebagai snapshot untuk eksperimen yang bisa direproduksi.
* Dikutip sebagai evidence dengan atribusi + nomor revisi.

## Apa yang tidak boleh

* **Redistribusi** teks korpus (repo, lampiran, dataset publik).
* **Fine-tuning** di atas teks ini tanpa dasar hukum yang jelas. Lisensi NC-SA
  membatasi penggunaan komersial dan mewajibkan ShareAlike; status data turunan
  untuk training bukan hal yang bisa diasumsikan.

## Konsekuensi teknis

* `allow_redistribute = false` dan `allow_ft_training = false` disimpan per dokumen
  di `register.parquet` dan `documents.parquet`.
* `artifacts/**` masuk `.gitignore`; hanya hash, URL, `revid`, dan laporan yang di-commit.
* Gate `lisensi: ft_training ditolak` gagal bila ada dokumen berizin training
  yang lolos ke korpus final.
* Untuk B5, dataset training harus berasal dari sumber berlisensi permisif
  (mis. data Pokémon Showdown) atau ditulis sendiri atas fakta, dengan ADR terpisah.

## Atribusi

Format: `Bulbapedia, CC BY-NC-SA 2.5, revision <revid>` — disimpan di kolom
`attribution` setiap dokumen dan wajib muncul di UI saat evidence ditampilkan.
