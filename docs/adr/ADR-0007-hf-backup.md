# ADR-0007 — Backup korpus ke Hugging Face (termasuk teks mentah)

**Status:** diterima
**Tanggal:** 2026-11-14

## Konteks

Korpus menghasilkan ~550 MB artefak: 320 HTML mentah, wikitext, dokumen bersih,
tiga set chunk, vektor, dan index Qdrant/BM25. GitHub bukan tempatnya (batas
ukuran, dan repo kode sebaiknya tetap ringan). Korpus juga tidak boleh hilang
karena satu kesalahan `rm -rf` atau disk rusak, sementara membangun ulang
memerlukan crawl 32 menit + embedding 58 menit.

Teks Bulbapedia berlisensi **CC BY-NC-SA 2.5**. Mengunggahnya berarti
redistribusi, yang sebelumnya dinonaktifkan (`allow_redistribute = false`).

## Keputusan

* Backup penuh ke **dataset repo Hugging Face**, **private**, satu commit per
  upload folder.
* Lisensi CC BY-NC-SA 2.5 **mengizinkan** redistribusi non-komersial dengan
  atribusi dan ShareAlike. Karena itu `allow_redistribute` diubah maknanya
  menjadi "boleh untuk arsip privat dengan atribusi", bukan "boleh
  didistribusikan ulang secara publik". Repo tetap private agar tidak menjadi
  sumber unduhan publik.
* Dataset card wajib memuat: lisensi, format atribusi
  (`Bulbapedia, CC BY-NC-SA 2.5, revision <revid>`), batasan non-komersial,
  ShareAlike, dan pernyataan bahwa gambar/fair-use tidak disertakan.
* `INTEGRITY.sha256` disertakan agar setiap file dapat diverifikasi setelah
  diunduh.

## Konsekuensi

* 670 file, ~435 MB. `indexes/` dibungkus `indexes.tar.gz` karena Qdrant lokal
  membuat ratusan file kecil dan lock file yang tidak berguna untuk di-version.
* Backup bersifat arsip, bukan publikasi: repo private, dan dataset card tetap
  menyatakan NC agar tidak disalahgunakan bila aksesnya dibuka nanti.
* Jika korpus berubah, `snapshot_id` berubah dan backup lama tetap valid sebagai
  arsip keadaan sebelumnya — bukan ditimpa diam-diam.
* Yang **tidak** diunggah: file sementara, lock Qdrant, dan `.gitkeep`.
