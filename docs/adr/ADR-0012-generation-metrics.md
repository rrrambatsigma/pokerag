# ADR-0012 — Metrik generation: otomatis vs dinilai

**Status:** diterima
**Tanggal:** 2026-11-22

## Konteks

Retrieval sudah punya metrik, tetapi separuh alat ukur brief belum ada:
kepatuhan abstain, mutu sitasi, dan kebenaran jawaban. Slice unanswerable juga
baru bisa dihitung jumlahnya, belum dinilai.

Ada dua jenis metrik yang sering dicampur, padahal sumbernya berbeda:

1. **Otomatis**, bisa dihitung dari prediksi dan gold spans: apakah sistem
   abstain, apakah sitasi menunjuk bukti yang benar.
2. **Dinilai**, butuh manusia atau verifier: apakah jawaban benar, apakah klaim
   didukung evidence.

Mencampur keduanya menghasilkan angka yang tidak bisa dipertahankan, misalnya
"akurasi" yang sebenarnya hanya mencocokkan string.

## Keputusan

1. **Pisahkan metrik otomatis dan dinilai.** `abstain_correct`,
   `false_confident`, `false_refusal`, `citation_precision`, dan
   `citation_recall` dihitung otomatis. `supported_correct` dan
   `unsupported_correct` hanya dihitung bila ada judgment.
2. **Judgment hilang menghasilkan `None`, bukan nol.** Nilai `None` dikeluarkan
   dari rata-rata dan ditulis `n/a`. Kalau dianggap nol, sistem yang tidak
   dinilai akan terlihat lebih buruk dari yang sebenarnya.
3. **`parametric_leak` adalah jawaban benar pada query unanswerable.** Metrik
   ini hanya bermakna bila ada judgment `correct`. Ia mengukur kebocoran
   parameter, bukan kualitas retrieval.
4. **Sitasi dinilai dengan irisan offset karakter.** `citation_precision` adalah
   proporsi sitasi yang beririsan dengan gold span; `citation_recall` adalah
   proporsi dokumen gold yang tersitasi.
5. **Agregasi per slice dan per `note`.** Query unanswerable dibagi tiga
   subtipe: type, catch rate, egg group. Karena itu gold qrels sekarang membawa
   `note` dari query set.

## Konsekuensi

- Kebenaran jawaban tidak bisa diklaim tanpa judgment. Laporan wajib menulis
  berapa query yang dinilai dan berapa yang `n/a`.
- Tidak ada skor gabungan. Setiap metrik dilaporkan terpisah, sesuai brief.
- Judgment bisa berasal dari manusia atau verifier, asal sumbernya dicatat di
  manifest run.

## Alternatif yang ditolak

- **Exact match terhadap `gold_answer`.** Jawaban bebas akan gagal karena
  perbedaan kalimat, bukan karena salah.
- **Menghitung judgment hilang sebagai salah.** Itu menghukum sistem yang belum
  dinilai dan mendorong orang menghindari penilaian.
- **Satu skor komposit.** Brief menuntut pemisahan penyebab error; skor gabungan
  justru menyembunyikannya.
