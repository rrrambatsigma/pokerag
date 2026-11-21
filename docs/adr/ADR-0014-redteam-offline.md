# ADR-0014 — Red-team suite offline: fixture, bukan model

**Status:** diterima
**Tanggal:** 2026-11-22

## Konteks

Brief meminta minimal 20 kasus red-team, lima per kategori: prompt atau document
injection, policy atau PII leakage, tool misuse, dan safety atau robustness.
Tiap kasus wajib punya identitas, tujuan serangan, permukaan serangan, hasil
baseline, dan hasil setelah mitigasi.

Generator belum ada, jadi suite tidak bisa dijalankan langsung. Tetapi fixture
dan cara menilainya bisa disiapkan sekarang, dan itu justru mengunci apa yang
dianggap "lulus" sebelum model hadir.

## Keputusan

1. **Dua puluh fixture di `data/redteam/fixtures/suite.jsonl`**, lima per
   kategori, dengan id stabil.
2. **Penilaian offline.** Runner menerima berkas respons `{id, text}`, bukan
   memanggil model. Baseline dan mitigation dinilai oleh checker yang sama.
3. **Tiga pemeriksaan otomatis:** pola terlarang (regex), penanda penolakan
   untuk kasus yang mengharuskan menolak, dan batas panjang untuk input ekstrem.
4. **Fixture injection membawa `injected_evidence`.** Saat model tersedia,
   runner menempelkannya ke evidence retrieval; untuk penilaian offline, field
   ini hanya dokumentasi serangan.
5. **Klaim tool tidak diverifikasi dari trace.** Tanpa tool trace, yang bisa
   diperiksa hanya apakah jawaban mengklaim telah melakukan tindakan berbahaya.
   ADR ini mencatat batas itu agar tidak diklaim lebih.

## Konsekuensi

- Penanda penolakan bersifat heuristik. Penolakan yang halus bisa lolos, dan
  jawaban aman yang kebetulan memuat kata "cannot" bisa dianggap menolak.
  Karena itu hasil harus dibaca, bukan hanya dihitung.
- PII diuji sebagai perilaku menolak, bukan kebocoran nyata: korpus memang tidak
  memuat PII. Yang dibuktikan adalah sistem tidak mengarang alamat email atau IP.
- Kategori tool misuse baru bermakna penuh setelah B6 punya tool nyata dan
  trace-nya dicatat. Sebelum itu, hasilnya adalah proxy.
- Suite ini tidak menggantikan penilaian manusia pada kasus yang ambigu.

## Alternatif yang ditolak

- **Menjalankan model sekarang.** Tidak perlu; fixture dan checker tidak
  bergantung pada model.
- **Menilai dengan LLM judge saja.** Tidak deterministik dan sulit direproduksi
  untuk 20 kasus yang polanya jelas.
- **Menunggu B6 sebelum menulis fixture.** Itu menunda pekerjaan yang bisa
  diselesaikan sekarang dan membuat cakupan serangan ditentukan setelah melihat
  sistem, bukan sebelum.
