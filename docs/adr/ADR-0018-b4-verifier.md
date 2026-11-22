# ADR-0018 — B4: verifier dengan perbaikan terbatas

**Status:** diterima
**Tanggal:** 2026-11-22

## Konteks

B3 sudah memperbaiki retrieval q0005: kedua baris egg group ada di rank 1 dan 2.
Namun B3 tetap menjawab "No". q0009 bahkan sudah punya kedua bukti sejak B2 dan
tetap dijawab "No". Jadi kegagalannya ada di generation, bukan retrieval.

B4 menurut brief menambahkan query planner, skor kecukupan evidence, retry
terbatas, verifier, dan abstain. Yang harus dibuktikan: komponen mana yang
benar-benar memperbaiki dua kasus itu.

## Keputusan

1. **Verifier memakai model yang sama dengan prompt berbeda.** Ia menilai apakah
   jawaban didukung evidence, dengan aturan khusus: pertanyaan ya/tidak diperiksa
   per sisi, dan abstain yang jujur dianggap supported.
2. **Satu kali perbaikan, lalu abstain.** Jika verifier menolak, generator
   menerima alasan verifier dan menjawab ulang. Kalau masih ditolak, sistem
   mengembalikan "Not enough evidence in the corpus." daripada mengirim klaim
   tanpa dukungan.
3. **Planner berbasis aturan untuk template perbandingan.** "Are X and Y both in
   the G egg group?" dipecah menjadi dua sub-query retrieval. Ini deterministik
   dan berjalan sebelum embedder dimuat, jadi tidak melanggar anggaran VRAM.
4. **Trace mencatat `verified`, `repairs`, dan token verifier.** Biaya verifikasi
   tidak boleh hilang dari manifest.

## Bukti

| Konfigurasi | q0005 | q0009 | Catatan |
| --- | --- | --- | --- |
| B3 (planner off, verifier off) | No (salah) | No (salah) | bukti sudah cukup |
| B4 planner saja (`--no-verifier`) | No (salah) | No (salah) | retrieval per entitas tidak memperbaiki penalaran |
| B4 verifier saja (`--no-planner`) | **Yes** | **Yes** | masing-masing 1 repair |
| B4 penuh | Yes | Yes | sama dengan verifier saja |

Kesimpulan: pada dua kasus ini, **verifier adalah komponen yang aktif**, bukan
planner.

## Konsekuensi

- Biaya naik: satu panggilan verifier per query, plus satu panggilan perbaikan
  bila ditolak. Latensi B4 sembilan query: p50 3,35 detik, p95 6,40 detik.
- Verifier memakai model yang sama, jadi ia bisa ikut salah. Pada dua kasus ini
  ia berhasil, tetapi itu bukan jaminan. A3 harus melaporkan false accept dan
  false reject, bukan hanya kasus yang berhasil.
- Planner tidak memperbaiki penalaran, tetapi tetap berguna untuk memastikan
  evidence kedua entitas masuk. Efeknya harus diukur di A6, bukan diasumsikan.

## Alternatif yang ditolak

- **Verifier model terpisah.** Lebih independen, tetapi menambah VRAM dan latensi;
  belum perlu selama generator masih 1,5B.
- **Perbaikan tanpa batas.** Biaya tidak terkendali dan bisa berputar.
- **Planner LLM sebelum retrieval.** Memaksa generator dimuat sebelum embedder,
  yang tidak muat di 6 GB tanpa membongkar dua-tahap.
