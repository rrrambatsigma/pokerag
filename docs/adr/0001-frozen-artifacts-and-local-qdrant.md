# ADR 0001: impor artefak beku dan Qdrant local mode

- Status: diterima
- Tanggal: 21 November 2026
- Konteks: M0

## Keputusan

Retrieval memakai artefak yang sudah dibangun di `meivawijaya/pokerag-corpus-v3`,
bukan membangun ulang korpus, chunk, embedding, dan index di repo ini. Vector
store dibuka dengan `QdrantClient(path=...)` (local mode), bukan Qdrant server.

## Alasan

1. Artefak sudah membawa `snapshot_id`, `index_version`, dan `INTEGRITY.sha256`.
   Membangun ulang berisiko menghasilkan index yang tidak identik dengan angka
   yang dilaporkan di `reports/A1-index.md`.
2. Embedding `BAAI/bge-m3` membutuhkan GPU dan waktu. Angka A1 yang sudah ada
   hanya sah kalau index tidak berubah.
3. Qdrant local mode adalah format asli artefak. Server memerlukan rebuild
   storage dan menambah komponen operasional tanpa manfaat pada 68.330 chunk.
4. Exact search pada 28.786 vektor cukup cepat dan lebih mudah direproduksi
   daripada HNSW.

## Konsekuensi

- Database tidak boleh dibuka dua proses sekaligus karena ada lock.
- Versi `qdrant-client` harus dipin. Schema local mode bisa berubah antar versi.
- Kalau versi tidak lagi kompatibel, index bisa dibangun ulang dari cache
  embedding (91.727 vektor) tanpa memanggil model lagi.
- Untuk serving dengan banyak pengguna, `VectorStore` bisa diganti ke server
  tanpa mengubah pipeline.

## Alternatif yang ditolak

- Rebuild penuh di repo ini: lambat, butuh GPU, dan mengubah artefak acuan.
- Qdrant server sejak awal: menambah docker dan volume, tanpa kebutuhan nyata
  pada tahap eksperimen.
