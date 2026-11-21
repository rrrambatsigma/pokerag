# ADR-0011 — Kontrak serving: FastAPI di atas retriever yang ada

**Status:** diterima
**Tanggal:** 2026-11-22

## Konteks

Retrieval hybrid sudah jalan dan UI chat sudah ada, tetapi masih memakai
`answerFor` dari `frontend/src/mock/data.ts`. Folder `frontend/src/api` kosong.
Yang dibutuhkan adalah kontrak HTTP yang bisa dipakai UI sekarang, tanpa
menunggu generator Qwen selesai.

Ada tiga hal yang harus diputuskan: bentuk payload, sumber metadata sitasi, dan
cara menjalankan API saat model belum ada.

## Keputusan

1. **FastAPI, satu proses, path `/api`.** Proxy Vite sudah ada di
   `frontend/vite.config.ts` (`/api` → `127.0.0.1:8000`), jadi tidak perlu CORS
   dan tidak perlu mengubah frontend.
2. **Field JSON memakai camelCase** persis seperti `frontend/src/types.ts`
   (`chunkId`, `docId`, `headingPath`, `topK`). Tidak ada lapisan pemetaan di UI.
3. **`off` adalah mode yang sah.** `dense=false` dan `bm25=false` berarti tanpa
   retrieval, yaitu posisi B0. Generator tetap dipanggil dengan evidence kosong.
4. **Metadata sitasi diambil dari `documents.parquet`**, bukan dari payload
   Qdrant: dex dan tipe ada di `infobox_json["0"]`. Parser toleran; baris yang
   hilang menghasilkan field kosong, bukan error.
5. **Stub generator di belakang protokol `Generator`.** API, tes, dan UI bisa
   berjalan tanpa GPU. Saat backend Qwen siap, hanya implementasi yang diganti.

## Konsekuensi

- `Retriever` dibangun lazy per strategi dan menahan lock Qdrant local mode.
  Satu proses API saja; eval tidak boleh membuka database yang sama bersamaan.
- `documents.parquet` dibaca saat sitasi pertama, bukan saat startup, supaya
  `/api/health` tetap murah.
- Generator asli wajib mengembalikan `cited_chunk_ids`; API hanya menyitir chunk
  yang benar-benar diambil, sehingga sitasi tidak bisa menunjuk dokumen yang
  tidak ada di konteks.
- Angka di laporan harus memakai konfigurasi yang sama dengan yang di-serve.
  Demo tidak boleh diam-diam memakai konfigurasi lain.

## Alternatif yang ditolak

- **GraphQL.** Tidak ada kebutuhan query kompleks; satu endpoint chat dan satu
  endpoint search cukup.
- **CORS.** Proxy Vite sudah menangani origin; menambah CORS hanya menambah
  permukaan yang tidak dipakai.
- **Menunggu generator selesai sebelum membuat API.** Itu menahan pekerjaan UI
  dan membuat skema jawaban ditemukan terlambat.
