# PokéRAG — Frontend

Antarmuka chat bergaya panel GBA/Pokédex untuk tahap retrieval. Saat ini UI
berjalan dengan **data mock**; belum tersambung ke API retrieval.

## Menjalankan

```bash
cd frontend
npm install
npm run dev
```

Buka `http://localhost:5173`.

Build produksi:

```bash
npm run build      # tsc --noEmit && vite build
npm run preview
```

## Yang sudah jalan

- Chat dengan jawaban dan sitasi bertipe Pokédex (nomor dex, tipe, section, skor).
- Panel sumber: klik badge sitasi atau kartu sumber untuk membuka entri lengkap
  beserta tautan ke Bulbapedia.
- Riwayat percakapan dengan pencarian? Tidak — hanya daftar, pilih, hapus, dan
  chat baru. Disimpan di `localStorage` (`pokerag.ui.v1`).
- Setelan: toggle pencarian makna (dense) dan kata kunci (BM25), strategi
  chunking, jumlah sumber. Mode turunan: `HYBRID`, `MAKNA`, `KATA KUNCI`,
  `MATI`.
- Setelan yang dipakai disimpan pada tiap jawaban asisten, sehingga asal sebuah
  jawaban bisa ditelusuri.
- Responsif: sidebar riwayat menjadi drawer di layar kecil.

## Yang masih mock

- Jawaban dan sitasi berasal dari `src/mock/data.ts` (Charizard, Eevee, Zapdos,
  Venusaur). Pertanyaan lain mengembalikan pesan bahwa topik itu belum tersedia.
- Belum ada panggilan jaringan. Kontrak yang dituju:

```
POST /api/search
{ "query": "...", "dense": true, "bm25": true, "strategy": "heading", "top_k": 5 }

200 { "mode": "hybrid", "results": [Citation, ...] }
```

Saat API siap, ganti `answerFor()` di `src/hooks/useChatStore.ts` dengan
pemanggilan `fetch('/api/search')`; tipe `Citation` di `src/types.ts` sudah
mencerminkan payload yang dibutuhkan (judul, dex, tipe, heading, kutipan, URL,
skor).

## Struktur

```
src/
  App.tsx                 layout, tab, modal
  components/             panel, chat, sitasi, setelan, komposer, primitives
  hooks/useChatStore.ts   state percakapan + localStorage + jawaban mock
  lib/                    mode, format tanggal, penyimpanan, helper percakapan
  mock/data.ts            snapshot, sitasi contoh, seed riwayat
  styles.css              token warna, panel pixel, responsif
```

## Catatan

- Tidak ada sprite, logo, atau aset gambar Pokémon. Aksen dibuat dari tipe,
  nomor dex, dan gaya panel.
- Teks korpus berlisensi CC BY-NC-SA 2.5; atribusi ditampilkan di sidebar dan
  panel sumber.
