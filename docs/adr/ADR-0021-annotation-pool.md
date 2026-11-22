# ADR-0021 — Pool anotasi A2 dengan unit chunk heading

**Status:** diterima
**Tanggal:** 2026-11-23

## Konteks

A2 membandingkan dense, BM25, hybrid, dan hybrid+reranking pada qrels yang sama,
lalu melaporkan Recall@5/10, nDCG@10, dan MRR@10. Supaya perbandingan sah,
setiap metode harus menyumbang top-10-nya ke satu pool yang dibekukan sebelum
anotasi.

Spek menetapkan B2 memakai *fixed chunk*, dan tangga B2-B5 di
`configs/ladder.yaml` memakai `strategy: fixed`. Di sisi lain, A1 secara
eksplisit membandingkan tiga strategi chunking dengan qrels identik. Artinya
unit qrels tidak ditetapkan oleh spek; yang ditetapkan adalah B2 memakai fixed,
bukan bahwa anotasi harus dilakukan pada chunk fixed.

## Keputusan

1. **Pool dibangun dari chunk `heading`**, top-10 dari empat metode, dedup per
   `chunk_id`, urutan diacak dengan seed tetap, lalu dibekukan dengan hash.
2. **Kedalaman 10 per metode**, karena metrik yang dilaporkan juga @10.
3. **Label bertingkat 0/1/2** per passage.
4. **Dua anotator menilai semua passage**, satu adjudikator memutus label yang
   berbeda atau belum dinilai. Nama dipilih saat ekspor.
5. **Provenance metode disimpan terpisah** di `pool_sources.jsonl` dan tidak
   pernah dikirim ke alat anotasi.
6. **A1 memakai proyeksi span.** Label heading diproyeksikan ke `fixed` dan
   `hierarchical` lewat irisan rentang karakter.

## Alasan memilih heading

Diukur pada 27.902 unit heading, 11.642 unit fixed, dan 28.786 child
hierarchical:

| | fixed | heading | hierarchical |
| --- | --- | --- | --- |
| Median karakter | 2.048 | 1.109 | 600 |
| Mulai di tengah kalimat | 94,4% | **62,5%** | 85,1% |
| Mulai dengan pronomina | 1,8% | 3,9% | 1,9% |

1. **Batas paling bersih.** Chunk heading paling sering mulai di batas section.
   Chunk fixed dan child hierarchical adalah jendela yang memotong kalimat dan
   konteks.
2. **Paling sering berdiri sendiri.** Median 1.109 karakter cukup memuat satu
   fakta beserta subjeknya, sehingga label 2 tidak ambigu. Pada chunk 600
   karakter, fakta sering terpisah dari konteksnya dan label 1 menjadi kabur.
3. **A2 tetap adil.** Keempat metode dijalankan pada index heading yang sama,
   jadi qrels benar-benar identik dan perbandingan tidak mencampur chunking.
4. **Tidak menambah beban.** Jumlah unit heading (27.902) hampir sama dengan
   child hierarchical (28.786), tetapi tiap passage lebih mudah dipahami.

Fixed ditolak untuk anotasi karena 94,4% chunk mulai di tengah kalimat dan
panjangnya 2.048 karakter, sehingga membaca satu label jauh lebih berat.
Hierarchical ditolak karena child-nya paling terfragmentasi dan keunggulan
parent context belum dipakai pipeline generation.

## Penyimpangan yang harus dipertahankan di laporan

Benchmark memakai heading, sedangkan B2-B5 tetap memakai fixed sesuai spek.
Konsekuensinya:

- Metrik retrieval A2 dihitung pada unit heading.
- Metrik generasi B2-B5 memakai sitasi berbasis span karakter, jadi perbedaan
  unit chunk tidak merusak pencocokan sitasi.
- Untuk A1, label heading diproyeksikan ke fixed dan hierarchical lewat irisan
  span, dan proyeksi ini lossy. Batasan harus disebut.
- Kalau A1 menunjukkan heading lebih baik, ADR-0005 sudah menetapkan jalannya:
  B2-B6 dijalankan ulang dengan heading dan dilaporkan sebagai sistem v2.

## Angka

| Metrik | Nilai |
| --- | --- |
| Query | 100 |
| Passage unik | 2.040 |
| Passage per query | min 15, rata-rata 20,4, maks 25 |
| Sumbangan eksklusif BM25 | 598 passage (29 persen pool) |
| Sumbangan eksklusif rerank | 236 |
| Sumbangan eksklusif dense | 84 |
| Sumbangan eksklusif hybrid | 26 |
| Hash pool | `d8ae071a36de…` |

Pool heading juga menyelaraskan unit anotasi dengan unit serving, karena
`configs/retrieval.serve.yaml` memakai heading.

## Alternatif yang ditolak

- **Pool fixed** karena tangga B memakainya. Lebih sesuai huruf spek, tetapi
  passage 2.048 karakter dan hampir semua mulai di tengah kalimat.
- **Pool hierarchical** karena parent context. Paling terfragmentasi, dan
  parent belum dipakai saat generation.
- **Kedalaman 5** untuk mengurangi beban. Tidak sah untuk nDCG@10.
- **Tiga anotator menilai semuanya.** Memenuhi minimal dua, tetapi menambah
  beban 50 persen dibanding dua penilai plus adjudikator.
