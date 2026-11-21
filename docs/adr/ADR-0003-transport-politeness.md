# ADR-0003 — Transport `curl_cffi` dan kepatuhan crawl-delay

**Status:** diterima
**Tanggal:** 2026-11-14

## Konteks

Diuji langsung pada 2026-11-14:

* `requests` dan `Invoke-WebRequest` mendapat **HTTP 403** dari CDN Bulbapedia,
  bahkan dengan User-Agent Chrome.
* Klien dengan fingerprint TLS browser (`curl_cffi`, `impersonate="chrome"`)
  mendapat **200**.
* `robots.txt` mengizinkan `/wiki/`, melarang `action=history` dan `oldid=`,
  dan menetapkan **`Crawl-delay: 5`**.
* `meta=siteinfo&siprop=rightsinfo` mengembalikan `CC BY-NC-SA 2.5`.

## Keputusan

* Transport: `curl_cffi` dengan UA akademik yang menyebut kontak, bukan UA palsu.
* Jeda minimum 5 detik antar request, diukur dari **selesainya** request sebelumnya
  (lebih konservatif daripada start-to-start).
* Semua request dicatat di `fetch_log.jsonl` (status, bytes, ms, `gap_seconds`),
  dan gate validasi memeriksa gap minimum.
* HTTP 403 tidak di-retry dan tidak diakali. Run berhenti dengan pesan jelas.
* Batching 5 judul per request untuk metadata; HTML tetap satu request per halaman.

## Konsekuensi

* Snapshot 320 halaman ≈ 32 menit. Ini harga kepatuhan; tidak dipercepat.
* `raw/` wajib disimpan agar tidak perlu crawl ulang saat aturan cleaning berubah.
* Jika situs mengubah kebijakan atau mengeluarkan challenge, langkah berikutnya
  adalah menghubungi pemilik situs, bukan mencari bypass.
