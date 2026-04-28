# KRISNABOT Backend Service

KRISNABOT adalah service backend API untuk chat dokumen KRISNA berbasis `FastAPI + PostgreSQL + Gemini File Search`.

Repo ini fokus pada backend, penyimpanan data, ingestion dokumen, dan komunikasi ke Gemini. Frontend berkomunikasi ke service ini lewat HTTP API.

## Arsitektur

- Frontend KRISNA: aplikasi terpisah yang memanggil backend ini lewat HTTP
- Backend API: service FastAPI di repo ini
- Database: PostgreSQL
- Retrieval dokumen: Gemini File Search
- File PDF asli, metadata ingest, chat log, dan audit log disimpan dari backend

## Setup Backend

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Salin `.env.example` menjadi `.env`, lalu isi minimal:

```env
APP_HOST=127.0.0.1
APP_PORT=8000
APP_RELOAD=false
DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/krisnabot
GEMINI_API_KEY=isi_api_key_gemini_anda
MODEL_NAME=gemini-3.1-flash-lite-preview
FILE_SEARCH_STORE=krisnabot-store
FILE_SEARCH_TOP_K=0
FILE_SEARCH_MAX_TOKENS_PER_CHUNK=0
FILE_SEARCH_MAX_OVERLAP_TOKENS=0
FILE_SEARCH_POLL_INTERVAL_SECONDS=2
FILE_SEARCH_OPERATION_TIMEOUT_SECONDS=300
FILE_SEARCH_DOCUMENT_POLL_INTERVAL_SECONDS=2
FILE_SEARCH_DOCUMENT_READY_TIMEOUT_SECONDS=300
CHAT_MAX_OUTPUT_TOKENS=0
CHAT_REQUEST_TIMEOUT_SECONDS=60
CHAT_RETRY_ATTEMPTS=2
CHAT_RETRY_BACKOFF_SECONDS=1
KRISNABOT_SERVICE_KEY=
KRISNABOT_REQUIRE_CHAT_KEY=false
```

`FILE_SEARCH_STORE` bisa diisi display name seperti `krisnabot-store` atau resource name Gemini dengan format `fileSearchStores/...` agar backend tidak perlu lookup store saat cache kosong.

Nilai `0` pada `FILE_SEARCH_TOP_K`, `FILE_SEARCH_MAX_TOKENS_PER_CHUNK`, `FILE_SEARCH_MAX_OVERLAP_TOKENS`, dan `CHAT_MAX_OUTPUT_TOKENS` berarti backend tidak mengirim override sehingga request mengikuti default Gemini.

Saat ingest, file PDF tetap disimpan di database lokal backend sebagai sumber utama. Backend memakai `upload_to_file_search_store` untuk mengirim file langsung ke Gemini File Search Store agar Gemini melakukan chunking, embedding, dan indexing. Data index/embedding di File Search Store tetap ada sampai dokumen index dihapus.

Jika `FILE_SEARCH_MAX_TOKENS_PER_CHUNK` diisi lebih dari `0`, backend mengirim `chunking_config` saat ingest ke Gemini File Search. Jalankan ingest ulang agar perubahan chunking berlaku pada dokumen yang sudah pernah diindeks.

Sesuai lifecycle `Document` di Gemini File Search, backend sekarang menunggu dokumen hasil upload benar-benar berstatus `ACTIVE` sebelum metadata indeks lokal disimpan. `FILE_SEARCH_DOCUMENT_POLL_INTERVAL_SECONDS` dan `FILE_SEARCH_DOCUMENT_READY_TIMEOUT_SECONDS` mengatur polling dan timeout tahap ini.

Untuk request chat, backend juga akan retry singkat saat Gemini mengembalikan error sementara seperti `503 high demand` atau `504 deadline exceeded`. Atur lewat `CHAT_RETRY_ATTEMPTS` dan `CHAT_RETRY_BACKOFF_SECONDS`.

Untuk melihat model Gemini yang bisa dipakai oleh API key Anda:

```bash
.\.venv\Scripts\python scripts\list_gemini_models.py
```

Jika ingin melihat detail action dan limit token:

```bash
.\.venv\Scripts\python scripts\list_gemini_models.py --details
```

Gunakan nama model dari output tersebut untuk `MODEL_NAME`.

Untuk frontend KRISNA lokal, CORS default sudah mengizinkan:

```env
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:9999,http://127.0.0.1:9999
CORS_ORIGIN_REGEX=^http://((localhost|127\.0\.0\.1)(:[0-9]+)?|([a-zA-Z0-9-]+\.)+loc(:[0-9]+)?)$
```

## Komunikasi Service

Login user tetap ditangani aplikasi KRISNA utama. KRISNABOT cukup memakai service key internal agar endpoint AI tidak bisa dipanggil bebas dari internet.

Set `KRISNABOT_SERVICE_KEY` di environment production, lalu kirim header ini dari aplikasi KRISNA saat memanggil KRISNABOT:

```http
X-KRISNABOT-KEY: isi-secret-production
```

Endpoint `/admin/*` memeriksa header tersebut jika `KRISNABOT_SERVICE_KEY` diisi. Pada `APP_ENV=production`, endpoint admin akan menolak request jika service key belum dikonfigurasi.

Untuk `/user/chat`, default-nya belum wajib key agar frontend lama tetap jalan. Jika request chat sudah diproxy lewat backend KRISNA, aktifkan:

```env
KRISNABOT_REQUIRE_CHAT_KEY=true
```

## Menjalankan Service

```bash
.\.venv\Scripts\python scripts\run_backend.py
```

Perintah langsung dengan Uvicorn juga tetap bisa:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Dokumentasi API interaktif tersedia saat service hidup:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

## Deploy Ke VPS / Linux

Untuk production di VPS, jalankan backend sebagai proses tunggal di belakang reverse proxy seperti Nginx.

Rekomendasi environment:

```env
APP_ENV=production
APP_RELOAD=false
APP_HOST=127.0.0.1
APP_PORT=8000
KRISNABOT_SERVICE_KEY=isi-secret-yang-kuat
KRISNABOT_REQUIRE_CHAT_KEY=true
CORS_ORIGINS=https://domain-frontend-anda
CORS_ORIGIN_REGEX=
```

Catatan deploy penting:

- Gunakan `APP_HOST=127.0.0.1` jika backend hanya diakses lewat Nginx pada server yang sama.
- Gunakan `APP_HOST=0.0.0.0` hanya jika memang perlu membuka port backend ke jaringan luar.
- Jangan menjalankan banyak worker app untuk endpoint ingest jika Anda mengandalkan status ingest dari memori proses. `ingest_job_manager` menyimpan state di memory proses saat ini.
- Pastikan PostgreSQL, `GEMINI_API_KEY`, dan `KRISNABOT_SERVICE_KEY` sudah tersedia sebelum service dijalankan.

Contoh menjalankan service di Linux:

```bash
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Contoh file `systemd` dan `nginx` ada di folder `deploy/`.

## Konfigurasi Frontend KRISNA

Frontend membaca `krisnabot_api`. Contoh:

```php
'krisnabot_api' => [
    'base_url' => 'http://127.0.0.1:8000',
    'with_credentials' => false,
],
```

Jika endpoint khusus tidak diisi, frontend akan memakai path default backend:

- `GET /health`
- `POST /user/chat`
- `GET /admin/status`
- `GET /admin/ingest/status`
- `POST /admin/ingest`
- `POST /admin/upload`
- `POST /admin/replace`
- `DELETE /admin/docs`
- `DELETE /admin/index`

## Kontrak API Utama

`POST /user/chat`

```json
{
  "question": "Bagaimana cara menggunakan KRISNA?"
}
```

Response:

```json
{
  "answer": "Jawaban dari dokumen jika ditemukan.",
  "message": "",
  "found": true,
  "used_files": ["Manual.pdf"],
  "grounding_chunks": [
    {
      "source_file": "Manual.pdf",
      "text": "Potongan teks dokumen yang menjadi rujukan jawaban."
    }
  ],
  "error": ""
}
```

Jika jawaban tidak ditemukan, `answer` kosong dan pesan fallback dikirim melalui `message`.

## Alur Dokumen

1. Admin upload PDF melalui UI KRISNA atau endpoint admin.
2. File mentah masuk ke tabel `documents`.
3. Saat ingest dijalankan, backend mengunggah file langsung ke Gemini File Search Store.
4. Gemini File Search melakukan chunking, embedding, dan indexing.
5. Backend menunggu `Document` Gemini selesai diproses dan berstatus `ACTIVE`.
6. Metadata remote disimpan di tabel `ingested_documents`.
7. Saat user bertanya, backend memanggil Gemini `generate_content` dengan tool `file_search`.

## Testing

```bash
.\.venv\Scripts\python -m unittest discover -s tests
```
