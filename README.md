# KRISNABOT Backend Service

KRISNABOT adalah service backend API untuk chat dokumen KRISNA berbasis `FastAPI + PostgreSQL + Gemini File Search`.

Frontend aktif berada di `D:\KRISNA\krisna`. Repo ini hanya menjalankan backend, penyimpanan data, ingestion dokumen, dan komunikasi ke Gemini. Frontend berkomunikasi ke service ini lewat HTTP API.

## Arsitektur

- Frontend KRISNA: `D:\KRISNA\krisna`
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
MODEL_NAME=gemini-2.5-flash-lite
FILE_SEARCH_MODEL_NAME=gemini-2.5-flash-lite
FILE_SEARCH_FALLBACK_MODEL_NAME=gemini-2.5-flash
FILE_SEARCH_STORE_DISPLAY_NAME=krisnabot-store
```

Untuk melihat model Gemini yang bisa dipakai oleh API key Anda:

```bash
.\.venv\Scripts\python scripts\list_gemini_models.py
```

Jika ingin melihat detail action dan limit token:

```bash
.\.venv\Scripts\python scripts\list_gemini_models.py --details
```

Gunakan nama model dari output tersebut untuk `MODEL_NAME`, `FILE_SEARCH_MODEL_NAME`, atau `FILE_SEARCH_FALLBACK_MODEL_NAME`.

Untuk frontend KRISNA lokal, CORS default sudah mengizinkan:

```env
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:9999,http://127.0.0.1:9999
CORS_ORIGIN_REGEX=^http://((localhost|127\.0\.0\.1)(:[0-9]+)?|([a-zA-Z0-9-]+\.)+loc(:[0-9]+)?)$
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

## Konfigurasi Frontend KRISNA

Di `D:\KRISNA\krisna`, konfigurasi frontend membaca `krisnabot_api`. Contoh:

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
  "error": ""
}
```

Jika jawaban tidak ditemukan, `answer` kosong dan pesan fallback dikirim melalui `message`.

## Alur Dokumen

1. Admin upload PDF melalui UI KRISNA atau endpoint admin.
2. File mentah masuk ke tabel `documents`.
3. Saat ingest dijalankan, backend mengunggah file ke Gemini File Search store.
4. Metadata remote disimpan di tabel `ingested_documents`.
5. Saat user bertanya, backend memanggil Gemini `generate_content` dengan tool `file_search`.

## Testing

```bash
.\.venv\Scripts\python -m unittest discover -s tests
```
