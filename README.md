# RAG Helpdesk API

API chatbot IT Helpdesk berbasis FastAPI. Aplikasi mengambil dokumen SOP lokal
dan knowledge tiket resolved dari Laravel, lalu menggunakan embedding Qwen,
opsional reranker Qwen, dan Gemini untuk menjawab pertanyaan.

## Persyaratan

- Python 3.10 atau lebih baru
- API key Google Gemini
- Koneksi internet saat pertama kali menjalankan aplikasi untuk mengunduh model
- (Opsional) API Laravel yang menyediakan endpoint knowledge dan webhook

## Instalasi

Jalankan perintah berikut dari direktori proyek:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Di Windows PowerShell, aktifkan virtual environment dengan:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Konfigurasi

Buat file `.env` dari contoh konfigurasi. Perintah ini tidak menimpa `.env`
yang sudah ada:

```bash
if [ ! -f .env ]; then cp .env.example .env; fi
```

Di Windows PowerShell:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env`, setidaknya isi `GOOGLE_API_KEY` dengan API key Gemini yang valid.
Alternatifnya, aplikasi juga menerima `GEMINI_API_KEY`.

| Variabel | Keterangan |
| --- | --- |
| `GOOGLE_API_KEY` atau `GEMINI_API_KEY` | Wajib. API key Google Gemini. |
| `GEMINI_MODEL` | Model utama Gemini. Default kode: `gemini-3.6-flash`. |
| `GEMINI_FALLBACK_MODELS` | Daftar model cadangan, dipisahkan koma. |
| `LARAVEL_BASE_URL` | URL dasar integrasi Laravel. Default kode: `http://127.0.0.1:8000/api/rag/integrations`. |
| `RAG_USERNAME`, `RAG_PASSWORD` | Kredensial Basic Auth Laravel. Auth hanya dikirim jika password diisi. |
| `DOCS_DIR` | Folder dokumen SOP. Default: `./Raw RAG Document`. |
| `EMBEDDING_MODEL`, `RERANKER_MODEL` | Model embedding dan reranker Hugging Face. |
| `RAG_DEVICE` | Perangkat model, misalnya `cpu` atau `cuda`. Default: `cpu`. |
| `RAG_ENABLE_RERANKER` | Aktifkan reranker dengan `true` atau nonaktifkan dengan `false`. |
| `CORS_ORIGINS` | Daftar origin frontend yang dipisahkan koma. |
| `HOST`, `PORT`, `RELOAD` | Opsi saat aplikasi dijalankan menggunakan `python api.py`. |

Nilai di `.env` akan mengesampingkan default di kode. Periksa kembali
`LARAVEL_BASE_URL` dan kredensial sesuai konfigurasi API Laravel Anda.
Jangan commit file `.env` atau membagikan API key.

## Dokumen SOP

Letakkan file PDF atau TXT di folder `Raw RAG Document` (atau folder yang
ditentukan oleh `DOCS_DIR`). Subfolder juga dipindai. Format DOCX tidak dimuat
oleh loader saat ini.

Saat aplikasi mulai, indeks vektor dibuat ulang di memori. Aplikasi juga
mencoba mengambil knowledge dari Laravel. Jika Laravel tidak tersedia, aplikasi
dapat berjalan menggunakan SOP lokal selama setidaknya satu dokumen SOP berhasil
dimuat. Pastikan folder SOP tersedia sebelum memulai jika API Laravel tidak
digunakan.

## Menjalankan API

Dengan virtual environment aktif:

```bash
uvicorn api:app --host 127.0.0.1 --port 8001 --reload
```

Atau jalankan langsung:

```bash
python api.py
```

Inisialisasi pertama dapat memerlukan waktu karena model embedding dan reranker
diunduh dan indeks dokumen dibangun. Jika berhasil, API tersedia di
`http://127.0.0.1:8001`.

- Dokumentasi interaktif: <http://127.0.0.1:8001/docs>
- Status kesiapan: <http://127.0.0.1:8001/api/health>

## Endpoint

| Method | Path | Keterangan |
| --- | --- | --- |
| `GET` | `/api/health` | Status kesiapan index dan model. |
| `POST` | `/api/analyze-issue` | Analisis pesan, klasifikasi kendala, dan pencarian SOP. |
| `POST` | `/api/chat` | Chat berdasarkan SOP dan knowledge tiket. |

Contoh analisis kendala:

```bash
curl -X POST http://127.0.0.1:8001/api/analyze-issue \
  -H "Content-Type: application/json" \
  -d '{
    "issue": "Printer kasir tidak bisa mencetak struk",
    "branch_name": "Cabang Jakarta"
  }'
```

Contoh chat:

```bash
curl -X POST http://127.0.0.1:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Apa yang harus diperiksa jika printer tidak mencetak?",
    "history": []
  }'
```

Untuk menghubungkan chat dengan tiket Laravel, tambahkan `"ticket_id": 123` pada
request `/api/chat`. API akan mencoba mengirim hasil atau status eskalasi ke
endpoint webhook Laravel.

## Troubleshooting

- **Aplikasi berhenti saat startup:** pastikan API key Gemini terisi dan format
  nilai konfigurasi di `.env` benar.
- **Tidak ada dokumen RAG:** tambahkan PDF/TXT ke folder SOP atau pastikan
  endpoint Laravel knowledge mengembalikan data.
- **Model gagal diunduh:** periksa koneksi internet dan ruang disk; model
  diunduh oleh Hugging Face saat pertama kali digunakan.
- **Status health `initializing`:** tunggu sampai proses pemuatan knowledge,
  embedding, dan model selesai.
- **Tidak bisa diakses dari frontend:** tambahkan origin frontend ke
  `CORS_ORIGINS`, lalu jalankan ulang API.
