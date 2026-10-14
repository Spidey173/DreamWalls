# DreamWalls 🎨

**DreamWalls** is an AI-powered interior and exterior architectural visualization web application developed for paint shops. When a customer uploads a photo of their building or room, the app analyzes the architecture, lighting, and existing color palette, recommends the 4 best matching **Asian Paints** shades/schemes using color harmony algorithms, and displays the walls realistically repainted in each shade with Asian Paints shade names and codes.

---

## 🌟 Key Features

1. **Configurable Multi-Provider LLM Vision Engine**:
   - Seamlessly switch between **Gemini** (`gemini-2.5-flash`), **Anthropic Claude** (`claude-3-5-sonnet`), and **Groq** (`llama-3.2-vision`) via `.env`.
   - Vision analysis extracts strict JSON: `room_type`, `lighting` (bright/dim/mixed), `current_wall_color_hex`, `floor_color_hex`, `dominant_furniture_colors`, `style`, and `mood`.
   - Robust Pydantic validation with automatic 1-attempt retry.

2. **Deterministic Color Harmony Shortlisting (in Code)**:
   - Uses **CIE-L\*a\*b\*** and **HSL** color science to filter the catalog down to 30–40 candidates.
   - Categorizes harmony into **Neutral**, **Analogous**, and **Complementary**.
   - **Lighting-Aware Filtering**: Dim rooms filter out low-light reflectance paints and favor high-luminance tones ($L^* \ge 68$); bright rooms accommodate deeper accents.

3. **Anti-Hallucination AI Selection Gate**:
   - The LLM selects the top 4 shades solely from the candidate IDs.
   - Every returned ID is strictly validated against the database. Invalid/hallucinated IDs are dropped and replenished from the top-scoring shortlist. AI cannot invent shade names or codes.

4. **SegFormer (ADE20K) ONNX & MobileSAM Detection**:
   - Uses pre-trained **SegFormer B2** ONNX model by default (with **B0** available via `SEGFORMER_MODEL_VARIANT=b0`) with bilinear logit upsampling and guided filter edge refinement.
   - Supports interactive **MobileSAM tap-to-select** with multi-point positive and negative prompts.
   - Masks are automatically cached per image hash (SHA-256) so segmentation is computed once per upload.

5. **Photorealistic LAB Space Recoloring**:
   - Swaps $a$ and $b$ chrominance channels with the target shade.
   - Shifts luminance ($L$) toward the target paint while preserving original ambient lighting, shadows, and wall textures.
   - Feathers mask boundaries using Gaussian edge filtering for natural transitions at trim and ceilings.

6. **Interactive "Fix Wall Area" Canvas Tool**:
   - Customers can tap/brush on their photo with an interactive canvas brush or eraser to correct the wall mask, instantly repainting all 4 shades without re-calling the LLM.

7. **Interactive Before/After Split Comparison Slider**:
   - Mobile-first interactive split slider comparing original room photo directly with each repainted shade.

8. **Admin Catalog & Bulk CSV Import**:
   - Password-protected `/admin` portal.
   - Full CRUD for paints (name, shade code, hex, finish, color family, price).
   - Bulk CSV import for fast store catalog updates.

9. **WhatsApp Integration & Legal Disclaimers**:
   - One-tap WhatsApp inquiry button with pre-filled shade card inquiries for the local paint shop owner.
   - Prominent disclaimer: *"Actual color may vary on the wall. Visit the shop for a shade card."*

---

## 🛠️ Tech Stack

- **Backend**: Python 3.12, FastAPI, SQLAlchemy, Alembic, PostgreSQL / SQLite, OpenCV Headless, ONNX Runtime, Pydantic V2.
- **Frontend**: Next.js 15+ (App Router), TypeScript, Tailwind CSS, Lucide Icons.
- **DevOps**: Docker & Docker Compose (`db`, `api`, `web`).

---

## 📁 Project Structure

```text
rtegrtdb/
├── docker-compose.yml
├── .env.example
├── README.md
├── output_samples/                 # Saved CLI output samples
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── data/
│   │   └── paints.csv              # ~35 Asian Paints placeholder shades
│   ├── models/
│   │   ├── segformer_b2_ade20k.onnx # Default SegFormer B2 ADE20K ONNX model (~105 MB)
│   │   ├── segformer_b0_ade20k.onnx # Optional lightweight B0 model (~15 MB)
│   │   ├── mobile_sam_encoder.onnx  # MobileSAM image encoder ONNX (~11 MB)
│   │   └── mobile_sam_decoder.onnx  # MobileSAM prompt decoder ONNX (~16 MB)
│   ├── app/
│   │   ├── main.py
│   │   ├── core/                   # config, database, rate_limiter, security
│   │   ├── models/                 # SQLAlchemy models (Paint)
│   │   ├── schemas/                # Pydantic schemas (vision, paint, selection, recolor)
│   │   ├── services/
│   │   │   ├── llm/                # BaseLLMClient, Gemini, Anthropic, Groq, Mock
│   │   │   ├── vision_service.py
│   │   │   ├── shortlist_service.py
│   │   │   ├── selection_service.py
│   │   │   ├── segmentation_service.py
│   │   │   └── recolor_service.py
│   │   └── api/v1/endpoints/       # analyze, recolor, admin, health
│   ├── scripts/
│   │   ├── seed_db.py              # Seeds database from CSV
│   │   ├── download_model.py       # Downloads SegFormer ONNX model
│   │   └── cli_test_pipeline.py    # End-to-end pipeline CLI tester
│   └── tests/
│       ├── test_shortlist.py
│       ├── test_selection_validation.py
│       ├── test_recolor.py
│       └── test_api.py
│
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx            # Drag & drop upload page + progress stepper
    │   │   ├── results/page.tsx    # Before/After slider + 4 cards + WhatsApp CTA
    │   │   └── admin/page.tsx      # Password-protected admin catalog manager
    │   ├── components/
    │   │   ├── Navbar.tsx
    │   │   ├── BeforeAfterSlider.tsx
    │   │   ├── ShadeCard.tsx
    │   │   ├── MaskEditorModal.tsx # Interactive canvas mask repair tool
    │   │   └── ProgressSteps.tsx
    │   └── lib/                    # api client and TypeScript types
```

---

## ⚙️ Environment Configuration

Create a `.env` file at the root of the project:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM service provider: `gemini`, `anthropic`, or `groq` | `gemini` |
| `GEMINI_API_KEY` | Google Gemini API key | `""` *(falls back to MockLLMClient if unset)* |
| `GEMINI_MODEL` | Gemini vision model | `gemini-2.5-flash` |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | `""` |
| `ANTHROPIC_MODEL` | Claude vision model | `claude-3-5-sonnet-20241022` |
| `GROQ_API_KEY` | Groq API key | `""` |
| `GROQ_MODEL` | Groq vision model | `llama-3.2-90b-vision-preview` |
| `DATABASE_URL` | SQLAlchemy database connection string | `sqlite:///./paintmywall.db` |
| `ADMIN_PASSWORD` | Password for accessing the `/admin` portal | `paintshop_admin_secret_2026` |
| `CORS_ORIGINS` | Allowed frontend origins (comma-separated, or `*`) | `*` |
| `SEGFORMER_MODEL_VARIANT` | SegFormer ONNX variant (`b2` default or `b0` low memory) | `b2` |
| `SHOP_WHATSAPP_NUMBER` | Shop phone number for WhatsApp inquiries | `+919876543210` |
| `NEXT_PUBLIC_API_URL` | Backend URL for Next.js client | `http://localhost:8000` |
| `NEXT_PUBLIC_SHOP_WHATSAPP_NUMBER` | WhatsApp number exposed to frontend client | `+919876543210` |

---

## 🚀 Running Locally

### 1. Backend Setup

```bash
cd backend

# Create and activate Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download SegFormer ADE20K ONNX model (approx 15 MB)
python scripts/download_model.py

# Seed the database with placeholder Asian Paints catalog
python scripts/seed_db.py

# Start FastAPI development server
uvicorn app.main:app --reload --port 8000
```

The API docs will be accessible at: `http://localhost:8000/docs`

### 2. Frontend Setup

In a new terminal window:

```bash
cd frontend

# Install node dependencies
npm install

# Start Next.js development server
npm run dev
```

Open `http://localhost:3000` in your browser.

For complete cloud deployment instructions (Vercel, Render, Railway, hardware sizing), see [DEPLOY.md](DEPLOY.md).

---

## 💾 Memory Requirements & Model Sizing

Based on empirical benchmarks running the full FastAPI pipeline (model loading, LLM analysis, candidate shortlisting, interactive MobileSAM segmentation, and 3-layer recoloring):

| Model Configuration | Base RSS | Models in RAM | Peak RSS (Request Cycle) | Minimum RAM Recommendation |
| :--- | :---: | :---: | :---: | :---: |
| **SegFormer B2 (Default) + MobileSAM** | ~118 MB | ~331 MB | **~659 MB** | **$\ge$ 1.0 GB dedicated RAM** |
| **SegFormer B0 (`SEGFORMER_MODEL_VARIANT=b0`)** | ~118 MB | ~190 MB | **~420 MB** | **$\ge$ 512 MB dedicated RAM** |

> [!IMPORTANT]
> - By default, **SegFormer B2** is loaded for highest segmentation boundary accuracy.
> - For constrained memory or low-tier cloud instances ($< 512$ MB free RAM), set `SEGFORMER_MODEL_VARIANT=b0` in `.env`.

---

## 🐳 Docker Compose & Database Runtime Status

> [!NOTE]
> **Testing Status**:
> - The application has been developed, tested, and validated locally using **Python 3.12 (`.venv`)**, **SQLite (`sqlite:///./paintmywall.db`)**, and **Next.js 16 (`npm run dev` / `npm run build`)**.
> - **Docker and PostgreSQL are UNTESTED in this environment**: Docker Desktop / Docker daemon was not installed on the development machine. The provided `docker-compose.yml` and PostgreSQL configurations are structured for production deployment but have not been executed or verified live in this environment.

To run with Docker Compose (requires Docker daemon installed):

```bash
docker compose up --build
```

- **Frontend Web App**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **PostgreSQL Database**: `localhost:5432`

---

## 🧪 Testing & Validation

### 1. Running Unit & Integration Tests

All tests run via `pytest`:

```bash
PYTHONPATH=backend ./backend/.venv/bin/pytest backend/tests/ -v
```

Test coverage includes:
- `test_shortlist.py`: CIE-L\*a\*b\* and HSL conversions, circular hue calculation, dim lighting filtering, and category diversity.
- `test_selection_validation.py`: Database ID verification, anti-hallucination protection, dropping invalid IDs, duplicate suppression, and deterministic fallback filling.
- `test_recolor.py`: Shape and dtype preservation, unmasked area protection, masked area chrominance transfer, texture/shadow variance preservation, and mask edge feathering.
- `test_api.py`: `/api/v1/health`, `/api/v1/admin/verify`, `/api/v1/admin/paints` password authentication, `/api/v1/analyze` upload validation and execution, and `/api/v1/recolor-mask` canvas correction.

### 2. Testing the Pipeline via CLI Script

To test the backend pipeline without running the browser frontend:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python ./backend/scripts/cli_test_pipeline.py
```

Optional arguments:
- `--image path/to/your/room.jpg` (if omitted, generates a sample room scene)
- `--output-dir path/to/save` (defaults to `output_samples/`)

The script outputs:
- `original_room.jpg`
- `wall_mask.png`
- `shade_1_<code_name>.jpg`
- `shade_2_<code_name>.jpg`
- `shade_3_<code_name>.jpg`
- `shade_4_<code_name>.jpg`

---

## 📋 How to Import the Real Asian Paints Catalog

> **Note**: Do NOT scrape asianpaints.com. The store owner should obtain the authorized shade list CSV directly from Asian Paints dealer software or export.

### Format of the CSV (`data/paints.csv`):

```csv
brand,name,shade_code,hex,finish,color_family,price
Asian Paints,Morning Glory,0942,#F4EBD9,Matte,Warm Neutral,450.0
Asian Paints,Canvas Cream,L102,#F8F2E4,Eggshell,Warm Neutral,420.0
Asian Paints,Midnight Navy,7370,#1E3A52,Matte,Cool Blue,580.0
```

### Method 1: Via the Web Admin Portal
1. Navigate to `http://localhost:3000/admin`.
2. Enter the admin password (`ADMIN_PASSWORD` in `.env`).
3. Click **"Import CSV Catalog"** and select your `.csv` file.
4. The system will bulk-import/update shades and display an import confirmation.

### Method 2: Via Terminal CLI
```bash
./backend/.venv/bin/python ./backend/scripts/seed_db.py --csv path/to/new_catalog.csv
```
