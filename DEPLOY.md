# PaintMyWall Deployment Guide 🚀

This document provides exact, step-by-step instructions for deploying both the **FastAPI Backend** and the **Next.js Frontend**, sizing hardware resources correctly, and diagnosing connectivity issues.

---

## 🖥️ System & Resource Requirements

### Backend Host Sizing
- **Recommended RAM**: **$\ge$ 2.0 GB RAM** (e.g. Render Standard, Railway 2GB, Fly.io 2GB, or AWS EC2 `t3.small` / `t4g.small`).
  - *Empirical Peak Memory*: Under live inference with **SegFormer B2** (105 MB ONNX) + **MobileSAM** (26 MB ONNX) performing full vision analysis, segmentation, and 3-layer LAB recoloring of 4 shades, the process peak RSS is **~659 MB**.
  - Sizing for $\ge$ 2 GB provides the required headroom for operating system buffers, OpenCV image arrays, and concurrent requests.
  - *Low-Memory Fallback*: If deploying to a 512 MB – 1 GB tier, set `SEGFORMER_MODEL_VARIANT=b0` in the backend environment variables (peak RSS stays under ~420 MB).
- **Disk Storage**: $\ge$ 2.0 GB free disk space for ONNX model weights and temporary uploaded session caches.
- **Python Runtime**: Python 3.12.

### Frontend Host Sizing
- Any standard Node.js / edge platform (e.g. **Vercel**, **Cloudflare Pages**, or **Render Static Site**).

---

## 🛠️ Step 1: Deploying the Backend (e.g. Render / Railway / VM)

### 1.1 Service Configuration
- **Environment**: Python 3.12
- **Root Directory**: `backend` (if deploying from repository root, prepend `backend/` to commands or set root directory to `backend`)
- **Build Command**:
  ```bash
  pip install -r requirements.txt && python scripts/download_model.py --variant b2 && python scripts/download_sam.py && python scripts/seed_db.py
  ```
  *(Note: Even if model pre-download is omitted during build, the backend's automatic startup lifespan check will detect missing ONNX models and download them on boot).*
- **Start Command**:
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port $PORT
  ```

### 1.2 Backend Environment Variables

| Variable | Required | Example / Recommended Value | Description |
| :--- | :---: | :--- | :--- |
| `LLM_PROVIDER` | Yes | `gemini` (or `groq`) | Active Vision & Selection LLM provider |
| `GEMINI_API_KEY` | Optional* | `AIzaSy...` | Google Gemini API key (*required if `LLM_PROVIDER=gemini`*) |
| `GROQ_API_KEY` | Optional* | `gsk_...` | Groq API key (*required if `LLM_PROVIDER=groq`*) |
| `CORS_ORIGINS` | **Yes** | `https://your-frontend.vercel.app,http://localhost:3000` | Comma-separated list of allowed frontend domains. Use `*` to permit all origins. |
| `ADMIN_PASSWORD` | **Yes** | `your_secure_password_2026` | Password for accessing the `/admin` paint management portal |
| `DATABASE_URL` | Optional | `sqlite:///./paintmywall.db` | Database connection string (SQLite by default, or PostgreSQL) |
| `SEGFORMER_MODEL_VARIANT`| Optional | `b2` | `b2` (default, high quality) or `b0` (low memory) |
| `SHOP_WHATSAPP_NUMBER` | Optional | `+919876543210` | Store owner WhatsApp number for customer inquiries |

---

## 🌐 Step 2: Deploying the Frontend (e.g. Vercel)

### 2.1 Project Settings on Vercel
1. Import the Git repository in your Vercel Dashboard.
2. Set **Root Directory** to `frontend`.
3. Framework Preset: **Next.js** (detected automatically).
4. Build Command: `npm run build`
5. Output Directory: `.next`

### 2.2 Frontend Environment Variables

Configure these in the Vercel project settings (**Settings > Environment Variables**):

| Variable | Required | Example Value | Description |
| :--- | :---: | :--- | :--- |
| `NEXT_PUBLIC_API_URL` | **CRITICAL** | `https://your-backend.onrender.com` | **Must point to your deployed backend URL with NO trailing slash.** |
| `NEXT_PUBLIC_SHOP_WHATSAPP_NUMBER` | Optional | `+919876543210` | WhatsApp number pre-filled in customer inquiry buttons. |

> [!WARNING]
> **Important Next.js Build-Time Behavior**:
> Next.js inlines all `NEXT_PUBLIC_*` variables into the static JavaScript bundle **during the build process**. If you change `NEXT_PUBLIC_API_URL` in your hosting dashboard, you **MUST trigger a Redeploy** (with "Clear build cache") for the change to take effect in the browser.

---

## 🔍 Root Cause Diagnosis: Why "Failed to fetch" Occurs

If clicking **"Generate 4 Asian Paints Shades"** displays `"Failed to fetch"` in the browser:

1. **Dead or Stale Tunnel URL (Most Common)**:
   - In earlier configurations, `NEXT_PUBLIC_API_URL` defaulted to a temporary Cloudflare tunnel (`customs-sat-literally-methods.trycloudflare.com`). When temporary tunnels expire or close, the browser cannot resolve the domain name (`net::ERR_NAME_NOT_RESOLVED`), resulting in an immediate `"Failed to fetch"` error.
   - **Fix**: Set `NEXT_PUBLIC_API_URL` to your permanent live backend URL in your frontend host settings and redeploy.
2. **Missing `NEXT_PUBLIC_API_URL` in Production**:
   - If `NEXT_PUBLIC_API_URL` is omitted in production, the frontend now throws a clear configuration error explaining that the variable must be set, rather than silently attempting to connect to `localhost`.
3. **CORS Block**:
   - If the backend's `CORS_ORIGINS` does not include your frontend's Vercel/custom domain, modern browsers will block the response before JavaScript can read it.
   - **Fix**: Set `CORS_ORIGINS=https://your-frontend.vercel.app` (or `CORS_ORIGINS=*`) on the backend service.
4. **Cold Starts or Out of Memory (OOM)**:
   - If the backend host has $< 1$ GB RAM, the Python process may be terminated by the OS Out-Of-Memory (OOM) killer during model initialization or image processing.
   - **Fix**: Ensure host has $\ge 2$ GB RAM, or set `SEGFORMER_MODEL_VARIANT=b0` to cut memory consumption down to $< 450$ MB.
