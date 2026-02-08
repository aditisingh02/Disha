# Disha - Urban Traffic Intelligence

**Datathon 2026**

### Problem Statement

Design a **machine learning-based time-series forecasting system** to **predict traffic congestion levels** in **urban environments**. The system should **integrate data from multiple sources** (GPS, traffic sensors, weather, city events) to generate **accurate hourly or daily congestion forecasts**. It must **handle multivariate inputs**, **adapt to evolving traffic patterns**, and produce **reliable predictions** that support **real-time traffic control**, **route planning**, and **congestion mitigation strategies**.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**

---

## Setup

### 1. Backend

```bash
cd neuroflow_backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Frontend

```bash
cd neuroflow_frontend
npm install
```

---

## Run

Start backend and frontend in separate terminals.

**Terminal 1 — Backend:**
```bash
cd neuroflow_backend
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd neuroflow_frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

