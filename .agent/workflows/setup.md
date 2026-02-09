---
description: Complete setup for the Disha repository (backend + frontend)
---

# Disha Repository Setup

This workflow guides you through setting up the **Disha - Urban Traffic Intelligence** repository from scratch.

## Prerequisites

Ensure you have:

- **Python 3.10+** installed
- **Node.js 18+** installed

## Setup Steps

### 1. Backend Setup

Navigate to backend directory and create virtual environment:

```bash
cd /Users/aditisingh/Documents/projects/Disha/neuroflow_backend
```

// turbo

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

// turbo
Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Frontend Setup

Navigate to frontend directory:

```bash
cd /Users/aditisingh/Documents/projects/Disha/neuroflow_frontend
```

// turbo
Install Node.js dependencies:

```bash
npm install
```

Create environment file from example:

```bash
cp .env.example .env
```

### 3. Verify Installation

Check backend dependencies:

```bash
cd /Users/aditisingh/Documents/projects/Disha/neuroflow_backend && source .venv/bin/activate && pip list
```

Check frontend dependencies:

```bash
cd /Users/aditisingh/Documents/projects/Disha/neuroflow_frontend && npm list --depth=0
```

## Running the Application

After setup is complete, you can run the application using two separate terminals:

**Terminal 1 — Backend:**

```bash
cd /Users/aditisingh/Documents/projects/Disha/neuroflow_backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**

```bash
cd /Users/aditisingh/Documents/projects/Disha/neuroflow_frontend
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) in your browser.
