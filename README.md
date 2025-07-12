# ❤️ HeartGuard‑Pro

A dual-component tool for heart failure detection:

- **model.py** – A reusable machine-learning model handler: supports training, evaluation, storage, selection of best model, visualization & summary.
- **app.py** – A Streamlit app that leverages model.py to offer an interactive heart failure detection dashboard.

---

## 🚀 Features

- **Unified ML API** via `model.py`
  - Load existing model or train new ones
  - Automatically selects best performer
  - Saves/restores model state (checkpointing)
  - Provides training history plots and model summaries
- **Real-time detection** via Streamlit
  - Upload or simulate input data
  - User-friendly interface for clinicians/researchers

---

## 🏁 Quick Start

### Prerequisites

- Python 3.9+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
