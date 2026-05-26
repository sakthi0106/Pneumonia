# 🫁 Pneumo AI — Intelligent Pneumonia Detection System

Advanced Pneumonia Detection & Care Assistant using EfficientNet-B4 CNN + Fuzzy Non-Max Suppression

---

## 📁 Project Structure

```
pneumo-ai/
├── index.html                  ← Full frontend (Login + Analysis + Chatbot)
├── README.md                   ← This file
└── backend/
    ├── app.py                  ← Flask REST API
    ├── fuzzy_nms.py            ← Fuzzy NMS algorithm module
    ├── train_model.py          ← EfficientNet-B4 training script
    └── requirements.txt        ← Python dependencies
```

---

## 🚀 Features

### Frontend (`index.html`)
- **Split Login Page** — Doctor & Patient portals with separate credentials
- **X-Ray Upload** — Drag & drop or click to browse chest X-ray images
- **AI Analysis Panel** — Step-by-step loading with animated progress
- **Results Display**
  - Stage classification: Normal / Stage 1 / Stage 2 / Stage 3
  - Confidence score with visual probability bars
  - Clinical observations per stage
  - Algorithm metadata (EfficientNet-B4, Fuzzy NMS, inference time)
- **AI Care Chatbot** — Gemini-powered, pneumonia-only assistant
  - Answers only pneumonia-related queries
  - Context-aware of current analysis result
  - Quick-reply chips for common questions
- **Recent Analyses** sidebar — History of past scans
- Fully responsive, no framework dependencies

### Backend (`backend/app.py`)
- Flask REST API
- `/api/analyze` — Image upload → CNN inference → Fuzzy NMS → JSON response
- `/api/login` — Auth endpoint (extend with JWT + DB for production)
- `/api/health` — Health check

### ML Pipeline
- **Model**: EfficientNet-B4 (ImageNet pretrained, fine-tuned on chest X-rays)
- **NMS**: Fuzzy Non-Maximum Suppression for region detection
  - Gaussian / Linear / Exponential membership functions
  - Soft score decay instead of hard box elimination
- **Training**: Two-phase transfer learning (frozen base → fine-tune top layers)

---

## 🛠 Setup & Run

### Frontend Only (No backend needed)
Run the Flask backend and open `http://127.0.0.1:5000`. The chatbot calls Google AI Studio's Gemini API through `/api/chat`.

> **Note**: Add your Google AI Studio API key as `GEMINI_API_KEY` in `.env` to enable the chatbot.  
> Without it, the chatbot runs in offline mode with built-in pneumonia responses.

---

### Backend Setup

```bash
# 1. Navigate to backend
cd pneumo-ai/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask server
python app.py
```

Server starts at: `http://localhost:5000`

---

### Connect Frontend to Backend

In `index.html`, update the analyze button handler to call:
```javascript
const response = await fetch("http://localhost:5000/api/analyze", {
  method: "POST",
  body: formData,   // FormData with image file
});
const result = await response.json();
```

---

## 🧠 AI Model Details

### EfficientNet-B4
| Property       | Value                  |
|----------------|------------------------|
| Input size     | 380 × 380 × 3          |
| Pretrained on  | ImageNet               |
| Fine-tuned on  | NIH Chest X-Ray / Kaggle Pneumonia |
| Output classes | 4 (Normal, Stage 1–3)  |
| Optimizer      | Adam (lr=1e-4 → 1e-5)  |

### Fuzzy Non-Max Suppression
| Parameter      | Value    |
|----------------|----------|
| IoU Threshold  | 0.45     |
| Method         | Gaussian |
| Sigma (σ)      | 0.5      |
| Alpha (decay)  | 0.7      |
| Score cutoff   | 0.05     |

**Why Fuzzy NMS?**  
Traditional NMS makes hard binary decisions (keep/discard) which can miss overlapping pathological regions.  
Fuzzy NMS uses a soft membership function to gradually reduce confidence scores — preserving ambiguous regions for radiologist review.

---

## 📊 Pneumonia Staging

| Stage   | Description                                   | Clinical Signs                              |
|---------|-----------------------------------------------|---------------------------------------------|
| Normal  | No pneumonia detected                         | Clear lung fields, normal cardiothoracic ratio |
| Stage 1 | Mild — early interstitial infiltrates         | Perihilar haziness, no effusion             |
| Stage 2 | Moderate — bilateral consolidation            | Air bronchograms, ground-glass opacity      |
| Stage 3 | Severe — extensive consolidation + effusion   | Urgent hospitalization required             |

---

## 🤖 Chatbot (PneumoBot)

Powered by Google AI Studio / Gemini API (`gemini-2.5-flash` by default).

**System prompt restricts responses to:**
- Pneumonia symptoms, stages, diagnosis
- Treatment options and medications
- Prevention strategies
- Interpretation of X-ray results
- General respiratory health

Non-pneumonia queries are politely redirected.

---

## 📦 Training Your Own Model

```bash
# 1. Prepare dataset (NIH ChestX-ray14 or Kaggle)
# Structure: data/train/{Normal,Stage1,Stage2,Stage3}/

# 2. Install ML dependencies
pip install tensorflow scikit-learn matplotlib seaborn

# 3. Train
python train_model.py
```

Outputs:
- `pneumonia_efficientnetb4.h5` — Trained model
- `training_curves.png` — Accuracy/loss plots
- `confusion_matrix.png` — Evaluation matrix
- `training_log.csv` — Epoch-by-epoch metrics

---

## ⚠️ Disclaimer

This system is for **research and educational purposes only**.  
It is NOT a substitute for professional medical diagnosis.  
Always consult a qualified radiologist or pulmonologist for clinical decisions.

---

## 📄 License

MIT License — Free to use, modify, and distribute with attribution.
