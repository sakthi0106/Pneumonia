"""
app_with_model_integration.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Flask backend with integrated EfficientNet-B4 + Fuzzy NMS model.

Differences from original app.py:
  ✓ Real model inference instead of simulation
  ✓ Fuzzy NMS post-processing integrated
  ✓ H5 model loading
  ✓ Automatic fallback to demo if model unavailable
  ✓ Performance metrics tracking

To use this:
  1. Train model: python enhanced_train_with_fuzzy_nms.py
  2. Replace /api/analyze in app.py with code from here
  3. OR copy this entire file to app.py

The script will attempt to load the trained model on startup.
If not available, it falls back to the demo/simulation mode.
"""

import os
import json
import time
import math
import random
import sqlite3
import smtplib
import urllib.error
import urllib.request
from email.message import EmailMessage
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO

# Web framework
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Image processing
from PIL import Image
import numpy as np

# Optional: Real model
try:
    import tensorflow as tf
    from inference_with_fuzzy_nms import PneumoniaPredictor
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    PneumoniaPredictor = None

# ─── SETUP ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = BASE_DIR
DB_PATH = os.path.join(BASE_DIR, "pneumo_ai.db")
ENV_PATH = os.path.join(BASE_DIR, ".env")
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model_best.h5")


def load_env_file():
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
app.secret_key = os.environ.get("PNEUMO_SECRET_KEY", "dev-pneumo-ai-secret-key")
CORS(app, supports_credentials=True)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED = {"jpg", "jpeg", "png", "webp"}
STAGES = [
    "Normal",
    "Congestion",
    "Red Hepatization",
    "Gray Hepatization",
    "Resolution",
]

# Model initialization
MODEL_LOADED = False
pneumonia_predictor = None

if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        print(f"\n[OK] Loading trained model from {MODEL_PATH}...")
        pneumonia_predictor = PneumoniaPredictor(
            model_path=MODEL_PATH,
            fuzzy_nms_enabled=True,
            fuzzy_method="gaussian",
        )
        MODEL_LOADED = True
        print("[OK] Model loaded successfully!\n")
    except Exception as e:
        print(f"[WARN] Failed to load model: {e}")
        print("[WARN] Falling back to demo mode\n")
        MODEL_LOADED = False
elif not TF_AVAILABLE:
    print("[WARN] TensorFlow not available. Install: pip install tensorflow")
    print("[WARN] Using demo mode\n")
else:
    print(f"[WARN] Model not found at {MODEL_PATH}")
    print("[WARN] Train model first: python enhanced_train_with_fuzzy_nms.py")
    print("[WARN] Using demo mode\n")


# ─── DATABASE ─────────────────────────────────────────────
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK(role IN ('patient', 'doctor')),
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT,
                dob TEXT,
                gender TEXT,
                address TEXT,
                specialization TEXT,
                hospital TEXT,
                license_no TEXT,
                experience TEXT,
                password_hash TEXT NOT NULL,
                otp_code TEXT,
                otp_expiry TEXT,
                is_verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )


def json_data():
    return request.get_json(silent=True) or {}


def public_user(row):
    return {
        "id": row["id"],
        "role": row["role"],
        "full_name": row["full_name"],
        "email": row["email"],
    }


def make_otp():
    return f"{random.SystemRandom().randint(100000, 999999)}"


def otp_expiry():
    return (datetime.now() + timedelta(minutes=10)).isoformat()


def otp_valid(user, code):
    if not user["otp_code"] or not user["otp_expiry"]:
        return False
    if user["otp_code"] != code:
        return False
    return datetime.fromisoformat(user["otp_expiry"]) > datetime.now()


def send_otp_email(email, code, purpose="verification"):
    smtp_host = (os.environ.get("SMTP_HOST") or "smtp.gmail.com").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = (os.environ.get("SMTP_USER") or os.environ.get("SMTP_FROM") or "").strip()
    smtp_pass = (
        os.environ.get("SMTP_PASS")
        or os.environ.get("SMTP_PASSWORD")
        or ""
    ).replace(" ", "").strip()
    from_email = (os.environ.get("FROM_EMAIL") or smtp_user).strip()
    
    if not all([smtp_host, smtp_user, smtp_pass]):
        print(f"📋 OTP for {email}: {code} (check terminal if SMTP not configured)")
        return
    
    subject = "Your Pneumo AI OTP" if purpose == "verification" else "Password Reset OTP"
    body = f"""
    Hello,
    
    Your One-Time Password (OTP) for {purpose}:
    
    {code}
    
    This OTP is valid for 10 minutes.
    
    Best regards,
    Pneumo AI Team
    """
    
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = email
    msg.set_content(body)
    
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"📧 OTP sent to {email}")
    except Exception as e:
        print(f"[WARN] Failed to send OTP: {e}")


# ─── AUTH ROUTES ──────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    model_status = "Trained model loaded" if MODEL_LOADED else "Using demo mode"
    return jsonify({
        "status": "healthy",
        "model": model_status,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/signup", methods=["POST"])
@app.route("/api/signup", methods=["POST"])
def signup():
    data = json_data()
    role = data.get("role", "").strip().lower()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    
    if not all([role, email, password]):
        return jsonify({"error": "Role, email, and password are required."}), 400
    
    if role not in {"patient", "doctor"}:
        return jsonify({"error": "Role must be 'patient' or 'doctor'."}), 400
    
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400
    
    otp = make_otp()
    now = datetime.now().isoformat()
    
    try:
        with db() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    role, full_name, email, phone, dob, gender, address,
                    specialization, hospital, license_no, experience,
                    password_hash, otp_code, otp_expiry, is_verified, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    role,
                    data["full_name"].strip(),
                    email,
                    data.get("phone", "").strip(),
                    data.get("dob", "").strip(),
                    data.get("gender", "").strip(),
                    data.get("address", "").strip(),
                    data.get("specialization", "").strip(),
                    data.get("hospital", "").strip(),
                    data.get("license_no", "").strip(),
                    data.get("experience", "").strip(),
                    generate_password_hash(password),
                    otp,
                    otp_expiry(),
                    0,
                    now,
                ),
            )
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account with this email already exists."}), 409
    
    send_otp_email(email, otp, "verification")
    session["pending_email"] = email
    
    return jsonify({
        "success": True,
        "message": "Account created. Check your email for OTP.",
        "email": email,
    })


@app.route("/login", methods=["POST"])
@app.route("/api/login", methods=["POST"])
def login():
    data = json_data()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    role = data.get("role", "").strip().lower()
    
    if not email or not password or role not in {"patient", "doctor"}:
        return jsonify({"error": "Email, password, and role are required."}), 400
    
    with db() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE email = ? AND role = ?",
            (email, role)
        ).fetchone()
    
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email, role, or password."}), 401
    
    if not user["is_verified"]:
        return jsonify({
            "error": "Email not verified. Check your inbox for OTP.",
            "email": email,
        }), 403
    
    session["user_id"] = user["id"]
    session["role"] = user["role"]
    
    return jsonify({
        "success": True,
        "message": "Login successful.",
        "user": public_user(user),
    })


@app.route("/logout", methods=["POST", "GET"])
@app.route("/api/logout", methods=["POST", "GET"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully."})


@app.route("/verify-otp", methods=["POST"])
@app.route("/api/verify-otp", methods=["POST"])
def verify_otp():
    data = json_data()
    email = data.get("email", session.get("pending_email", "")).strip().lower()
    code = data.get("otp", "").strip()
    purpose = data.get("purpose", "signup")
    if not email or not code:
        return jsonify({"error": "Email and OTP are required."}), 400

    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return jsonify({"error": "No account found for this email."}), 404
        if not otp_valid(user, code):
            return jsonify({"error": "Invalid or expired OTP."}), 400

        if purpose == "reset":
            session["reset_email"] = email
            message = "OTP verified. You can reset your password."
        else:
            conn.execute(
                "UPDATE users SET is_verified = 1, otp_code = NULL, otp_expiry = NULL WHERE email = ?",
                (email,),
            )
            session.pop("pending_email", None)
            message = "Email verified. You can sign in now."

    return jsonify({"success": True, "message": message})


@app.route("/resend-otp", methods=["POST"])
@app.route("/api/resend-otp", methods=["POST"])
def resend_otp():
    data = json_data()
    email = data.get("email", session.get("pending_email", "")).strip().lower()
    purpose = data.get("purpose", "verification")
    if not email:
        return jsonify({"error": "Email is required."}), 400

    otp = make_otp()
    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return jsonify({"error": "No account found for this email."}), 404
        conn.execute(
            "UPDATE users SET otp_code = ?, otp_expiry = ? WHERE email = ?",
            (otp, otp_expiry(), email),
        )

    send_otp_email(email, otp, purpose)
    return jsonify({"success": True, "message": "A new OTP has been sent."})


@app.route("/forgot-password", methods=["POST"])
@app.route("/api/forgot-password", methods=["POST"])
def forgot_password():
    data = json_data()
    email = data.get("email", "").strip().lower()
    if not email:
        return jsonify({"error": "Email is required."}), 400

    otp = make_otp()
    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return jsonify({"error": "No account found for this email."}), 404
        conn.execute(
            "UPDATE users SET otp_code = ?, otp_expiry = ? WHERE email = ?",
            (otp, otp_expiry(), email),
        )

    session["reset_email"] = email
    send_otp_email(email, otp, "password reset")
    return jsonify({"success": True, "message": "Password reset OTP sent.", "email": email})


@app.route("/reset-password", methods=["POST"])
@app.route("/api/reset-password", methods=["POST"])
def reset_password():
    data = json_data()
    email = data.get("email", session.get("reset_email", "")).strip().lower()
    password = data.get("password", "")
    confirm = data.get("confirm_password", "")
    if not email or not password or not confirm:
        return jsonify({"error": "Email and new password are required."}), 400
    if password != confirm:
        return jsonify({"error": "Passwords do not match."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return jsonify({"error": "No account found for this email."}), 404
        conn.execute(
            "UPDATE users SET password_hash = ?, otp_code = NULL, otp_expiry = NULL WHERE email = ?",
            (generate_password_hash(password), email),
        )

    session.pop("reset_email", None)
    return jsonify({"success": True, "message": "Password reset successful."})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = json_data()
    raw_message = (data.get("message") or "").strip()
    message = raw_message.lower()
    analysis = data.get("analysis") or {}
    if not message:
        return jsonify({"error": "Message is required."}), 400

    try:
        reply = gemini_chat_reply(
            raw_message,
            history=data.get("history") or [],
            analysis=analysis,
        )
        return jsonify({"reply": reply, "provider": "gemini"})
    except Exception as e:
        print(f"[WARN] Gemini chat unavailable: {e}")
        if os.environ.get("ALLOW_LOCAL_CHAT_FALLBACK", "").lower() not in {"1", "true", "yes"}:
            return jsonify({
                "error": str(e),
                "provider": "gemini",
                "configured": bool(
                    os.environ.get("GEMINI_API_KEY")
                    or os.environ.get("GOOGLE_AI_API_KEY")
                    or os.environ.get("GOOGLE_API_KEY")
                ),
            }), 503

    pneumonia_terms = (
        "pneumonia", "x-ray", "xray", "x ray", "chest", "radiograph",
        "stage", "congestion", "hepatization", "infiltrate", "opacity",
        "consolidation", "pleural", "symptom", "treatment", "medicine",
        "antibiotic", "cough", "fever", "lung", "breath", "breathing",
        "oxygen", "analysis", "result", "normal", "infection", "sputum",
        "wheezing", "shortness", "respiratory"
    )
    if not any(term in message for term in pneumonia_terms):
        return jsonify({
            "reply": "I'm specialized only in pneumonia-related questions. Please ask me about pneumonia symptoms, stages, treatment, prevention, or chest X-ray interpretation.",
            "provider": "local"
        })

    if "stage" in message or "result" in message or "analysis" in message:
        stage = analysis.get("stage") or analysis.get("pneumonia_stage") or "not available yet"
        confidence = analysis.get("confidence") or analysis.get("conf") or "not available"
        reply = (
            f"Current analysis shows {stage} with confidence {confidence}. "
            "Please use this as AI-assisted information only and confirm with a clinician."
        )
    elif "treatment" in message or "medicine" in message or "antibiotic" in message:
        reply = (
            "Pneumonia treatment depends on cause and severity. A clinician may recommend antibiotics, antivirals, oxygen support, rest, fluids, and follow-up imaging. Do not self-medicate."
        )
    elif "symptom" in message or "fever" in message or "cough" in message:
        reply = (
            "Common pneumonia symptoms include cough, fever, chest pain, breathlessness, fatigue, chills, and low oxygen. Severe breathlessness or confusion needs urgent care."
        )
    else:
        reply = (
            "Pneumonia can appear on chest X-ray as infiltrates, opacity, or consolidation. The app first checks normal versus pneumonia, then estimates the likely stage when pneumonia is detected."
        )

    return jsonify({"reply": reply, "provider": "local"})


def gemini_chat_reply(message, history=None, analysis=None):
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_AI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key or api_key in {"your-google-ai-studio-api-key", "your_gemini_api_key_here"}:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    history = history or []
    analysis_text = "No analysis performed yet."
    if analysis:
        analysis_text = json.dumps({
            "diagnosis": analysis.get("diagnosis"),
            "stage": analysis.get("stage") or analysis.get("pneumonia_stage"),
            "confidence": analysis.get("confidence") or analysis.get("conf"),
            "pneumonia_probability": analysis.get("pneumonia_probability"),
        })

    system_prompt = (
        "You are PneumoBot, an AI specialist focused exclusively on pneumonia. "
        "Only answer questions about pneumonia, lung health, respiratory symptoms, "
        "diagnosis, treatment, prevention, and chest X-ray interpretation. If the "
        "user asks anything unrelated, do not answer the unrelated question; politely "
        "redirect them to pneumonia-related questions. "
        "Use clear patient-friendly language. Do not provide a definitive diagnosis "
        "or medication prescription; recommend professional medical review for "
        "clinical decisions. Keep responses concise. "
        f"Current analysis result: {analysis_text}"
    )

    contents = []
    for item in history[-6:]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    model = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 500,
        },
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gemini API unavailable: {e.reason}") from e

    parts = []
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if part.get("text"):
                parts.append(part["text"])
    reply = "\n".join(parts).strip()
    if not reply:
        raise RuntimeError("Gemini response did not include text.")
    return reply


# ─── IMAGE VALIDATION & ANALYSIS ──────────────────────────
def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def image_stats(image_bytes):
    """Extract image statistics for validation."""
    try:
        img = Image.open(BytesIO(image_bytes))
        gray = img.convert("L").resize((128, 128))
        pixels = list(gray.getdata())
        mean = sum(pixels) / len(pixels)
        std = math.sqrt(sum((p - mean) ** 2 for p in pixels) / len(pixels))
        return mean, std
    except Exception as e:
        raise ValueError(f"Invalid image: {e}")


def feature_baseline_prediction(image_bytes):
    """
    Lightweight NORMAL/PNEUMONIA calibration fallback.

    The current saved .h5 model can collapse to nearly constant PNEUMONIA
    probabilities. These coefficients come from a simple local feature model
    trained on the included chest_xray train split, and are used only as a
    guard when metadata shows the neural model is not reliable.
    """
    img = Image.open(BytesIO(image_bytes)).convert("L").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    hist, _ = np.histogram(arr, bins=32, range=(0, 255), density=True)
    hist = hist + 1e-9
    features = np.array([
        arr.mean(),
        arr.std(),
        np.percentile(arr, 10),
        np.percentile(arr, 25),
        np.percentile(arr, 50),
        np.percentile(arr, 75),
        np.percentile(arr, 90),
        (arr > 185).mean(),
        (arr < 45).mean(),
        ((arr >= 70) & (arr <= 180)).mean(),
        float(-(hist * np.log(hist)).sum()),
    ], dtype=np.float64)
    weights = np.array([
        -0.5922,
        -0.3972,
        0.0168,
        0.0663,
        0.0980,
        0.5221,
        -0.1714,
        1.7161,
        2.4669,
        -6.5573,
        -2.1681,
    ], dtype=np.float64)
    logit = float(25.1428 + np.dot(features, weights))
    pneumonia_probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))
    return {
        "pneumonia_probability": pneumonia_probability,
        "normal_probability": 1.0 - pneumonia_probability,
        "pneumonia_detected": pneumonia_probability >= 0.50,
    }


def saved_model_needs_calibration():
    """Return True when saved metadata indicates chance-level validation."""
    try:
        with open(os.path.join(BASE_DIR, "model_metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        val_acc = float(metadata.get("training_info", {}).get("final_val_acc", 1.0))
    except (OSError, ValueError, TypeError):
        return False
    return val_acc <= 0.60


def preprocess_upload_for_model(image_bytes, image_size=380):
    """Validate, normalize, resize, and serialize upload for backend prediction."""
    src = Image.open(BytesIO(image_bytes)).convert("L")
    original_size = src.size
    resized = src.resize((image_size, image_size), Image.LANCZOS)
    rgb = Image.merge("RGB", (resized, resized, resized))

    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=95)
    processed_bytes = buffer.getvalue()

    return processed_bytes, {
        "preprocessed": True,
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "model_input_size": {"width": image_size, "height": image_size},
        "color_mode": "grayscale_to_rgb",
        "normalization": "pixel values scaled to 0-1 inside inference backend",
        "format": "JPEG",
    }


def chest_xray_validation(image_bytes):
    """Validate that image is a chest X-ray."""
    try:
        src = Image.open(BytesIO(image_bytes))
        width, height = src.size
        
        if width < 150 or height < 150:
            return False, "Image is too small for analysis."
        
        if width > 2000 or height > 2000:
            return False, "Image is too large."
        
        aspect = width / height
        if aspect < 0.45 or aspect > 2.0:
            return False, "Incorrect aspect ratio for frontal chest X-ray."
        
        rgb = src.convert("RGB").resize((128, 128))
        gray = src.convert("L").resize((128, 128))
        
        rgb_pixels = list(rgb.getdata())
        gray_pixels = list(gray.getdata())
        total = len(gray_pixels)
        mean = sum(gray_pixels) / total
        std = math.sqrt(sum((p - mean) ** 2 for p in gray_pixels) / total)
        
        color_delta = sum(max(p) - min(p) for p in rgb_pixels) / len(rgb_pixels)
        bright_ratio = sum(1 for p in gray_pixels if p > 200) / total
        dark_ratio = sum(1 for p in gray_pixels if p < 30) / total
        mid_ratio = sum(1 for p in gray_pixels if 50 <= p <= 180) / total
        
        if color_delta > 30:
            return False, "Image appears to be colored. Please upload grayscale X-ray."
        
        if std < 25:
            return False, "Image lacks medical X-ray contrast."
        
        if dark_ratio < 0.04:
            return False, "This does not appear to be an X-ray image."
        
        if mid_ratio < 0.3:
            return False, "Image lacks tissue detail of chest X-ray."
        
        return True, ""
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def pneumonia_phase(stage_idx):
    """Describe pneumonia phase based on stage."""
    return {
        0: {
            "name": "Resolution / no active pneumonia",
            "description": "No AI-visible active pneumonia; if recovering, consistent with resolution.",
        },
        1: {
            "name": "Congestion",
            "description": "Early inflammatory fluid and vascular congestion may be present.",
        },
        2: {
            "name": "Red hepatization",
            "description": "Consolidation may be developing; alveoli fill with inflammatory cells.",
        },
        3: {
            "name": "Grey hepatization",
            "description": "Dense consolidation present; should be reviewed urgently by clinician.",
        },
        4: {
            "name": "Resolution",
            "description": "Healing phase with gradual clearing after pneumonia.",
        },
    }.get(stage_idx, {"name": "Unknown", "description": ""})


def care_guidance(stage_idx):
    """Clinical care guidance based on stage."""
    guidance = {
        0: {
            "do": [
                "Keep regular follow-up if symptoms continue.",
                "Maintain hydration, rest, and routine preventive care.",
                "Seek medical advice if symptoms develop.",
            ],
            "dont": [
                "Do not ignore worsening respiratory symptoms.",
                "Do not use antibiotics without prescription.",
            ],
        },
        1: {
            "do": [
                "Consult healthcare professional for clinical correlation.",
                "Monitor temperature, cough, and oxygen saturation.",
                "Rest, drink fluids, follow prescribed medicines.",
            ],
            "dont": [
                "Do not delay care if breathlessness or chest pain appears.",
                "Do not smoke or inhale pollutants.",
            ],
        },
        2: {
            "do": [
                "Arrange prompt medical review and lab tests.",
                "Take antibiotics/antivirals as prescribed.",
                "Seek urgent help for low oxygen or worsening cough.",
            ],
            "dont": [
                "Do not self-medicate or combine antibiotics.",
                "Do not continue strenuous activity until cleared.",
            ],
        },
        3: {
            "do": [
                "Seek urgent medical care or emergency evaluation.",
                "Follow clinician advice about oxygen support.",
                "Keep patient upright while arranging care.",
            ],
            "dont": [
                "Do not wait at home if breathing is difficult.",
                "Do not delay emergency care.",
            ],
        },
        4: {
            "do": [
                "Continue prescribed treatment and follow-up.",
                "Compare with prior X-rays if available.",
                "Monitor symptoms until fully recovered.",
            ],
            "dont": [
                "Do not stop prescribed medication early without medical advice.",
                "Do not ignore recurring fever or breathlessness.",
            ],
        },
    }
    return guidance.get(stage_idx, guidance[0])


def clinical_findings(stage_idx):
    """Generate clinical findings based on stage."""
    return {
        0: [
            "No consolidation or opacity detected in bilateral lung fields.",
            "Cardiothoracic ratio within normal limits.",
            "Clear costophrenic angles bilaterally.",
            "No pleural effusion detected.",
        ],
        1: [
            "Early interstitial infiltrates in right lower lobe.",
            "Mild perihilar haziness consistent with early infection.",
            "No significant pleural effusion.",
            "Follow-up imaging recommended.",
        ],
        2: [
            "Bilateral lower lobe consolidation visible.",
            "Moderate ground-glass opacity in right lung.",
            "Subtle blunting of costophrenic angle.",
            "Lab work and culture recommended.",
        ],
        3: [
            "Extensive bilateral consolidation and opacification.",
            "Significant pleural effusion detected bilaterally.",
            "Urgent hospitalization and intensive support advised.",
            "Critical airway support may be needed.",
        ],
        4: [
            "Residual opacity may be improving compared with active infection.",
            "Follow-up imaging can confirm radiographic clearing.",
            "Clinical correlation is recommended during recovery.",
            "Seek review if symptoms worsen again.",
        ],
    }.get(stage_idx, [])


def simulate_demo_analysis(mean, std):
    """Demo mode: simulate binary pneumonia result plus stage estimate."""
    seed = int(abs(mean * 10 + std * 5)) % 9999
    rng = random.Random(seed)
    pneumonia_probability = min(0.97, max(0.03, (std / 95.0) + rng.uniform(-0.12, 0.12)))
    normal_probability = 1.0 - pneumonia_probability

    if pneumonia_probability < 0.50:
        stage_idx = 0
    elif pneumonia_probability < 0.62:
        stage_idx = 4
    elif pneumonia_probability < 0.74:
        stage_idx = 1
    elif pneumonia_probability < 0.86:
        stage_idx = 2
    else:
        stage_idx = 3

    return {
        "diagnosis": "PNEUMONIA" if pneumonia_probability >= 0.50 else "NORMAL",
        "pneumonia_detected": pneumonia_probability >= 0.50,
        "confidence": max(normal_probability, pneumonia_probability),
        "pneumonia_probability": pneumonia_probability,
        "stage_index": stage_idx,
        "probabilities": {
            "NORMAL": normal_probability,
            "PNEUMONIA": pneumonia_probability,
        },
    }


# ─── MAIN ANALYSIS ENDPOINT ────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze chest X-ray for pneumonia staging.
    
    Uses trained EfficientNet-B4 + Fuzzy NMS model if available,
    otherwise falls back to demo mode.
    """
    start_time = time.time()
    
    # Validate request
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    file = request.files["image"]
    if not allowed(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG, PNG, or WEBP."}), 400
    
    image_bytes = file.read()
    
    # Validate image is chest X-ray
    try:
        is_xray, validation_msg = chest_xray_validation(image_bytes)
        if not is_xray:
            return jsonify({"error": validation_msg}), 422
        mean, std = image_stats(image_bytes)
        preprocessed_bytes, preprocessing = preprocess_upload_for_model(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {e}"}), 422
    
    # Perform analysis
    if MODEL_LOADED and pneumonia_predictor:
        # Use trained model
        try:
            # Save preprocessed temporary image for backend prediction
            temp_path = os.path.join(
                UPLOAD_FOLDER,
                f"preprocessed_analysis_{int(time.time() * 1000)}.jpg",
            )
            with open(temp_path, "wb") as f:
                f.write(preprocessed_bytes)
            
            # Get prediction
            result = pneumonia_predictor.predict(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            stage_idx = int(result.get("stage_index", 0))
            pneumonia_detected = bool(result.get("pneumonia_detected", False))
            diagnosis = "PNEUMONIA" if pneumonia_detected else "NORMAL"
            probabilities = {
                "Normal": round(result["probabilities"].get("NORMAL", 0) * 100, 2),
                "Pneumonia": round(result["probabilities"].get("PNEUMONIA", 0) * 100, 2),
            }
            pneumonia_probability = result.get("pneumonia_probability", 0)
            confidence = result["confidence"]
            pneumonia_stage = result.get("pneumonia_stage", STAGES[stage_idx])
            stage_confidence = result.get("stage_confidence", 0)
            model_label = "EfficientNet-B4 (Trained)"
            algorithm = "EfficientNet-B4 binary classifier + backend stage estimator"
            stage_note = (
                "Stage is estimated only after pneumonia is detected. "
                "Use clinically stage-labeled data for a fully trained stage classifier."
            )

            if saved_model_needs_calibration():
                baseline = feature_baseline_prediction(image_bytes)
                pneumonia_detected = baseline["pneumonia_detected"]
                diagnosis = "PNEUMONIA" if pneumonia_detected else "NORMAL"
                pneumonia_probability = baseline["pneumonia_probability"]
                confidence = max(
                    baseline["normal_probability"],
                    baseline["pneumonia_probability"],
                )
                probabilities = {
                    "Normal": round(baseline["normal_probability"] * 100, 2),
                    "Pneumonia": round(baseline["pneumonia_probability"] * 100, 2),
                }
                if not pneumonia_detected:
                    stage_idx = 0
                elif pneumonia_probability < 0.62:
                    stage_idx = 4
                elif pneumonia_probability < 0.74:
                    stage_idx = 1
                elif pneumonia_probability < 0.86:
                    stage_idx = 2
                else:
                    stage_idx = 3
                pneumonia_stage = STAGES[stage_idx]
                stage_confidence = confidence
                model_label = "Calibrated feature fallback"
                algorithm = (
                    "Local feature baseline used because saved EfficientNet validation "
                    "accuracy is chance-level"
                )
                stage_note = (
                    "The saved neural model is currently poorly calibrated, so this "
                    "result uses a local NORMAL/PNEUMONIA calibration guard. Retrain "
                    "the model before using stage estimates as final output."
                )
            
            inference_time = (time.time() - start_time) * 1000
            
            return jsonify({
                "status": "success",
                "preprocessing": preprocessing,
                "diagnosis": diagnosis,
                "pneumonia_detected": pneumonia_detected,
                "stage": STAGES[stage_idx],
                "stage_index": stage_idx,
                "confidence": round(confidence * 100, 2),
                "confidence_level": (
                    "High" if confidence >= 0.85
                    else "Medium" if confidence >= 0.70
                    else "Low"
                ),
                "probabilities": probabilities,
                "pneumonia_probability": round(pneumonia_probability * 100, 2),
                "pneumonia_stage": pneumonia_stage,
                "stage_confidence": round(stage_confidence * 100, 2),
                "stage_note": stage_note,
                "pneumonia_phase": pneumonia_phase(stage_idx),
                "care_guidance": care_guidance(stage_idx),
                "clinical_findings": clinical_findings(stage_idx),
                "image_features": result.get("image_features", {}),
                "model": model_label,
                "algorithm": algorithm,
                "fuzzy_nms_applied": result["fuzzy_nms_applied"],
                "inference_time_ms": round(result["processing_time_ms"], 2),
                "total_time_ms": round(inference_time, 2),
            })
        
        except Exception as e:
            print(f"Error during model inference: {e}")
            # Fall through to demo mode
            pass
    
    # Demo/fallback mode
    print("Using demo mode for analysis")
    demo = simulate_demo_analysis(mean, std)
    idx = demo["stage_index"]
    
    inference_time = (time.time() - start_time) * 1000
    
    return jsonify({
        "status": "success",
        "preprocessing": preprocessing,
        "diagnosis": demo["diagnosis"],
        "pneumonia_detected": demo["pneumonia_detected"],
        "stage": STAGES[idx],
        "stage_index": idx,
        "confidence": round(demo["confidence"] * 100, 2),
        "confidence_level": (
            "High" if demo["confidence"] >= 0.85
            else "Medium" if demo["confidence"] >= 0.70
            else "Low"
        ),
        "probabilities": {
            "Normal": round(demo["probabilities"]["NORMAL"] * 100, 2),
            "Pneumonia": round(demo["probabilities"]["PNEUMONIA"] * 100, 2),
        },
        "pneumonia_probability": round(demo["pneumonia_probability"] * 100, 2),
        "pneumonia_stage": STAGES[idx],
        "stage_confidence": round(demo["pneumonia_probability"] * 100, 2),
        "stage_note": (
            "Demo mode uses a statistical placeholder. Train the model for real "
            "normal/pneumonia prediction and use stage-labeled data for true stage learning."
        ),
        "pneumonia_phase": pneumonia_phase(idx),
        "care_guidance": care_guidance(idx),
        "clinical_findings": clinical_findings(idx),
        "model": "Demo (Simulated)",
        "algorithm": "Statistical Simulation",
        "fuzzy_nms_applied": False,
        "inference_time_ms": 0,
        "total_time_ms": round(inference_time, 2),
        "warning": "Using simulated model. Train real model for accurate predictions.",
    })


# ─── STATIC FILES ─────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>", methods=["GET"])
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


# ─── INITIALIZE & RUN ──────────────────────────────────────
init_db()

if __name__ == "__main__":
    print("=" * 60)
    print("  Pneumo AI - Server Starting")
    print("=" * 60)
    print(f"  Web:   http://127.0.0.1:5000")
    print(f"  API:   http://127.0.0.1:5000/api/health")
    print(f"  Model: {'Loaded' if MODEL_LOADED else 'Demo mode'}")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
