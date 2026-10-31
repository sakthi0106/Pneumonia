"""
Pneumo AI - Flask Backend
SQLite authentication + deterministic demo pneumonia analysis.
"""

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from io import BytesIO
import os
import time
import math
import random
import json
import sqlite3
import smtplib
import urllib.error
import urllib.request
from email.message import EmailMessage
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = BASE_DIR
DB_PATH = os.path.join(BASE_DIR, "pneumo_ai.db")
ENV_PATH = os.path.join(BASE_DIR, ".env")


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
FRONTEND_ORIGIN = os.environ.get(
    "FRONTEND_ORIGIN",
    "https://pneumonia-five.vercel.app"
)

CORS(
    app,
    resources={r"/*": {"origins": [FRONTEND_ORIGIN]}},
    supports_credentials=True
)

app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED = {"jpg", "jpeg", "png", "webp"}
STAGES = ["Normal", "Stage 1 (Mild)", "Stage 2 (Moderate)", "Stage 3 (Severe)"]


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
    return (datetime.utcnow() + timedelta(minutes=5)).isoformat(timespec="seconds")


def otp_valid(row, code):
    if not row["otp_code"] or not row["otp_expiry"]:
        return False
    try:
        expires = datetime.fromisoformat(row["otp_expiry"])
    except ValueError:
        return False
    return row["otp_code"] == code and datetime.utcnow() <= expires


def send_otp_email(email, otp, purpose="verification"):
    subject = "Pneumo AI verification code"
    body = (
        f"Your Pneumo AI {purpose} OTP is {otp}.\n\n"
        "This code expires in 5 minutes."
    )

    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    sender = os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER") or os.environ.get("FROM_EMAIL")
    password = os.environ.get("SMTP_PASSWORD") or os.environ.get("SMTP_PASS")
    if password:
        password = password.replace(" ", "").strip()
    port = int(os.environ.get("SMTP_PORT", "587"))

    if not host or not sender or not password:
        print("=" * 55)
        print(f"Pneumo AI OTP for {email}: {otp}")
        print("SMTP is not configured, so the OTP is printed here.")
        print("=" * 55)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = email
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(sender, password)
            smtp.send_message(msg)
    except Exception as e:
        print("=" * 55)
        print(f"Could not send OTP email to {email}: {e}")
        print(f"Pneumo AI OTP for {email}: {otp}")
        print("Fix Gmail SMTP settings, then try Resend OTP.")
        print("=" * 55)
        raise


def require_fields(data, fields):
    missing = [f for f in fields if not str(data.get(f, "")).strip()]
    if missing:
        return f"Missing required field(s): {', '.join(missing)}"
    return None


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/signup", methods=["POST"])
def signup():
    data = json_data()
    role = data.get("role", "").strip().lower()
    if role not in {"patient", "doctor"}:
        return jsonify({"error": "Please select Patient or Doctor."}), 400

    common = ["full_name", "email", "phone", "password", "confirm_password"]
    role_fields = (
        ["dob", "gender", "address"]
        if role == "patient"
        else ["specialization", "hospital", "license_no", "experience"]
    )
    error = require_fields(data, common + role_fields)
    if error:
        return jsonify({"error": error}), 400

    email = data["email"].strip().lower()
    password = data["password"]
    if password != data["confirm_password"]:
        return jsonify({"error": "Passwords do not match."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    now = datetime.utcnow().isoformat(timespec="seconds")
    try:
        with db() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    role, full_name, email, phone, dob, gender, address,
                    specialization, hospital, license_no, experience,
                    password_hash, otp_code, otp_expiry, is_verified, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 1, ?)
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
                    now,
                ),
            )
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account with this email already exists."}), 409

    return jsonify({
        "success": True,
        "message": "Account created successfully. Please sign in.",
        "email": email,
    })


@app.route("/verify-otp", methods=["POST"])
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


@app.route("/login", methods=["POST"])
@app.route("/api/login", methods=["POST"])
def login():
    data = json_data()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    role = data.get("role", "").strip().lower()
    if not email or not password or role not in {"patient", "doctor"}:
        return jsonify({"error": "Role, email, and password are required."}), 400

    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ? AND role = ?", (email, role)).fetchone()

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
        "dashboard": f"/{user['role']}-dashboard",
        "user": public_user(user),
    })


@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out."})


@app.route("/forgot-password", methods=["POST"])
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
def reset_password():
    data = json_data()
    email = data.get("email", session.get("reset_email", "")).strip().lower()
    password = data.get("password", "")
    confirm = data.get("confirm_password", "")
    if not email or not password or not confirm:
        return jsonify({"error": "Email and new password are required."}), 400
    if password != confirm:
        return jsonify({"error": "Passwords do not match."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return jsonify({"error": "No account found for this email."}), 404
        conn.execute(
            """
            UPDATE users
            SET password_hash = ?, otp_code = NULL, otp_expiry = NULL
            WHERE email = ?
            """,
            (generate_password_hash(password), email),
        )

    session.pop("reset_email", None)
    return jsonify({"success": True, "message": "Password reset successful. Please sign in."})


def dashboard_payload(expected_role):
    user_id = session.get("user_id")
    if not user_id or session.get("role") != expected_role:
        return jsonify({"error": "Unauthorized."}), 401
    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        session.clear()
        return jsonify({"error": "Session expired."}), 401
    return jsonify({"success": True, "user": public_user(user), "dashboard": expected_role})


@app.route("/patient-dashboard")
def patient_dashboard():
    return dashboard_payload("patient")


@app.route("/doctor-dashboard")
def doctor_dashboard():
    return dashboard_payload("doctor")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "Pneumo AI", "version": "1.0.0"})


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
            "stage": analysis.get("stage"),
            "confidence": analysis.get("confidence") or analysis.get("conf"),
            "pneumonia_probability": analysis.get("pneumonia_probability"),
            "phase": (analysis.get("phase") or {}).get("name"),
        })

    system_prompt = (
        "You are PneumoBot, an AI specialist focused exclusively on pneumonia. "
        "Only answer questions related to pneumonia, lung health, respiratory symptoms, "
        "diagnosis, treatment, prevention, and chest X-ray interpretation. If the user "
        "asks about anything unrelated, do not answer the unrelated question; politely "
        "redirect them to pneumonia-related questions. "
        "Use clear patient-friendly language. Do not diagnose definitively; advise "
        "professional medical review for clinical decisions. "
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

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 500,
        },
    }
    model = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

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
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError("Gemini response did not include text.")
    return text


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message is required."}), 400
    try:
        reply = gemini_chat_reply(
            message,
            history=payload.get("history") or [],
            analysis=payload.get("analysis") or None,
        )
        return jsonify({"reply": reply, "provider": "gemini"})
    except Exception as e:
        return jsonify({"error": str(e)}), 503


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def image_stats(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("L")
    img = img.resize((64, 64))
    pixels = list(img.getdata())
    mean = sum(pixels) / len(pixels)
    std = math.sqrt(sum((p - mean) ** 2 for p in pixels) / len(pixels))
    return mean, std


def chest_xray_validation(image_bytes):
    """Enhanced validation to detect chest X-rays and reject non-medical images."""
    with Image.open(BytesIO(image_bytes)) as src:
        width, height = src.size
        if min(width, height) < 200:
            return False, "Image is too small. Please upload a clear chest X-ray (min 200px)."

        # Stricter aspect ratio for chest X-rays (typically vertical or square)
        aspect = width / height
        if aspect < 0.6 or aspect > 1.4:
            return False, "Please upload a frontal chest X-ray image, not a cropped or tilted image."

        rgb = src.convert("RGB").resize((128, 128))
        gray = src.convert("L").resize((128, 128))

    rgb_pixels = list(rgb.getdata())
    gray_pixels = list(gray.getdata())
    total = len(gray_pixels)
    mean = sum(gray_pixels) / total
    std = math.sqrt(sum((p - mean) ** 2 for p in gray_pixels) / total)

    # Check color channels (X-rays are grayscale)
    color_delta = sum(max(p) - min(p) for p in rgb_pixels) / len(rgb_pixels)
    bright_ratio = sum(1 for p in gray_pixels if p > 200) / total
    dark_ratio = sum(1 for p in gray_pixels if p < 30) / total
    mid_ratio = sum(1 for p in gray_pixels if 50 <= p <= 180) / total

    # STRICT: Reject colored images
    if color_delta > 30:
        return False, "This appears to be a colored image. Please upload a grayscale chest X-ray."
    
    # STRICT: X-rays must have significant contrast and detail
    if std < 25:
        return False, "The image lacks medical X-ray contrast. Please upload a chest X-ray image."
    
    # X-rays typically have dark backgrounds and mid-tone content
    if dark_ratio < 0.04:
        return False, "This does not appear to be an X-ray image. Please upload a chest X-ray."
    
    if mid_ratio < 0.3:
        return False, "This image lacks the tissue detail of a chest X-ray. Please upload a chest X-ray."

    def region_mean(x1, y1, x2, y2):
        values = []
        for y in range(y1, y2):
            row = y * 128
            for x in range(x1, x2):
                values.append(gray_pixels[row + x])
        return sum(values) / len(values) if values else 0

    # Check lung regions (should be darker/denser in X-rays)
    left_lung = region_mean(20, 32, 55, 100)
    right_lung = region_mean(73, 32, 108, 100)
    center_chest = region_mean(58, 26, 70, 102)
    lung_mean = (left_lung + right_lung) / 2
    side_difference = abs(left_lung - right_lung)

    # Check edges (should be darker - background)
    edge_values = []
    for y in range(128):
        for x in range(128):
            if x < 10 or x >= 118 or y < 10 or y >= 118:
                edge_values.append(gray_pixels[y * 128 + x])
    edge_mean = sum(edge_values) / len(edge_values)

    score = 0
    # Lung regions should be darker than edges
    if left_lung < edge_mean and right_lung < edge_mean:
        score += 2
    # Left and right lungs should be similar
    if side_difference <= 25:
        score += 1
    # Center (mediastinum) should be denser
    if abs(center_chest - lung_mean) >= 5:
        score += 1
    # Contrast check
    if 25 <= std <= 120:
        score += 1
    # Bright ratio should be low (not a bright photo)
    if bright_ratio < 0.4:
        score += 1
    # Edge should be darker than mean
    if edge_mean <= mean - 10:
        score += 1

    if score < 5:
        return False, "This image does not match chest X-ray characteristics. Please upload a frontal chest X-ray."

    return True, ""


def softmax(values):
    e = [math.exp(v) for v in values]
    t = sum(e)
    return [x / t for x in e]


def simulate_cnn(mean, std):
    seed = int(abs(mean * 10 + std * 5)) % 9999
    rng = random.Random(seed)
    logits = [rng.uniform(0.5, 2.0) for _ in range(4)]
    winner = logits.index(max(logits))
    logits[winner] += rng.uniform(1.5, 3.0)
    return softmax(logits)


def fuzzy_nms(boxes, scores, iou_thresh=0.45, alpha=0.7):
    if not boxes:
        return [], []
    paired = sorted(zip(scores, boxes), key=lambda x: x[0], reverse=True)
    scores = [p[0] for p in paired]
    boxes = [p[1] for p in paired]
    adj = list(scores)
    kept = []
    active = list(range(len(boxes)))
    while active:
        i = active[0]
        kept.append(i)
        b1 = boxes[i]
        area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        remaining = []
        for j in active[1:]:
            b2 = boxes[j]
            area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2])
            iy2 = min(b1[3], b2[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            iou = inter / (area1 + area2 - inter + 1e-6)
            w = math.exp(-alpha * iou**2) if iou >= iou_thresh else 1.0
            adj[j] *= w
            if adj[j] > 0.05:
                remaining.append(j)
        active = remaining
    return [boxes[i] for i in kept], [adj[i] for i in kept]


def detection_boxes(stage_idx):
    if stage_idx == 0:
        return {"boxes": [], "scores": []}
    rng = random.Random(stage_idx * 42)
    regions = [[0.25, 0.4, 0.55, 0.85], [0.45, 0.35, 0.78, 0.82]]
    boxes, scores = [], []
    n = 6 + stage_idx * 3
    for reg in regions:
        cx = (reg[0] + reg[2]) / 2 * 380
        cy = (reg[1] + reg[3]) / 2 * 380
        w = (reg[2] - reg[0]) * 380
        h = (reg[3] - reg[1]) * 380
        for _ in range(n // 2):
            bx1 = max(0, cx - w / 2 + rng.uniform(-0.12 * w, 0.12 * w))
            by1 = max(0, cy - h / 2 + rng.uniform(-0.12 * h, 0.12 * h))
            bx2 = min(380, bx1 + w * rng.uniform(0.8, 1.2))
            by2 = min(380, by1 + h * rng.uniform(0.8, 1.2))
            boxes.append([bx1, by1, bx2, by2])
            scores.append(rng.uniform(0.5, 0.99))
    kb, ks = fuzzy_nms(boxes, scores)
    return {"boxes": kb, "scores": ks}


def findings(stage_idx):
    return {
        0: [
            "No consolidation or opacity detected in bilateral lung fields.",
            "Cardiothoracic ratio within normal limits (<0.5).",
            "Clear costophrenic angles bilaterally.",
            "No pleural effusion detected.",
        ],
        1: [
            "Early interstitial infiltrates noted in right lower lobe.",
            "Mild perihilar haziness, consistent with early-stage infection.",
            "No significant pleural effusion.",
            "Recommend follow-up chest X-ray in 5-7 days.",
        ],
        2: [
            "Bilateral lower lobe consolidation with air bronchograms.",
            "Moderate ground-glass opacity in right lung field.",
            "Subtle blunting of left costophrenic angle.",
            "Consider sputum culture and CBC for causative pathogen.",
        ],
        3: [
            "Extensive bilateral consolidation and opacification.",
            "Significant pleural effusion detected bilaterally.",
            "Air bronchograms visible; marked respiratory compromise.",
            "Urgent hospitalization and intensive respiratory support advised.",
        ],
    }.get(stage_idx, [])


def pneumonia_phase(stage_idx):
    return {
        0: {
            "name": "Resolution / no active pneumonia",
            "description": "No AI-visible active pneumonia is suggested; if recovering, this is most consistent with resolution.",
        },
        1: {
            "name": "Congestion",
            "description": "Early inflammatory fluid and vascular congestion may be present.",
        },
        2: {
            "name": "Red hepatization",
            "description": "Consolidation may be developing as alveoli fill with inflammatory cells and fluid.",
        },
        3: {
            "name": "Grey hepatization",
            "description": "Dense consolidation may be present and should be reviewed urgently by a clinician.",
        },
    }.get(stage_idx, {
        "name": "Resolution",
        "description": "Findings may be improving after treatment; compare with prior X-rays when available.",
    })


def care_guidance(stage_idx):
    guidance = {
        0: {
            "do": [
                "Keep regular follow-up if symptoms continue.",
                "Maintain hydration, rest, and routine preventive care.",
                "Seek medical advice if fever, cough, chest pain, or breathing trouble develops.",
            ],
            "dont": [
                "Do not ignore worsening respiratory symptoms.",
                "Do not use antibiotics without a clinician's prescription.",
            ],
        },
        1: {
            "do": [
                "Consult a healthcare professional for clinical correlation.",
                "Monitor temperature, cough, oxygen saturation, and breathing effort.",
                "Rest, drink fluids, and follow prescribed medicines exactly.",
            ],
            "dont": [
                "Do not delay care if breathlessness, chest pain, or high fever appears.",
                "Do not smoke or inhale dust, fumes, or strong pollutants.",
                "Do not stop prescribed treatment early without medical advice.",
            ],
        },
        2: {
            "do": [
                "Arrange prompt medical review and recommended lab tests.",
                "Take antibiotics or antivirals only as prescribed.",
                "Seek urgent help for low oxygen, confusion, persistent fever, or worsening cough.",
            ],
            "dont": [
                "Do not self-medicate or combine antibiotics without supervision.",
                "Do not continue strenuous activity until cleared by a clinician.",
                "Do not miss follow-up imaging if the doctor requests it.",
            ],
        },
        3: {
            "do": [
                "Seek urgent medical care or emergency evaluation.",
                "Follow clinician advice about oxygen support, admission, or intensive monitoring.",
                "Keep the patient upright and monitor breathing while arranging care.",
            ],
            "dont": [
                "Do not wait at home if breathing is difficult or oxygen is low.",
                "Do not give sedatives or cough suppressants without medical advice.",
                "Do not delay emergency care for older adults, infants, or high-risk patients.",
            ],
        },
    }
    return guidance.get(stage_idx, guidance[0])


@app.route("/api/analyze", methods=["POST"])
def analyze():
    start = time.time()
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    file = request.files["image"]
    if not allowed(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG, PNG, or WEBP."}), 400
    image_bytes = file.read()
    try:
        is_chest_xray, validation_error = chest_xray_validation(image_bytes)
        if not is_chest_xray:
            return jsonify({"error": validation_error}), 422
        mean, std = image_stats(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {e}"}), 422
    time.sleep(0.6)
    probs = simulate_cnn(mean, std)
    idx = probs.index(max(probs))
    conf = probs[idx] * 100
    return jsonify({
        "stage": STAGES[idx],
        "stage_index": idx,
        "confidence": round(conf, 2),
        "confidence_level": "High" if conf >= 85 else "Medium" if conf >= 70 else "Low",
        "probabilities": {
            "Normal": round(probs[0] * 100, 2),
            "Stage 1 (Mild)": round(probs[1] * 100, 2),
            "Stage 2 (Moderate)": round(probs[2] * 100, 2),
            "Stage 3 (Severe)": round(probs[3] * 100, 2),
        },
        "pneumonia_phase": pneumonia_phase(idx),
        "care_guidance": care_guidance(idx),
        "detection_boxes": detection_boxes(idx),
        "clinical_findings": findings(idx),
        "model": "EfficientNet-B4 (simulated)",
        "algorithm": "Fuzzy Non-Max Suppression (a=0.7, IoU=0.45)",
        "inference_time_s": round(time.time() - start, 3),
    })


init_db()

if __name__ == "__main__":
    print("=" * 55)
    print("  Pneumo AI - Server starting")
    print("  Open:  http://127.0.0.1:5000")
    print("  API:   http://127.0.0.1:5000/api/health")
    print("  OTP:   Check this terminal unless SMTP env vars are set")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
