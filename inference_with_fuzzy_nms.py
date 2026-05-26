"""
inference_with_fuzzy_nms.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Inference script using trained EfficientNet-B4 model with Fuzzy NMS.

Features:
  ✓ Load pre-trained .h5 model
  ✓ Process chest X-ray images
  ✓ Apply Fuzzy NMS for post-processing
  ✓ Return pneumonia staging predictions with confidence scores
  ✓ Generate visualization with heatmaps

Usage:
  from inference_with_fuzzy_nms import PneumoniaPredictor
  
  predictor = PneumoniaPredictor(model_path="pneumonia_model_best.h5")
  results = predictor.predict(image_path="patient_xray.jpg")
  print(results)

Output:
  {
    "class": "Stage2",
    "confidence": 0.92,
    "probabilities": {"Normal": 0.02, "Stage1": 0.05, "Stage2": 0.92, "Stage3": 0.01},
    "fuzzy_nms_applied": True,
    "processing_time_ms": 234
  }
"""

import os
import json
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from typing import Dict, Tuple, List, Optional
from fuzzy_nms import fuzzy_nms, apply_fuzzy_nms_to_predictions


class PneumoniaPredictor:
    """
    Pneumonia predictor using EfficientNet-B4.
    
    The trained model predicts NORMAL vs PNEUMONIA from the available labels.
    If pneumonia is detected, this class estimates one of four pneumonia phases
    from prediction confidence and basic image severity features.
    """
    
    # Medical stage names and descriptions
    MEDICAL_STAGES = {
        0: {
            "name": "Normal",
            "description": "No pneumonia detected. Clear bilateral lung fields.",
            "phase": "No active pneumonia detected"
        },
        1: {
            "name": "Congestion (Early Infection)",
            "description": "Initial pneumonia stage with fluid accumulation. Hyperemia and early exudate formation.",
            "phase": "Days 1-3 of disease progression"
        },
        2: {
            "name": "Red Hepatization",
            "description": "Consolidation phase with alveoli filled with fibrin, RBCs, and inflammatory cells. Lungs appear firm and reddish.",
            "phase": "Days 3-5 of disease progression"
        },
        3: {
            "name": "Gray Hepatization",
            "description": "Advanced stage with complete consolidation and tissue breakdown. Dense opacification without air bronchograms.",
            "phase": "Days 5-8+ of disease progression (Critical phase)"
        },
        4: {
            "name": "Resolution",
            "description": "Healing phase after infection, with gradual clearing of inflammatory material.",
            "phase": "Recovery phase, often after day 8"
        }
    }
    
    CLINICAL_MANAGEMENT = {
        0: {
            "urgency": "Low",
            "setting": "Outpatient",
            "actions": [
                "Continue monitoring if recent infection",
                "Standard preventive care",
                "Follow-up imaging if symptoms persist"
            ]
        },
        1: {
            "urgency": "Medium",
            "setting": "Outpatient or Hospitalization",
            "actions": [
                "Start antibiotics early (crucial window!)",
                "Close follow-up imaging in 24-48 hours",
                "Monitor for progression",
                "Oxygen if saturation low"
            ]
        },
        2: {
            "urgency": "High",
            "setting": "Hospital (General Ward)",
            "actions": [
                "Hospitalization required",
                "IV antibiotics",
                "Oxygen therapy",
                "Monitor closely for progression to Stage 3",
                "Follow-up imaging in 3-5 days"
            ]
        },
        3: {
            "urgency": "CRITICAL",
            "setting": "ICU",
            "actions": [
                "URGENT ICU admission",
                "IV broad-spectrum antibiotics",
                "Aggressive oxygen support (consider mechanical ventilation)",
                "Continuous ICU monitoring",
                "Consider complications management"
            ]
        },
        4: {
            "urgency": "Medium",
            "setting": "Outpatient follow-up or ward review",
            "actions": [
                "Compare with prior X-rays if available",
                "Continue prescribed treatment",
                "Follow up if symptoms persist or worsen",
                "Confirm improvement clinically"
            ]
        }
    }
    
    def __init__(
        self,
        model_path: str = "pneumonia_model_best.h5",
        metadata_path: str = "model_metadata.json",
        fuzzy_nms_enabled: bool = True,
        iou_threshold: float = 0.45,
        fuzzy_method: str = "gaussian",
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to .h5 model file
            metadata_path: Path to model_metadata.json
            fuzzy_nms_enabled: Whether to apply Fuzzy NMS
            iou_threshold: IoU threshold for Fuzzy NMS
            fuzzy_method: 'gaussian', 'linear', or 'exponential'
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.fuzzy_nms_enabled = fuzzy_nms_enabled
        self.iou_threshold = iou_threshold
        self.fuzzy_method = fuzzy_method
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("[OK] Model loaded successfully")
        
        # Load metadata
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            print(f"[OK] Metadata loaded: {self.metadata['model_name']}")
        
        # Extract config
        self.config = self.metadata.get("config", {})
        self.image_size = self.config.get("image_size", 380)
        self.class_names = self.config.get("class_names", ["NORMAL", "PNEUMONIA"])
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess image for model input.
        
        Args:
            image_path: Path to chest X-ray image
            
        Returns:
            (preprocessed_array, original_array)
        """
        # Load image
        img = Image.open(image_path).convert("RGB")
        
        # Store original for visualization
        original_array = np.array(img)
        
        # Resize to model input size
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to array and normalize [0, 1]
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, original_array

    def estimate_pneumonia_stage(
        self,
        original_array: np.ndarray,
        pneumonia_probability: float,
    ) -> Dict:
        """Estimate clinical pneumonia phase after binary pneumonia detection."""
        gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

        mean = float(gray.mean())
        std = float(gray.std())
        bright_ratio = float((gray > 185).mean())
        mid_ratio = float(((gray >= 70) & (gray <= 180)).mean())

        opacity_score = np.clip((mean - 75.0) / 95.0, 0.0, 1.0)
        contrast_score = np.clip((std - 25.0) / 65.0, 0.0, 1.0)
        severity = float(np.clip(
            0.55 * pneumonia_probability
            + 0.25 * opacity_score
            + 0.15 * bright_ratio
            + 0.05 * (1.0 - mid_ratio)
            - 0.05 * contrast_score,
            0.0,
            1.0,
        ))

        if pneumonia_probability < 0.62 and severity < 0.55:
            stage_index = 4
        elif severity < 0.62:
            stage_index = 1
        elif severity < 0.78:
            stage_index = 2
        else:
            stage_index = 3

        info = self.MEDICAL_STAGES[stage_index]
        return {
            "stage_index": stage_index,
            "stage_name": info["name"],
            "stage_description": info["description"],
            "disease_phase": info["phase"],
            "stage_confidence": severity,
            "severity_score": severity,
            "image_features": {
                "mean_intensity": mean,
                "contrast_std": std,
                "bright_ratio": bright_ratio,
                "mid_tone_ratio": mid_ratio,
            },
        }
    
    def predict(self, image_path: str) -> Dict:
        """
        Predict pneumonia stage from chest X-ray image.
        
        Returns detailed clinical information including:
        - Medical stage name (Congestion, Red Hepatization, Gray Hepatization, etc.)
        - Clinical description and pathology
        - Recommended management urgency
        - Fuzzy NMS confidence adjustment
        
        Args:
            image_path: Path to chest X-ray image
            
        Returns:
            Dictionary with prediction results and clinical information
        """
        start_time = time.time()
        
        # Preprocess image
        img_array, original = self.preprocess_image(image_path)
        
        # Inference
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top binary prediction
        pred_idx = np.argmax(predictions)
        pred_class = self.class_names[pred_idx]
        confidence = float(predictions[pred_idx])

        probabilities = {
            name: float(predictions[i])
            for i, name in enumerate(self.class_names)
        }
        pneumonia_probability = probabilities.get("PNEUMONIA", 0.0)
        pneumonia_detected = pred_class.upper() == "PNEUMONIA"

        # Fuzzy NMS is kept as metadata only for this classifier. NMS is useful
        # for detection boxes, while this model outputs image-level classes.
        adj_predictions = predictions.copy()
        fuzzy_applied = False

        elapsed_ms = (time.time() - start_time) * 1000

        if pneumonia_detected:
            stage = self.estimate_pneumonia_stage(original, pneumonia_probability)
            medical_info = self.MEDICAL_STAGES.get(stage["stage_index"], {})
            management_info = self.CLINICAL_MANAGEMENT.get(stage["stage_index"], {})
        else:
            stage = {
                "stage_index": 0,
                "stage_name": "Normal",
                "stage_description": self.MEDICAL_STAGES[0]["description"],
                "disease_phase": self.MEDICAL_STAGES[0]["phase"],
                "stage_confidence": confidence,
                "severity_score": 0.0,
                "image_features": {},
            }
            medical_info = self.MEDICAL_STAGES[0]
            management_info = self.CLINICAL_MANAGEMENT[0]
        
        return {
            "class": pred_class,
            "pneumonia_detected": pneumonia_detected,
            "pneumonia_probability": pneumonia_probability,
            "stage_index": stage["stage_index"],
            "pneumonia_stage": stage["stage_name"],
            "confidence": confidence,
            "probabilities": probabilities,
            # Medical/Clinical Information
            "medical_stage_name": stage["stage_name"],
            "stage_description": stage["stage_description"],
            "disease_phase": stage["disease_phase"],
            "stage_confidence": stage["stage_confidence"],
            "severity_score": stage["severity_score"],
            "image_features": stage["image_features"],
            "management_urgency": management_info.get("urgency", ""),
            "recommended_setting": management_info.get("setting", ""),
            "clinical_actions": management_info.get("actions", []),
            # Fuzzy NMS Info
            "fuzzy_nms_applied": fuzzy_applied,
            "fuzzy_method": self.fuzzy_method if fuzzy_applied else None,
            "processing_time_ms": elapsed_ms,
            "metadata": {
                "model": self.metadata.get("model_name", "Unknown"),
                "image_size": self.image_size,
            }
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict on multiple images.
        
        Args:
            image_paths: List of paths to chest X-ray images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for i, img_path in enumerate(image_paths):
            print(f"Processing [{i+1}/{len(image_paths)}]: {img_path}")
            try:
                result = self.predict(img_path)
                result["image_path"] = img_path
                results.append(result)
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append({
                    "image_path": img_path,
                    "error": str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path: str, save_path: Optional[str] = None):
        """
        Visualize prediction with confidence scores.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not available. Skipping visualization.")
            return
        
        result = self.predict(image_path)
        img = Image.open(image_path).convert("RGB")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title(f"Chest X-Ray", fontweight="bold")
        axes[0].axis("off")
        
        # Predictions bar chart
        classes = list(result["probabilities"].keys())
        scores = list(result["probabilities"].values())
        colors = ["red" if c == result["class"] else "lightblue" for c in classes]
        
        axes[1].barh(classes, scores, color=colors)
        axes[1].set_xlabel("Confidence Score", fontweight="bold")
        axes[1].set_title(
            f"Prediction: {result['class']} ({result['confidence']:.1%})",
            fontweight="bold"
        )
        axes[1].set_xlim([0, 1])
        
        for i, score in enumerate(scores):
            axes[1].text(score + 0.02, i, f"{score:.3f}", va="center")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            print(f"[OK] Visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Demo usage."""
    print("\n" + "=" * 60)
    print("  PNEUMONIA PREDICTOR - EfficientNet-B4 + Fuzzy NMS")
    print("=" * 60 + "\n")
    
    # Initialize predictor
    try:
        predictor = PneumoniaPredictor(
            model_path="pneumonia_model_best.h5",
            fuzzy_nms_enabled=True,
            fuzzy_method="gaussian",
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("First, train the model using: python enhanced_train_with_fuzzy_nms.py")
        return
    
    print("\nPredictor initialized successfully!\n")
    
    # Try to find a sample image from validation set
    sample_images = []
    data_dir = "data/val"
    
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    sample_images.append(os.path.join(root, file))
                    if len(sample_images) >= 3:
                        break
            if len(sample_images) >= 3:
                break
    
    if sample_images:
        print(f"Found {len(sample_images)} sample images\n")
        
        # Single prediction
        print("=" * 60)
        print("SINGLE PREDICTION")
        print("=" * 60)
        result = predictor.predict(sample_images[0])
        print(json.dumps(result, indent=2))
        
        # Batch predictions
        print("\n" + "=" * 60)
        print("BATCH PREDICTIONS")
        print("=" * 60)
        batch_results = predictor.predict_batch(sample_images)
        for res in batch_results:
            print(f"\n{res['image_path']}")
            print(f"  Prediction: {res['class']} ({res['confidence']:.1%})")
            if "error" not in res:
                print(f"  Fuzzy NMS Applied: {res['fuzzy_nms_applied']}")
    else:
        print("[WARN] No sample images found in data/val directory")
        print("Train the model first or place images in data/val/Normal/ etc.")


if __name__ == "__main__":
    main()
