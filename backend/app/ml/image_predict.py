"""Image disease prediction with a Grad-CAM heatmap overlay.

Loads lazily: if the image model hasn't been trained yet, the app still runs
(symptom prediction keeps working) and image endpoints report that clearly.
"""

import base64
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "image_model.keras")
CLASSES_PATH = os.path.join(HERE, "image_classes.json")
DISEASES_PATH = os.path.join(HERE, "diseases.json")

IMG_SIZE = 224

# Map image-model classes to the knowledge-base disease ids
IMG_CLASS_TO_KB = {"healthy": None, "fmd": "fmd", "lsd": "lsd"}

_state = {"model": None, "classes": None, "loaded": False, "available": False}

with open(DISEASES_PATH, encoding="utf-8") as f:
    _kb = {d["id"]: d for d in json.load(f)["diseases"]}


def is_available():
    _lazy_load()
    return _state["available"]


def _lazy_load():
    if _state["loaded"]:
        return
    _state["loaded"] = True
    if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)):
        _state["available"] = False
        return
    try:
        import tensorflow as tf  # imported only when needed
    except ImportError:
        # Model is present but TensorFlow isn't installed — degrade gracefully.
        _state["available"] = False
        return
    _state["model"] = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        _state["classes"] = json.load(f)["classes"]
    _state["available"] = True


def _preprocess(image_bytes):
    import tensorflow as tf
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(arr.copy())
    return arr, np.expand_dims(x, 0)


def _grad_cam(model, x, class_idx):
    """Return a 0-1 heatmap over the image for the predicted class."""
    import tensorflow as tf
    # The input to the global-average-pooling layer is the last conv feature map,
    # and it is connected to the model input (robust for nested MobileNetV2).
    gap = next(l for l in model.layers
               if l.__class__.__name__ == "GlobalAveragePooling2D")
    grad_model = tf.keras.models.Model(model.inputs, [gap.input, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def _overlay(original_arr, cam):
    """Blend the heatmap over the original image, return a base64 PNG."""
    from PIL import Image
    h, w = original_arr.shape[:2]
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((w, h))
    cam_res = np.array(cam_img).astype("float32") / 255.0
    # simple red heatmap
    heat = np.zeros_like(original_arr)
    heat[..., 0] = cam_res * 255
    blended = np.uint8(0.6 * original_arr + 0.4 * heat)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def check_human_face(image_bytes):
    """Detect human faces using OpenCV YuNet ONNX model."""
    yunet_model = os.path.join(HERE, "face_detection_yunet.onnx")
    if not os.path.exists(yunet_model):
        return False
    try:
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]
        detector = cv2.FaceDetectorYN_create(yunet_model, "", (w, h), score_threshold=0.6)
        _, faces = detector.detect(img)
        return bool(faces is not None and len(faces) > 0)
    except Exception:
        return False


_imagenet_model = None
NON_LIVESTOCK_TERMS = {
    'car', 'automobile', 'jeep', 'truck', 'bus', 'vehicle', 'bicycle', 'motorcycle',
    'chair', 'table', 'desk', 'sofa', 'couch', 'bed', 'furniture', 'wardrobe',
    'laptop', 'computer', 'keyboard', 'screen', 'monitor', 'cellular_telephone', 'phone', 'ipod', 'remote_control',
    'bottle', 'cup', 'mug', 'bowl', 'plate', 'fork', 'spoon', 'knife', 'pot',
    'shoe', 'sneaker', 'sandal', 'boot', 'suit', 'jersey', 'shirt', 'jean', 'dress', 'coat', 'jacket',
    'book', 'paper', 'envelope', 'packet', 'carton', 'box', 'web_site', 'comic_book',
    'traffic_light', 'street_sign', 'building', 'bridge', 'castle', 'church', 'house'
}


def check_non_livestock_object(image_bytes):
    """Detect non-livestock everyday objects or man-made articles using ImageNet MobileNetV2."""
    global _imagenet_model
    try:
        from PIL import Image
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
        if _imagenet_model is None:
            _imagenet_model = MobileNetV2(weights="imagenet")
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype="float32")
        x = preprocess_input(np.expand_dims(arr, 0))
        preds = _imagenet_model.predict(x, verbose=0)
        top3 = decode_predictions(preds, top=3)[0]
        for _, name, conf in top3:
            name_lower = name.lower()
            if conf >= 0.25 and any(term in name_lower for term in NON_LIVESTOCK_TERMS):
                readable = name.replace("_", " ").title()
                return True, readable, float(conf)
        return False, None, 0.0
    except Exception:
        return False, None, 0.0


def validate_livestock_image(image_bytes):
    """Validate uploaded photo: reject human faces, blank images, or non-livestock items."""
    # 1. Human Face Check (OpenCV YuNet)
    if check_human_face(image_bytes):
        return (
            False,
            "मानव चेहरा पहचाना गया (Human Face Detected)। VetAI केवल गाय एवं भैंस (मवेशी) के त्वचा एवं घाव रोगों के परीक्षण के लिए है। कृपया केवल पशु के प्रभावित अंग या त्वचा की स्पष्ट तस्वीर अपलोड करें। (Human face detected. VetAI is exclusively designed for cattle disease screening. Please upload a photo of the animal's affected area.)",
            "human_face"
        )

    # 2. Blank / Dark / Low-Variance Check
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        if arr.std() < 12.0:
            return (
                False,
                "अस्पष्ट या खाली फोटो (Blank / Unclear Photo)। फोटो में पर्याप्त रोशनी नहीं है या मवेशी दिखाई नहीं दे रहा है। कृपया साफ रोशनी में पशु की तस्वीर लें। (Photo is too dark or lacks sufficient contrast. Please take a clear, well-lit photo.)",
                "blank_image"
            )
    except Exception:
        pass

    # 3. Non-livestock Everyday Objects / Screens Check
    is_non_live, obj_name, _ = check_non_livestock_object(image_bytes)
    if is_non_live:
        return (
            False,
            f"गैर-पशु फोटो पहचानी गई ({obj_name})। VetAI केवल गाय और भैंस के त्वचा एवं घाव परीक्षण के लिए है। कृपया केवल पशु की स्पष्ट तस्वीर अपलोड करें। (Non-livestock image detected: {obj_name}. Please upload a clear photo of cattle skin or lesion.)",
            "non_livestock"
        )

    return True, None, None


def predict_image(image_bytes, skip_validation=False):
    _lazy_load()
    if not _state["available"]:
        return {"available": False,
                "message": "Image model not trained yet. Train it with train_image_model.py."}

    # Strict OOD and Human Face validation
    if not skip_validation:
        is_valid, err_msg, _ = validate_livestock_image(image_bytes)
        if not is_valid:
            raise ValueError(err_msg)

    model, classes = _state["model"], _state["classes"]
    original, x = _preprocess(image_bytes)
    proba = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(proba))
    top_class = classes[top_idx]

    ranked = sorted(
        [{"class": classes[i], "confidence": round(float(proba[i]) * 100, 1)}
         for i in range(len(classes))],
        key=lambda d: d["confidence"], reverse=True,
    )

    top_conf = ranked[0]["confidence"]
    margin = ranked[0]["confidence"] - ranked[1]["confidence"]

    # If the model prediction is highly uncertain or flat (ambiguous/OOD image)
    if top_conf < 58.0 or margin < 10.0:
        raise ValueError(
            "अस्पष्ट या अनिर्धारित फोटो (Ambiguous / Out-of-Distribution Image)। "
            f"छवि में मवेशी के रोग लक्षण स्पष्ट रूप से नहीं पहचाने जा सके (विश्वास स्तर: {top_conf}% < 58%)। "
            "कृपया घाव की नजदीकी व स्पष्ट तस्वीर अपलोड करें या नीचे दिए गए लक्षणों को चुनकर जांच करें। "
            "(Image features are ambiguous or unrecognized. Please provide a clear close-up of cattle lesions or select symptoms manually.)"
        )

    kb_id = IMG_CLASS_TO_KB.get(top_class)
    disease = _kb.get(kb_id) if kb_id else None
    heatmap = _overlay(original, _grad_cam(model, x, top_idx))

    return {
        "available": True,
        "top_class": top_class,
        "is_healthy": top_class == "healthy",
        "confidence": top_conf,
        "ranked": ranked,
        "proba": {classes[i]: float(proba[i]) for i in range(len(classes))},
        "disease": disease,           # None if healthy
        "heatmap": heatmap,           # base64 PNG overlay
    }
