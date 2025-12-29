from flask import Flask, request, render_template, send_file
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os, cv2
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------- APP SETUP ----------------
app = Flask(__name__)
model = tf.keras.models.load_model("defect_model.h5")

# ---------------- GLOBAL STATE ----------------
inspection_history = []
stats = {"total": 0, "defect": 0, "normal": 0}
last_graph = {"defect": 0, "normal": 0}
latest_report = None

# ---------------- IMAGE QUALITY CHECK ----------------
def image_quality_check(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.Laplacian(img, cv2.CV_64F).var()
    brightness = np.mean(img)

    warnings = []
    if blur < 100:
        warnings.append("⚠ Image appears blurry")
    if brightness < 60:
        warnings.append("⚠ Image appears too dark")
    return warnings

# ---------------- ANALYSIS ----------------
def analyze_image(img_path, threshold=0.3):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0][0]
    if pred < threshold:
        return "DEFECT", round((1 - pred) * 100, 2)
    else:
        return "NORMAL", round(pred * 100, 2)

# ---------------- GRAD-CAM ----------------
def generate_gradcam(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    last_conv_layer = None
    for layer in model.layers[::-1]:
        if len(layer.output_shape) == 4:
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    output_path = "static/heatmap.jpg"
    cv2.imwrite(output_path, overlay)

    return output_path

# ---------------- PDF ----------------
def generate_pdf(filename, result, confidence):
    global latest_report
    os.makedirs("static/reports", exist_ok=True)
    path = "static/reports/Inspection_Report.pdf"

    c = canvas.Canvas(path, pagesize=A4)
    c.setFont("Times-Roman", 16)
    c.drawCentredString(300, 820, "SteelSafe AI – Inspection Report")

    c.setFont("Times-Roman", 12)
    y = 760
    time_str = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    c.drawString(50, y, f"Inspection Time : {time_str}"); y -= 30
    c.drawString(50, y, f"Image Name      : {filename}"); y -= 30
    c.drawString(50, y, f"Result          : {result}"); y -= 30

    # ✅ FIX: this MUST be inside the function
    if result == "DEFECT":
        c.drawString(50, y, f"Defect Type     : General Surface Defect"); y -= 30

    c.drawString(50, y, f"Confidence      : {confidence}%")

    c.showPage()
    c.save()
    latest_report = path


# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    role = request.args.get("role", "operator")
    threshold = float(request.form.get("threshold", 0.3))

    results = []
    warnings = []
    heatmap = None
    has_result = False

    # ✅ FIX: initialize safely
    last_inspection = inspection_history[0] if inspection_history else None

    if request.method == "POST" and role == "operator":
        files = request.files.getlist("images") or [request.files.get("image")]
        os.makedirs("static/uploads", exist_ok=True)

        for file in files:
            if not file:
                continue

            path = os.path.join("static/uploads", file.filename)
            file.save(path)

            warnings += image_quality_check(path)
            status, confidence = analyze_image(path, threshold)
            has_result = True

            last_graph["defect"] = 1 if status == "DEFECT" else 0
            last_graph["normal"] = 1 if status == "NORMAL" else 0

            stats["total"] += 1
            stats[status.lower()] += 1

            inspection_history.insert(0, {
                "time": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
                "file": file.filename,
                "result": status,
                "confidence": confidence
            })

            inspection_history[:] = inspection_history[:5]
            last_inspection = inspection_history[0]

            generate_pdf(file.filename, status, confidence)

            if status == "DEFECT":
                heatmap = generate_gradcam(path)

            results.append({
                "file": file.filename,
                "status": status,
                "confidence": confidence
            })

    return render_template(
        "index.html",
        role=role,
        results=results,
        warnings=warnings,
        history=inspection_history,
        stats=stats,
        last_graph=last_graph,
        heatmap=heatmap,
        has_result=has_result,
        last_inspection=last_inspection,
        report_ready=latest_report is not None
    )

@app.route("/download_report")
def download_report():
    return send_file(latest_report, as_attachment=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
