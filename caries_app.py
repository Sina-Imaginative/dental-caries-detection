import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Load your trained model
model = YOLO("best.pt")

def detect_caries(image):
    results = model.predict(
        source=image,
        conf=0.15,
        save=False,
        verbose=False
    )
    
    # Get the annotated image
    annotated = results[0].plot()
    annotated_rgb = Image.fromarray(annotated[..., ::-1])
    
    # Build detection summary
    boxes = results[0].boxes
    names = model.names
    
    if boxes is None or len(boxes) == 0:
        summary = "✅ No caries detected with current confidence threshold."
    else:
        detections = []
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls]
            detections.append(f"• {label.capitalize()} — confidence: {conf:.1%}")
        summary = f"🦷 {len(boxes)} region(s) detected:\n\n" + "\n".join(detections)
    
    return annotated_rgb, summary

with gr.Blocks(title="Caries Detector", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🦷 Caries Detection AI
    ### Powered by YOLOv8 · Trained on dental bitewing radiographs
    Upload a dental X-ray and the model will automatically detect and outline carious lesions.
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Dental X-ray")
            run_btn = gr.Button("Detect Caries", variant="primary", size="lg")
        with gr.Column():
            output_image = gr.Image(label="Detection Result")
            output_text = gr.Textbox(label="Detection Summary", lines=6)
    
    run_btn.click(
        fn=detect_caries,
        inputs=input_image,
        outputs=[output_image, output_text]
    )
    
    gr.Markdown("""
    ---
    *Model: YOLOv8n-seg · Dataset: 1,653 dental X-rays · mAP50: 0.813 · Built by Sina*
    """)

app.launch(share=False)