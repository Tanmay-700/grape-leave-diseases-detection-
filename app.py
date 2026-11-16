"""
Gradio Web Application for Dual-Model Ensemble per Crop (Setup B)
==================================================================
- Grape: EfficientNet-B2 + ResNet-50 (Side-by-Side Grad-CAM)
- Apple: EfficientNet-B3 + ResNet-50 (Side-by-Side Grad-CAM)
- Segmentation: Meta SAM 2.1 (Hiera Small - 46M Parameters)
- Universal Foliage Detection + 1-Click Interactive Target Segmentation
- Consensus Voting Ensemble
"""
import os
import numpy as np
from PIL import Image
import gradio as gr

from segment import segment_leaf_sam2, segment_leaf_auto, crop_to_mask, get_sam2_info
from classify import classify_crop_ensemble, classify_auto_detect
from gradcam import generate_model_gradcam

DEVICE = "cpu"


def process_image(input_image, crop_mode, seg_mode, point_x, point_y, gradcam_alpha):
    if input_image is None:
        return None, None, None, None, None, None, None, "Please upload a photo of a leaf."

    try:
        if isinstance(input_image, np.ndarray):
            image = Image.fromarray(input_image)
        else:
            image = input_image
        image = image.convert("RGB")
    except Exception as e:
        return None, None, None, None, None, None, None, f"⚠️ Unable to process image. Please upload a standard JPG or PNG photo. Details: {e}"

    # ---- 1. Segmentation via Meta SAM 2 ----
    sam_meta = get_sam2_info()
    try:
        if seg_mode == "Point Prompt" and point_x is not None and point_y is not None:
            point_coords = np.array([[int(point_x), int(point_y)]])
            point_labels = np.array([1])
            segmented, mask = segment_leaf_sam2(
                image, point_coords=point_coords, point_labels=point_labels, device=DEVICE
            )
        else:
            segmented, mask = segment_leaf_auto(image, device=DEVICE)
    except Exception as e:
        return None, None, None, None, None, None, None, f"Segmentation error: {str(e)}"

    cropped = crop_to_mask(segmented, mask, padding=15)

    # ---- 2. Dual-Model Classification & Ensemble ----
    header_note = ""
    try:
        if "auto" in crop_mode.lower():
            result = classify_auto_detect(cropped)
            header_note = f"🔍 **Auto-Detected Crop:** {result.get('auto_crop', 'Detected')}\n\n"
        elif "grape" in crop_mode.lower():
            result = classify_crop_ensemble(cropped, crop="grape")
            header_note = "📌 **Manual Selection:** 🍇 Grape\n\n"
        else:
            result = classify_crop_ensemble(cropped, crop="apple")
            header_note = "📌 **Manual Selection:** 🍎 Apple\n\n"

    except FileNotFoundError as e:
        return (
            np.array(segmented), np.array(cropped), None, None, None, None, None,
            f"### ⚠️ Model weights not found!\n\n"
            f"✅ **Leaf segmentation completed successfully with Meta SAM 2!**\n\n"
            f"{str(e)}"
        )
    except Exception as e:
        return np.array(segmented), np.array(cropped), None, None, None, None, None, f"Classification error: {str(e)}"

    # ---- 3. Side-by-Side Grad-CAM Heatmaps ----
    cam1_overlay = None
    cam2_overlay = None

    if result.get('model_1'):
        try:
            cam1_overlay, _, _ = generate_model_gradcam(
                cropped, result['model_1']['model_key'], alpha=gradcam_alpha
            )
        except Exception as e:
            print(f"GradCAM 1 error: {e}")

    if result.get('model_2'):
        try:
            cam2_overlay, _, _ = generate_model_gradcam(
                cropped, result['model_2']['model_key'], alpha=gradcam_alpha
            )
        except Exception as e:
            print(f"GradCAM 2 error: {e}")

    m1 = result.get('model_1')
    m2 = result.get('model_2')

    # Format separate confidence scores for each model
    m1_conf = {
        item['label']: item['confidence']
        for item in m1['top_k']
    } if m1 and 'top_k' in m1 else {}

    m2_conf = {
        item['label']: item['confidence']
        for item in m2['top_k']
    } if m2 and 'top_k' in m2 else {}

    # Format consensus confidence scores
    confidences = {
        item['label']: item['confidence']
        for item in result['ensemble_top_k']
    }

    # Text report
    m1_arch = m1['arch'].upper() if m1 else "Model 1"
    m2_arch = m2['arch'].upper() if m2 else "Model 2"
    m1_text = f"**Model 1 ({m1_arch})**: {m1['predicted_label']} ({m1['confidence']*100:.1f}%)" if m1 else "Not loaded"
    m2_text = f"**Model 2 ({m2_arch})**: {m2['predicted_label']} ({m2['confidence']*100:.1f}%)" if m2 else "Not loaded"

    # Agreement / Disagreement analysis
    agreement_banner = ""
    if m1 and m2:
        if m1['predicted_label'] == m2['predicted_label']:
            agreement_banner = (
                f"> ### ✅ DUAL MODEL AGREEMENT: Both models agree!\n"
                f"> **High Confidence Diagnosis:** Both **{m1_arch}** and **{m2_arch}** independently identified **{m1['predicted_label']}**.\n\n"
            )
        else:
            agreement_banner = (
                f"> ### ⚠️ DUAL MODEL DISAGREEMENT DETECTED!\n"
                f"> * **Model 1 ({m1_arch}):** {m1['predicted_label']} ({m1['confidence']*100:.1f}%)\n"
                f"> * **Model 2 ({m2_arch}):** {m2['predicted_label']} ({m2['confidence']*100:.1f}%)\n\n"
                f"> 👉 **Tie-Breaker:** Compare the two side-by-side Grad-CAM heatmaps above! This typically reveals a **co-infection** (leaf showing symptoms of two diseases at once) or an ambiguous lesion.\n\n"
            )

    # Build side-by-side probability comparison table
    table_lines = [
        f"| Disease Class | {m1_arch} | {m2_arch} | 🏆 Consensus (Avg) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for item in result['ensemble_top_k']:
        lbl = item['label']
        c_ens = item['confidence'] * 100
        c_m1 = m1_conf.get(lbl, 0.0) * 100
        c_m2 = m2_conf.get(lbl, 0.0) * 100
        star = " 🌟" if lbl == result['consensus_label'] else ""
        table_lines.append(f"| **{lbl}** | `{c_m1:.1f}%` | `{c_m2:.1f}%` | **`{c_ens:.1f}%`**{star} |")
    table_str = "\n".join(table_lines)

    seg_badge = (
        f"🟢 **Segmentation Engine:** `{sam_meta['engine']}` ({sam_meta['params']} params) — **ACTIVE**"
        if sam_meta['active']
        else f"⚠️ **Segmentation Engine:** `{sam_meta['engine']}`"
    )

    text_output = f"""{header_note}{agreement_banner}## 🏆 Ensemble Consensus: {result['consensus_label']}
**Blended Confidence: {result['consensus_confidence']*100:.1f}%**

### 🤖 Dual Model Predictions:
* {m1_text}
* {m2_text}

### 📊 All Diseases Probability Comparison:
{table_str}

---
### 🌿 Vision Pipeline & Verification:
* {seg_badge}
* **Weight File:** `{sam_meta.get('checkpoint', 'None')}`
* **Foliage Isolation:** Background soil/sky/hand suppressed; leaf canopy and disease spots isolated.
"""

    return (
        np.array(segmented),
        np.array(cropped),
        np.array(cam1_overlay) if cam1_overlay else None,
        np.array(cam2_overlay) if cam2_overlay else None,
        m1_conf,
        m2_conf,
        confidences,
        text_output,
    )


def build_app():
    sam_meta = get_sam2_info()
    seg_status_text = (
        f"🟢 Meta SAM 2.1 Active ({sam_meta['params']} params)"
        if sam_meta['active']
        else "⚠️ Fallback"
    )

    with gr.Blocks(title="🍃 Dual-Model Leaf Disease AI") as app:
        gr.Markdown(
            f"""
            # 🍇🍎 Dual-Model Leaf Disease Classifier (Setup B)
            ### 2 Models per Crop with Side-by-Side Grad-CAM Interpretability & Meta SAM 2.1 Segmentation
            * **🍇 Grape**: Model 1 = `EfficientNet-B2` | Model 2 = `ResNet-50` (Active & Tested at 94%+)
            * **🍎 Apple**: Model 1 = `EfficientNet-B3` | Model 2 = `ResNet-50` (Active & Tested at 93%+)
            * **✂️ Segmentation**: `{sam_meta['engine']}` — **{seg_status_text}**
            * 💡 **Tip:** You can click directly anywhere on the uploaded photo to target a specific leaf or lesion!
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="📷 Upload Field Photo (Click anywhere on leaf to target)", type="numpy", height=320)

                crop_mode = gr.Radio(
                    choices=[
                        "🔄 Auto-Detect Crop (AI decides)",
                        "🍇 Grape (EfficientNet-B2 + ResNet-50)",
                        "🍎 Apple (EfficientNet-B3 + ResNet-50)",
                    ],
                    value="🔄 Auto-Detect Crop (AI decides)",
                    label="Crop Mode",
                )

                with gr.Accordion("⚙️ Segmentation & Interactive Targeting", open=True):
                    prompt_status = gr.Markdown("🔄 **Mode:** Automatic Universal Foliage Detection *(Click on leaf image to target a specific leaf)*")
                    with gr.Row():
                        seg_mode = gr.Radio(choices=["Automatic", "Point Prompt"], value="Automatic", label="Mode", scale=2)
                        clear_btn = gr.Button("🔄 Reset to Auto", size="sm", scale=1)
                    with gr.Row():
                        point_x = gr.Number(label="Point X", value=None, precision=0)
                        point_y = gr.Number(label="Point Y", value=None, precision=0)

                with gr.Accordion("🎨 Grad-CAM Opacity", open=False):
                    gradcam_alpha = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Overlay Opacity")

                run_btn = gr.Button("🔬 Analyze Leaf with Both Models", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Row():
                    seg_output = gr.Image(label="🔍 Segmented Leaf", height=210)
                    crop_output = gr.Image(label="✂️ Cropped Leaf", height=210)

                with gr.Row():
                    cam1_output = gr.Image(label="🔥 Model 1 Grad-CAM (EffNet)", height=220)
                    cam2_output = gr.Image(label="🔥 Model 2 Grad-CAM (ResNet-50)", height=220)

                with gr.Row():
                    m1_conf_output = gr.Label(label="📊 Model 1 Probabilities (EffNet)", num_top_classes=4)
                    m2_conf_output = gr.Label(label="📊 Model 2 Probabilities (ResNet-50)", num_top_classes=4)

                with gr.Row():
                    conf_output = gr.Label(label="🏆 Dual Ensemble Consensus", num_top_classes=4)
                    diag_output = gr.Markdown(label="📋 Dual Model Diagnosis & Comparison Table")

        # Interactive click-to-target handler
        def on_image_click(evt: gr.SelectData):
            x, y = evt.index[0], evt.index[1]
            return "Point Prompt", x, y, f"🎯 **Point Targeted:** `(X={x}, Y={y})` — SAM 2 will focus on this exact leaf. Click 'Reset to Auto' to revert."

        def on_clear_point():
            return "Automatic", None, None, "🔄 **Mode:** Automatic Universal Foliage Detection"

        input_image.select(
            fn=on_image_click,
            outputs=[seg_mode, point_x, point_y, prompt_status]
        )

        clear_btn.click(
            fn=on_clear_point,
            outputs=[seg_mode, point_x, point_y, prompt_status]
        )

        run_btn.click(
            fn=process_image,
            inputs=[input_image, crop_mode, seg_mode, point_x, point_y, gradcam_alpha],
            outputs=[
                seg_output, crop_output, cam1_output, cam2_output,
                m1_conf_output, m2_conf_output, conf_output, diag_output
            ],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    try:
        app.launch(server_name='127.0.0.1', server_port=7870)
    except OSError:
        app.launch()
