import os
import mimetypes
import urllib.parse
import requests
import gradio as gr


# -------------------------
# Config
# -------------------------
DATA_BASE = os.environ.get("DATA_BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL_BASE = os.environ.get("MODEL_BASE", "http://127.0.0.1:9000").rstrip("/")

INGEST_URL = urllib.parse.urljoin(DATA_BASE + "/", "images/ingest_url")
INGEST_UPLOAD = urllib.parse.urljoin(DATA_BASE + "/", "images/ingest_upload")
IMAGE_FILE = lambda image_id: urllib.parse.urljoin(DATA_BASE + "/", f"images/{image_id}/file")
META_IMAGE = lambda image_id : urllib.parse.urljoin(DATA_BASE + "/", f"images/{image_id}/meta")

CHAT = urllib.parse.urljoin(MODEL_BASE + "/", "chat/internvl2_5_2b")
CONVOS_ENDPOINT = urllib.parse.urljoin(DATA_BASE + "/", "convos")


JSON_VISIBLE = False
THUMBS_VISIBLE = False

# -------------------------
# Helpers
# -------------------------


def _post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {url}: {r.text}")
    return r.json()


def _post_multipart(url: str, data: dict, file_field: str, file_path: str, timeout: int = 120) -> dict:
    mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    with open(file_path, "rb") as f:
        files = {file_field: (os.path.basename(file_path), f, mime)}
        r = requests.post(url, data=data, files=files, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {url}: {r.text}")
    return r.json()


# -------------------------
# Actions
# -------------------------
def ingest_action(user_id: int, image_url: str, upload_file):
    """
    upload_file comes from gr.File -> either a path string or a dict with {'path': ...}
    Returns: (image_preview, image_id_state, status, ingest_json, chat_history_state)
    """
    image_url = (image_url or "").strip()

    # normalize upload_file
    upload_path = None
    if upload_file:
        if isinstance(upload_file, str):
            upload_path = upload_file
        elif isinstance(upload_file, dict) and "path" in upload_file:
            upload_path = upload_file["path"]

    has_url = bool(image_url)
    has_upload = bool(upload_path)

    if has_url and has_upload:
        return None, None, "❌ Choose URL OR Upload (not both).", None, None
    if not has_url and not has_upload:
        return None, None, "❌ Provide a URL or upload a file.", None, None

    try:
        if has_url:
            resp = _post_json(INGEST_URL, {"user_id": int(user_id), "image_url": image_url}, timeout=60)
            image_id = resp.get("image_id")
            # Preview: let Gradio render the URL directly OR use canonical endpoint if you prefer
            preview = image_url

        else:
            resp = _post_multipart(INGEST_UPLOAD, {"user_id": str(int(user_id))}, "file", upload_path, timeout=120)
            image_id = resp.get("image_id")
            # Preview: show canonical stored image via data service endpoint
            #preview = IMAGE_FILE(image_id)
            preview = resp.get("image_path")

        status = f"✅ Ingested ({resp.get('status')}), image_id={image_id}"
        return preview, image_id, status, resp, None  # reset chat history after new image

    except Exception as e:
        return None, None, f"❌ {type(e).__name__}: {e}", None, None


def chat_action(user_id: int, image_id: int, prompt: str, history_state, max_new_tokens: int):
    """
    Returns: (response_text, updated_history_state, chat_json)
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return "❌ Provide a prompt.", history_state, None

    payload = {
        "prompt": prompt,
        "image_id": int(image_id) if image_id else None,
        "history": history_state,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "return_history": True,
    }

    try:
        out = _post_json(CHAT, payload, timeout=180)
        response = out.get("response", "")
        history = out.get("history", None)
        return response, history, out
    except Exception as e:
        return f"❌ {type(e).__name__}: {e}", history_state, None


def save_convo_to_data_service(payload: dict) -> dict:
    r = requests.post(CONVOS_ENDPOINT, json=payload, timeout=30)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {CONVOS_ENDPOINT}: {r.text}")
    return r.json()


def save_convo_action(
    user_id: int,
    image_id: int,
    prompt: str,
    model_response: str,
    feedback_text: str,
    thumbs: str,
    task: str,
    model_name: str = "internvl2.5_2B",
    model_type: str = "vlm",
    monetized: bool = True,
    enabled: bool = True,
):
    if not image_id:
        return "❌ No image_id. Ingest an image first.", None

    prompt = (prompt or "").strip()
    model_response = (model_response or "").strip()

    if not prompt or not model_response:
        return "❌ Need both prompt and model response before saving.", None

    # Training-style format you showed earlier
    conversations = [
        {"from": "human", "value": f"<image>\n{prompt}" if "<image>" not in prompt else prompt},
        {"from": "gpt", "value": model_response},
    ]

    # Combine thumbs + free text feedback into one field (simple + searchable)
    fb = (feedback_text or "").strip()
    if thumbs and thumbs != "None":
        fb = f"[thumbs={thumbs}] " + fb

    payload = {
        "image_id": int(image_id),
        "user_id": int(user_id),
        "conversations": conversations,
        "model_name": model_name,
        "model_type": model_type,
        "task": (task or "general_vqa").strip(),
        "feedback": fb,
        "monetized": bool(monetized),
        "enabled": bool(enabled),
    }

    try:
        resp = save_convo_to_data_service(payload)
        return f"✅ Saved convo. convo_id={resp.get('convo_id')}", resp
    except Exception as e:
        return f"❌ {type(e).__name__}: {e}", None


def ensure_ingested_then_chat(
        user_id: int, 
        image_id: int,
        image_url: str, 
        upload_file,
        prompt: str, 
        history_state, 
        max_new_tokens: int):
    
    image_url = (image_url or "").strip()
    prompt = (prompt or "").strip()

    # normalize upload_file -> path
    upload_path = None
    if upload_file:
        if isinstance(upload_file, str):
            upload_path = upload_file
        elif isinstance(upload_file, dict) and "path" in upload_file:
            upload_path = upload_file["path"]

    # 1) Ingest if needed
    if not image_id:
        has_url = bool(image_url)
        has_upload = bool(upload_path)

        if has_url and has_upload:
            return None, None, None, history_state, "❌ Choose URL OR Upload (not both).", None
        if not has_url and not has_upload:
            return None, None, None, history_state, "❌ Provide a URL or upload an image.", None

        # ingest
        if has_url:
            ingest_resp = _post_json(INGEST_URL, {"user_id": int(user_id), "image_url": image_url}, timeout=60)
        else:
            ingest_resp = _post_multipart(
                INGEST_UPLOAD,
                {"user_id": str(int(user_id))},
                "file",
                upload_path,
                timeout=120,
            )

        image_id = ingest_resp.get("image_id")
        preview = ingest_resp.get("image_path")
        if not image_id:
            return None, None, None, history_state, f"❌ Ingest failed: {ingest_resp}", None
    else:
        #preview = requests.get(IMAGE_FILE(image_id), timeout=15)["path"]
        preview = requests.get(META_IMAGE(image_id), timeout=15).json()["image_path"]


    # 2) Chat
    if not prompt:
        return None, preview, image_id, history_state, "❌ Provide a prompt.", None

    chat_payload = {
        "prompt": prompt,
        "image_id": int(image_id),
        "history": history_state,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "return_history": True,
    }

    try:
        out = _post_json(CHAT, chat_payload, timeout=180)
        response = out.get("response", "")
        history = out.get("history", None)

        # return:
        # - updated image_id_state
        # - updated history_state
        # - status text (or hidden response)
        # - full json for debugging (optional)
        status = "✅ Ran chat"  # keep minimal; you can include image_id if you want
        return response, preview, image_id, history, status, out

    except Exception as e:
        return response, preview, image_id, history_state, f"❌ {type(e).__name__}: {e}", None

def reset_image_state():
    return None, None, ""

# -------------------------
# UI
# -------------------------
with gr.Blocks(title="InternVL Chat (Data + Model Services)") as demo:
    gr.Markdown(
        """
# InternVL2.5-2B Chat
Workflow:
1) Ingest image (URL or Upload) → Data service
2) Chat with InternVL2.5-2B using `image_id` → Model service
"""
    )

    
    with gr.Row():
        user_id = gr.Number(label="user_id", value=1, precision=0)
        max_new_tokens = gr.Slider(32, 1024, value=256, step=32, label="max_new_tokens")

    with gr.Row():
        image_url = gr.Textbox(label="Image URL (optional)", placeholder="https://...")
        upload = gr.File(label="Upload image (optional)")  # more reliable than gr.Image for multipart

    #ingest_btn = gr.Button("Ingest Image", variant="primary")

    with gr.Row():
        preview = gr.Image(label="Preview", height=320, width=320)
        ingest_status = gr.Textbox(label="Ingest status", lines=2, visible=JSON_VISIBLE)

    ingest_json = gr.JSON(label="Ingest response", visible=JSON_VISIBLE)

    # state
    image_id_state = gr.State(None)
    history_state = gr.State(None)

    # Display image_id explicitly
    image_id_box = gr.Number(label="Current image_id", value=None, precision=0, visible=JSON_VISIBLE)

    prompt = gr.Textbox(label="Prompt", placeholder="e.g. What is in this image?", lines=4)
    chat_btn = gr.Button("Chat", variant="primary")

    response = gr.Textbox(label="Model response", lines=6)
    chat_json = gr.JSON(label="Chat response (includes history)", visible=JSON_VISIBLE)

    thumbs = gr.Radio(["up", "down", "None"], value="None", label="Quick rating", visible=THUMBS_VISIBLE)
    feedback_text = gr.Textbox(label="Feedback (optional)", placeholder="What was wrong/right about the answer?", lines=3)
    task = gr.Textbox(label="Task", value="general_vqa")

    save_btn = gr.Button("Save convo + feedback", variant="primary")
    save_status = gr.Textbox(label="Save status", lines=2)
    save_json = gr.JSON(label="Save response")
    
    # Ingest wiring
    """ingest_btn.click(
        fn=ingest_action,
        inputs=[user_id, image_url, upload],
        outputs=[preview, image_id_state, ingest_status, ingest_json, history_state],
    ).then(
        fn=lambda image_id: image_id,
        inputs=[image_id_state],
        outputs=[image_id_box],
    )
    """
    image_url.change(fn=reset_image_state, inputs=[], outputs=[image_id_state, history_state, ingest_status])
    upload.change(fn=reset_image_state, inputs=[], outputs=[image_id_state, history_state, ingest_status])

    # Chat wiring
    chat_btn.click(
        fn=ensure_ingested_then_chat,
        inputs=[user_id, image_id_state, image_url, upload, prompt, history_state, max_new_tokens],
        outputs=[response, preview, image_id_state, history_state, ingest_status, chat_json],
    )

    save_btn.click(
        fn=save_convo_action,
        inputs=[user_id, image_id_state, prompt, response, feedback_text, thumbs, task],
        outputs=[save_status, save_json],
    )


if __name__ == "__main__":
    # Use localhost if you're SSH port-forwarding.
    #demo.launch(server_name="127.0.0.1", server_port=7086, show_error=True)
    demo.launch(server_name="0.0.0.0", server_port=7086, show_error=True, share=True)
