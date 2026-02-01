import os
import urllib.parse
import requests
import gradio as gr
import mimetypes


API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")

INGEST_URL_ENDPOINT = urllib.parse.urljoin(API_BASE + "/", "images/ingest_url")
INGEST_UPLOAD_ENDPOINT = urllib.parse.urljoin(API_BASE + "/", "images/ingest_upload")




def ingest_from_url(user_id: int, image_url: str) -> dict:
    payload = {"user_id": int(user_id), "image_url": image_url.strip()}
    r = requests.post(INGEST_URL_ENDPOINT, json=payload, timeout=30)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {r.text}")
    return r.json()



def ingest_from_upload(user_id: int, uploaded_path: str) -> dict:
    # uploaded_path is a local temp filepath on the machine running Gradio
    with open(uploaded_path, "rb") as f:
        mime = mimetypes.guess_type(uploaded_path)[0] or "application/octet-stream"
        files = {"file": (os.path.basename(uploaded_path), f, mime)}
        data = {"user_id": str(int(user_id))}
        r = requests.post(INGEST_UPLOAD_ENDPOINT, data=data, files=files, timeout=60)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {r.text}")
    return r.json()


def ingest(user_id: int, image_url: str, uploaded_path: str):
    """
    Returns: (preview, status_text, response_json)
    preview is either the URL (for gr.Image) or the uploaded image file path.
    """
    image_url = (image_url or "").strip()
    has_url = len(image_url) > 0
    has_upload = uploaded_path is not None and str(uploaded_path).strip() != ""

    if has_url and has_upload:
        return None, "❌ Provide either a URL OR an upload (not both).", None

    if not has_url and not has_upload:
        return None, "❌ Provide a URL or upload an image.", None

    try:
        if has_url:
            resp = ingest_from_url(user_id, image_url)
            # For URL case, Gradio can preview directly from URL
            status = f"✅ URL ingested: status={resp.get('status')} image_id={resp.get('image_id')}"
            return image_url, status, resp

        # upload case
        resp = ingest_from_upload(user_id, uploaded_path)
        status = f"✅ Upload ingested: status={resp.get('status')} image_id={resp.get('image_id')}"
        # For upload, preview the uploaded file path
        return uploaded_path, status, resp

    except Exception as e:
        return None, f"❌ {type(e).__name__}: {e}", None



with gr.Blocks(title="Image Ingestion") as demo:
    gr.Markdown("## Image ingestion\nIngest either an image URL or an uploaded image into the backend DB+storage.")

    with gr.Row():
        user_id = gr.Number(label="user_id", value=1, precision=0)
        ingest_btn = gr.Button("Ingest", variant="primary")

    with gr.Row():
        image_url = gr.Textbox(label="Image URL (optional)", placeholder="https://...")
        upload = gr.Image(label="Upload image (optional)", type="filepath")

    preview = gr.Image(label="Preview")
    status = gr.Textbox(label="Status", lines=3)
    resp_json = gr.JSON(label="Backend response")

    ingest_btn.click(
        fn=ingest,
        inputs=[user_id, image_url, upload],
        outputs=[preview, status, resp_json],
    )

if __name__ == "__main__":
    # Bind to localhost if you're SSH port-forwarding; use 0.0.0.0 only if exposing directly.
    demo.launch(server_name="127.0.0.1", server_port=7086, show_error=True)


