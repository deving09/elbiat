import os
import mimetypes
import urllib.parse
import requests
import gradio as gr


# -------------------------
# Config
# -------------------------

DEMO_DIR = "app/static/demo"

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


AUTH_TOKEN = urllib.parse.urljoin(DATA_BASE + "/", "auth/token")
SU_ENDPOINT = urllib.parse.urljoin(DATA_BASE + "/", "auth/signup")




#####
#
#   Demo Examples
#
#
#####


DEMO_CASES = [
    {
        "title": "1) Upload → Extract key info",
        "image": os.path.join(DEMO_DIR, "receipt.jpg"),
        "prompt": "Extract vendor, date, total, and list line items as JSON.",
        "response": 
        """
            ```json
			{
			  "vendor": "Target",
			  "date": "03/06/2016",
			  "total": "$251.83",
			  "lines": [
				{
				  "line_item": "OREO COOKIE",
				  "quantity": "2 @ $2.99 ea",
				  "price": "$5.98"
				},
				{
				  "line_item": "NATURAL ENER",
				  "quantity": "2 @ $5.89 FC",
				  "price": "$5.89"
				},
				{
				  "line_item": "MOTT'S FRTSN",
				  "quantity": "3 @ $1.50 ea",
				  "price": "$4.50"
				},
				{
				  "line_item": "SAVED S",
				  "quantity": "V8",
				  "price": "$4.19"
				},
				{
				  "line_item": "OSCAR MAYER",
				  "quantity": "2 @ $2.50 ea",
				  "price": "$3.07"
				},
				{
				  "line_item": "NAKED",
				  "quantity": "15% off",
				  "price": "$2.50"
				},
				{
				  "line..... 
        """,
        "feedback" :
            "Natval Ener instead of Natural Ener and there is only one purchase at 5.89. Saved S should be V8 and the quantity should be 1. Oscar Mayer has two line items one with quantity 1 for 2.50 and another for one quantity for 3.07. There are other errors."
    },
    {
        "title": "2) Upload → Describe + suggestions",
        "image": os.path.join(DEMO_DIR, "whiteboard.jpg"),
        "prompt": "Summarize the whiteboard and propose 3 next steps.",
        "response": """The whiteboard appears to outline a project or plan involving several key components:

			1. **Goal/Objectives:**
			   - Goal: Improve performance and user experience.
			   - Key areas include:
				 - Data points
				 - User interface
				 - Web application
				 - Data storage

			2. **Current Status:**
			   - Current data storage method.
			   - Current data processing.
			   - Current data visualization.

			3. **Future Steps:**
			   - Implement a new data model.
			   - Enhance user interface.
			   - Improve web application.
			   - Optimize data storage and processing.

			4. **Technologies/Tools:**
			   - Use of Node.js, React, and MongoDB.
			   - Tools for data visualization and user interface design.

			Next steps could involve:
			- Implementing the new data model.
			- Developing the enhanced user interface.
			- Optimizing the web application for performance.
			- Conducting a thorough testing and user feedback loop to ensure improvements.
        """,
        "feedback": "There is no mention of node.js, react, or MongoDB in the whiteboard.  the focus of the project is on data acquisition, model training, and task evaluation."
    },
    {
        "title": "3) Upload → QA over a chart",
        "image": os.path.join(DEMO_DIR, "chart.png"),
        "prompt": "What trend do you see? Provide 2 plausible explanations and 1 risk.",
        "response": """### Trend:
			The graph shows a general upward trend in the total desktop market share for Intel across the observed period, while AMD and NVIDIA experience fluctuations.

			### Explanation 1:
			**Intel's Dominance and Innovation:**
			Intel has consistently maintained a dominant market share, especially in the latter part of the observed period. This could be attributed to its continuous innovation and the widespread adoption of its processors in both consumer and enterprise markets. Intel's advancements in technology and manufacturing improvements have likely contributed to its sustained dominance.

			### Explanation 2:
			**AMD's Resurgence and Market Share Changes:**
			While AMD has seen fluctuations, particularly in the earlier part of the period, it has experienced a resurgence in recent quarters. This could be due to strategic investments in R&D, partnerships, and market expansion. AMD's focus on emerging markets and its ability to offer competitive pricing have also played a role in its recent growth.

			### Risk:
			**Intel's Market Dominance:**
			Despite its dominance, Intel's market share could be challenged by new entrants in the market, particularly from companies like Razer, ASUS, and Acer, which are gaining traction. Intel's reliance on older manufacturing technologies might pose a risk if these technologies become obsolete or less """,
       "feedback": "AMD does not appear to have a resurgence on this chart. " 
    },
]




# -------------------------
# Helpers
# -------------------------

def demo_load(case_idx: int):
    c = DEMO_CASES[case_idx]
    # return image to uploader + prompt to textbox
    return c["image"], c["prompt"], f"Loaded: **{c['title']}**", c["response"], c["feedback"]








def auth_headers(token: str | None):
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}



def _get_json(url: str, token: str | None = None, timeout: int = 30) -> dict:
    r = requests.get(url, headers=auth_headers(token), timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {url}: {r.text}")
    return r.json()

def _post_json(url: str, payload: dict, token: str | None = None, timeout: int = 60) -> dict:
    r = requests.post(url, json=payload, timeout=timeout, headers=auth_headers(token))
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {url}: {r.text}")
    return r.json()


def _post_multipart(url: str, data: dict, 
        file_field: str, file_path: str, 
        token: str | None = None, timeout: int = 120) -> dict:
    mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    with open(file_path, "rb") as f:
        files = {file_field: (os.path.basename(file_path), f, mime)}
        r = requests.post(url, data=data, files=files, headers=auth_headers(token), timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {url}: {r.text}")
    return r.json()


def _post_form(url: str, payload: dict, headers=None, timeout=30) -> dict:
    r = requests.post(url, data=payload, headers=headers or {}, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"detail": r.text}
    if r.status_code >= 400:
        raise ValueError(data.get("detail") or data)
    return data


# -------------------------
# Actions
# -------------------------


# IMPORTANT: replace this with your real handler
# It should accept (jwt, image_upload, image_url, prompt, chat_state, etc.)
# For demo, we’ll just call the same chat/run function you already have.
def demo_run(demo_image_path, demo_prompt, jwt):
    if not demo_image_path:
        return "❌ Pick a demo example first."

    # If your real pipeline requires login, you can either:
    # A) allow demo to run without saving to DB (recommended), OR
    # B) require login
    #
    # Here’s the recommended behavior:
    if not jwt:
        # run “local-only” inference path (no DB write) OR just explain
        return "🔒 Log in to run the full pipeline (save conversations). Demo is showing the setup."

    # Otherwise call your existing chat pipeline.
    # Example signature (adjust to your actual function):
    # result_text = run_chat(jwt, image_upload=demo_image_path, image_url=None, prompt=demo_prompt, ...)
    #
    # For now, return a placeholder:
    return f"✅ Would run pipeline on `{demo_image_path}` with prompt:\n\n{demo_prompt}"







def login_action(email: str, password: str):
    data = {"username": email.strip().lower(), "password": password}
    r = requests.post(AUTH_TOKEN, data=data, timeout=15)  # OAuth2PasswordRequestForm uses form-encoded
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {AUTH_TOKEN}: {r.text}")
    token = r.json()["access_token"]
    return token, {"email": email},  "✅ Logged in"

def signup_action(email: str, password: str, confirm: str):
    if not email or not password:
        return "❌ Email and password are required."
    if password != confirm:
        return "❌ Passwords do not match."
    if len(password.encode("utf-8")) > 72:
        return "❌ Password must be ≤ 72 bytes (bcrypt limit)."

    try:
        _ = _post_json(SU_ENDPOINT, {"email": email, "password": password})
        return "✅ Account created. Now log in."
    except Exception as e:
        return f"❌ Signup failed: {e}"


def logout_action():
    return None, None, gr.update(visible=True), gr.update(visible=False), ""


def update_visibility(jwt, user):
    authed = jwt is not None
    header = f"Signed in as **{user['email']}**" if authed and user else ""
    return (
        gr.update(visible=not authed),  # auth_view
        gr.update(visible=authed),      # app_view
        header,
    )


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


def save_convo_to_data_service(payload: dict, token:str | None = None) -> dict:
    #headers = {}
    #if token:
    #    headers["Authorization"] = f"Bearer {token}"

    #r = requests.post(CONVOS_ENDPOINT, json=payload, headers=headers, timeout=30)
    r = _post_json(CONVOS_ENDPOINT, payload, token=token, timeout=30)
    return r


def save_convo_action(
    token: str,
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
        "image_id": int(image_id),   # "user_id": int(user_id),
        "conversations": conversations,
        "model_name": model_name,
        "model_type": model_type,
        "task": (task or "general_vqa").strip(),
        "feedback": fb,
        "monetized": bool(monetized),
        "enabled": bool(enabled),
    }

    try:
        resp = save_convo_to_data_service(payload, token=token)
        return f"✅ Saved convo. convo_id={resp.get('convo_id')}", resp
    except Exception as e:
        return f"❌ {type(e).__name__}: {e}", None


def ensure_ingested_then_chat(
        token: str, 
        image_id: int,
        image_url: str, 
        upload_file,
        prompt: str, 
        history_state, 
        max_new_tokens: int):

    user_id = 1 # DEVIN DELETE SOON

    cleared_feedback = ""

    if not token:
       return None, image_id, history_state, "❌ Please log in first.", "",None, cleared_feedback
    
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
            return None, None, None, history_state, "❌ Choose URL OR Upload (not both).",None, cleared_feedback
        if not has_url and not has_upload:
            return None, None, None, history_state, "❌ Provide a URL or upload an image.",None, cleared_feedback

        # ingest
        if has_url:
            #ingest_resp = _post_json(INGEST_URL, {"user_id": int(user_id), "image_url": image_url}, timeout=60)
            ingest_resp = _post_json(INGEST_URL, {"image_url": image_url}, token=token, timeout=60)
        else:
            ingest_resp = _post_multipart(
                INGEST_UPLOAD,
                {}, #{"user_id": str(int(user_id))},
                "file",
                upload_path,
                token=token,
                timeout=120,
            )

        image_id = ingest_resp.get("image_id")
        preview = ingest_resp.get("image_path")
        if not image_id:
            return None, None, None, history_state, f"❌ Ingest failed: {ingest_resp}", None, cleared_feedback
    else:
        #preview = requests.get(IMAGE_FILE(image_id), timeout=15)["path"]
        #preview = requests.get(META_IMAGE(image_id), timeout=15).json()["image_path"]
        preview = _get_json(META_IMAGE(image_id), token=token, timeout=15).get("image_path")


    # 2) Chat
    if not prompt:
        return None, preview, image_id, history_state, "❌ Provide a prompt.",None, cleared_feedback

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
        return response, preview, image_id, history, status, out, cleared_feedback

    except Exception as e:
        return None, preview, image_id, history_state, f"❌ {type(e).__name__}: {e}", None, cleared_feedback

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
   
    token_state = gr.State(None)
    #jwt_state = gr.State(value=None)
    user_state = gr.State(value=None)
    
    with gr.Column(visible=True) as auth_view:
        gr.Markdown("## Welcome\nCreate an account or log in to continue.")

        with gr.Tabs():
            with gr.Tab("Log in"):
                li_email = gr.Textbox(label="Email", placeholder="you@example.com")
                li_pw = gr.Textbox(label="Password", type="password")
                li_btn = gr.Button("Log in", variant="primary")
                li_msg = gr.Markdown()

            with gr.Tab("Sign up"):
                su_email = gr.Textbox(label="Email", placeholder="you@example.com")
                su_pw = gr.Textbox(label="Password", type="password")
                su_pw2 = gr.Textbox(label="Confirm password", type="password")
                su_btn = gr.Button("Create account", variant="primary")
                su_msg = gr.Markdown()


      
    with gr.Column(visible=False) as app_view:
        header = gr.Markdown("")
        logout_btn = gr.Button("Log out")

        #with gr.Tabs() as tabs:
        with gr.Tab("Chat"):

            with gr.Row():
                #user_id = gr.Number(label="user_id", value=1, precision=0)
                max_new_tokens = gr.Slider(32, 1024, value=256, step=32, label="max_new_tokens", visible=JSON_VISIBLE)

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

            response = gr.Textbox(label="Model response", lines=20)
            chat_json = gr.JSON(label="Chat response (includes history)", visible=JSON_VISIBLE)

            thumbs = gr.Radio(["up", "down", "None"], value="None", label="Quick rating", visible=THUMBS_VISIBLE)
            feedback_text = gr.Textbox(label="Feedback (optional)", placeholder="What was wrong/right about the answer?", lines=5)
            task = gr.Textbox(label="Task", value="general_vqa")

            save_btn = gr.Button("Save convo + feedback", variant="primary")
            save_status = gr.Textbox(label="Save status", lines=2)
            save_json = gr.JSON(label="Save response")
            

        with gr.Tab("Demo"):
            demo_status = gr.Markdown("")
            with gr.Row():
                demo_image = gr.Image(label="Demo image", type="filepath", height=320)
                demo_prompt = gr.Textbox(label="Prompt", lines=6)

            with gr.Row():
                run_btn = gr.Button("Run Demo", variant="primary")

            gr.Markdown("### Examples")
            with gr.Row():
                ex0 = gr.Button(DEMO_CASES[0]["title"])
                ex1 = gr.Button(DEMO_CASES[1]["title"])
                ex2 = gr.Button(DEMO_CASES[2]["title"])

            demo_output = gr.Markdown("")

            dummy_response = gr.State(None)
            
            demo_feedback_text = gr.Textbox(label="Feedback (optional)", placeholder="What was wrong/right about the answer?", lines=5)
            demo_save_btn = gr.Button("Save convo + feedback", variant="primary")
            save_status = gr.Markdown("Save Status")


            ex0.click(lambda: demo_load(0), outputs=[demo_image, demo_prompt, demo_status, dummy_response, demo_feedback_text])
            ex1.click(lambda: demo_load(1), outputs=[demo_image, demo_prompt, demo_status, dummy_response, demo_feedback_text])
            ex2.click(lambda: demo_load(2), outputs=[demo_image, demo_prompt, demo_status, dummy_response, demo_feedback_text])


            r = p = iis = hs = ings = chat_j = gr.State(None)
            #p = gr.Markdown("", visible=False)
            #iis = gr.Markdown("", visible=False)

            run_btn.click(
                fn=ensure_ingested_then_chat,
                inputs=[token_state, gr.State(None), gr.State(None), demo_image, demo_prompt, gr.State(None), max_new_tokens],
                #outputs=[_, demo_image, _, _, _, _],
                outputs=[demo_output, p, iis, hs, ings, chat_j, demo_feedback_text],
            )
            
            def dummy_save(ts, do, dft):
                return "Feedback Saved"


            demo_save_btn.click(
                    fn=dummy_save,
                    inputs=[token_state, demo_output, demo_feedback_text],
                    outputs=[save_status]
                    )
        
        """
        token: str,
        image_id: int,
        image_url: str,
        upload_file,
        prompt: str,
        history_state,
        max_new_tokens: int): 
        """
    # Wire buttons
    su_btn.click(
        fn=signup_action,
        inputs=[su_email, su_pw, su_pw2],
        outputs=[su_msg],
    )

    li_btn.click(
        fn=login_action,
        inputs=[li_email, li_pw],
        outputs=[token_state, user_state, li_msg],
    ).then(
        fn=update_visibility,
        inputs=[token_state, user_state],
        outputs=[auth_view, app_view, header],
    )

    logout_btn.click(
        fn=logout_action,
        inputs=[],
        outputs=[token_state, user_state, auth_view, app_view, header],
    )

    image_url.change(fn=reset_image_state, inputs=[], outputs=[image_id_state, history_state, ingest_status])
    upload.change(fn=reset_image_state, inputs=[], outputs=[image_id_state, history_state, ingest_status])

    # Chat wiring
    chat_btn.click(
        fn=ensure_ingested_then_chat,
        inputs=[token_state, image_id_state, image_url, upload, prompt, history_state, max_new_tokens],
        outputs=[response, preview, image_id_state, history_state, ingest_status, chat_json, feedback_text],
    )

    save_btn.click(
        fn=save_convo_action,
        inputs=[token_state, image_id_state, prompt, response, feedback_text, thumbs, task],
        outputs=[save_status, save_json],
    )

 
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


if __name__ == "__main__":
    # Use localhost if you're SSH port-forwarding.
    #demo.launch(server_name="127.0.0.1", server_port=7086, show_error=True)
    demo.launch(server_name="127.0.0.1", server_port=7860) #, show_error=True, share=True)
