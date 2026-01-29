import io
import os
import urllib.parse
import hashlib
import requests
import gradio as gr
from PIL import Image
from io import BytesIO
import numpy as np


# ---- Configure your FastAPI base URL ----
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

IMG_HASH_CHECK = urllib.parse.urljoin(API_BASE + "/", "img_hash_check")
IMG_NEW_FN     = urllib.parse.urljoin(API_BASE + "/", "img_new_fn")
SAVE_IMG_INFO  = urllib.parse.urljoin(API_BASE + "/", "save_img_info")
IMG_URL_CHECK  = urllib.parse.urljoin(API_BASE + "/", "img_url_check")




MAX_IMAGE_SIZE = 2048


def resize_for_saving(img, max_image_size=MAX_IMAGE_SIZE):
    o_width, o_height = img.size
        
    
    if o_width > max_image_size and o_width > o_height:
        new_width = max_image_size
        new_height = int((max_image_size * o_height) / o_width)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    elif o_height > max_image_size and o_height > o_width:
        new_height = max_image_size
        new_width = int((max_image_size * o_width) / o_height)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img


def load_image(image_path):
    """
    loading image from filename
    """
    image = Image.open(image_path).convert('RGB')
    image = resize_for_saving(image, max_image_size=MAX_IMAGE_SIZE)
    return image


def load_image_from_url(image_url):
    """
    loading image from http request and resizing to the max allotable
    """
    headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    

    print(image_url)
    print("Big Devin")
    response = requests.get(image_url, headers=headers)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    image = resize_for_saving(image, max_image_size=MAX_IMAGE_SIZE)
    return image



def normalize_image_for_hash(img: Image.Image) -> Image.Image:
    """
    Normalize to a stable format before perceptual hash:
    - convert to RGB
    - handle weird modes (P, LA, RGBA, etc.)
    """
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img





def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phash(img: Image.Image, hash_size: int = 8) -> str:
    """
    Perceptual hash (pHash) implementation using DCT.
    Returns a hex string. Good for near-duplicate detection.
    """

    img = normalize_image_for_hash(img).convert("L")  # grayscale
    # pHash uses a larger resize than hash_size; common is 32x32
    img = img.resize((32, 32), Image.Resampling.LANCZOS)

    pixels = np.asarray(img, dtype=np.float32)

    # 2D DCT
    # implement DCT via scipy if available; otherwise use numpy FFT trick fallback
    try:
        from scipy.fftpack import dct
        dct_rows = dct(pixels, axis=0, norm="ortho")
        dct_2d = dct(dct_rows, axis=1, norm="ortho")
    except Exception:
        # fallback: slower/rougher but works without scipy
        import numpy.fft as fft
        dct_2d = fft.fft2(pixels).real

    # take top-left 8x8 (excluding DC term optionally)
    dct_lowfreq = dct_2d[:hash_size, :hash_size]
    # median excluding DC for stability
    dct_flat = dct_lowfreq.flatten()
    med = float(np.median(dct_flat[1:]))

    bits = (dct_lowfreq > med).astype(np.uint8).flatten()
    # pack bits into hex
    bit_string = "".join("1" if b else "0" for b in bits)
    return hex(int(bit_string, 2))[2:].zfill((hash_size * hash_size + 3) // 4)

def hash_image(img: Image.Image):
    data = img.tobytes()
    sha = sha256_bytes(data)
    p = phash(img)
    return {
        "sha256": sha,
        "phash": p,
        "content_length": len(data)
    }


def api_post(url: str, json: dict, params: dict | None = None, timeout: int = 15) -> dict:
    r = requests.post(url, json=json, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_get(url: str, params: dict | None = None, timeout: int = 10) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()



def check_then_maybe_save(image_url, user_id):

    if not image_url or not image_url.strip():
        return None, "Provide an image URL.", None

    image_url = image_url.strip()
    try:

        user_id = int(user_id)

        url_check = api_get(IMG_URL_CHECK, params={"image_url": image_url})
        if url_check.get("found"):
            filename = url_check.get("filename")

            preview  = load_image(filename)

            return {
                preview,
                f"✅ URL already exists in DB. filename={filename}",
                {"found": True, "via": "url_check", "filename": filename, "image_url": image_url},
                }
        
       
        # 1) Fetch + hash
        try: 
           preview = load_image_from_url(image_url)
        except:
            preview = None
        
        img_hash = hash_image(preview)
        sha = img_hash.get("sha256")
        
        # 3) Check sha256 + content_length
        #check_payload = {"sha256": sha, "content_length": clen}
        check_resp = api_post(IMG_HASH_CHECK, json=img_hash, params={"check_type": "sha256"})
        found = bool(check_resp.get("found"))  

        if found:
            return preview, f"✅ Found existing image (sha256+content_length match).sha = {sha[::12]}…", {
                "found": True,
                "sha256": img_hash["sha256"],
                "content_length": img_hash["content_length"],
                "phash": img_hash["phash"],
            }
       
        # 4) Ask server for new filename (your hook)
        new_fn = api_post(IMG_NEW_FN, json={})
        image_path = new_fn.get("filename") + ".jpg"
        #image_path = new_fn.get("path") or f"images/{filename}"
        
        
        img_hash["image_url"] = image_url
        img_hash["image_path"] = image_path
        img_hash["user_id"] = int(user_id)
        
        save_resp = api_post(SAVE_IMG_INFO, json=img_hash)
        
        status = save_resp.get("status", "unknown")
        image_id = save_resp.get("image_id") or save_resp.get("image_hash_id")
        
        
        return preview, f"🆕 Saved image. status={status} image_id={image_id} path={image_path}",{
             "found": False,
             "sha256": img_hash["sha256"],
             "content_length": img_hash["content_length"],
             "phash": img_hash["phash"],
             "saved": save_resp,
             "assigned_path": image_path
         }
    
    except Exception as e:
        return None, f"❌ Error: {type(e).__name__}: {e}", None



with gr.Blocks(title="Image Ingest + Dedupe") as demo:
    gr.Markdown("## Image ingest + sha256 dedupe\nChecks your FastAPI DB and saves if new.")

    with gr.Row():
        image_url = gr.Textbox(label="Image URL", placeholder="https://...")
        user_id = gr.Number(label="user_id", value=1, precision=0)
    
    run_btn = gr.Button("Check + Save", variant="primary")
    
    with gr.Row():
        img_preview = gr.Image(label="Preview", type="pil")
        status = gr.Textbox(label="Status", lines=4)
    
    debug = gr.JSON(label="Debug (hashes / response)", visible=True)
    
    run_btn.click(
        fn=check_then_maybe_save,
        inputs=[image_url, user_id],
        outputs=[img_preview, status, debug],
    )
    
if __name__ == "__main__":
    image_url="https://a.storyblok.com/f/176726/2000x2000/2c8b36a632/kitten-under-blanket_edited.jpg/m/1200x0"
    user_id = 10

    image_preview, status, debug = check_then_maybe_save(image_url, user_id)

    print(status)
    print(debug)

    demo.launch(server_name="0.0.0.0", server_port=7000, share=False, show_error=True, debug=True)

    #demo.launch(server_name="0.0.0.0", server_port=7000, share=False, show_error=True, debug=True)




