# main_2.py
import io
import os
import json
import time
import base64
import logging
from typing import List

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import requests  # used to call Gemini/HTTP endpoint

app = FastAPI(title="Attribute Normalization (Gemini LLM backend)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
logging.basicConfig(level=logging.INFO)

# Environment variables (set these to your provider's values)
GEMINI_API_URL = os.environ.get("GEMINI_API_URL", None)  # e.g. "https://api.your-provider.com/v1/generate"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)

if not GEMINI_API_URL or not GEMINI_API_KEY:
    logging.warning("GEMINI_API_URL or GEMINI_API_KEY not set. LLM calls will fail until these are configured.")


def image_to_base64_jpeg(file_bytes: bytes, max_side=800) -> str:
    """Downscale image and return base64 JPEG (keeps payload reasonable)."""
    im = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    w, h = im.size
    scale = 1.0
    if max(w,h) > max_side:
        scale = max_side / max(w,h)
        im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    b = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b

def build_gemini_prompt(title: str, description: str, images_b64: List[str]):
    """
    Build a careful instruction prompt for a multimodal Gemini model.
    We instruct the model to ONLY respond with valid JSON in the exact schema:
    {
      "final_category": {"value": <string>, "confidence": <0..1>},
      "per_label": [[label, score], ...],
      "images": [{"filename": "img1.jpg", "colors": [{"rgb":[r,g,b],"mapped_color":"maroon","confidence":0.9}], "notes":"..."}, ...],
      "conflict_flag": true|false
    }
    """
    header = (
        "You are an assistant that analyzes e-commerce product listings. "
        "You will be given a product title, an optional description, and one or more images encoded in base64 JPEG. "
        "Your job: identify the product's most likely canonical category (one of: saree, lehenga, kurti, jeans, t-shirt, dress, blouse, jacket, men's accessory), "
        "return per-label scores for these canonical categories, extract dominant colors for each image (RGB and normalized color name), and decide whether there is a mismatch between image and text. "
        "IMPORTANT: output ONLY valid JSON and NOTHING else. Do not include any commentary or extra fields. Use numbers for confidences (0..1). Use array for per_label in descending score order."
    )

    labels_str = ", ".join(["saree","lehenga","kurti","jeans","t-shirt","dress","blouse","jacket","men's accessory"])
    prompt = f"{header}\n\nPRODUCT TITLE: {title}\n\nDESCRIPTION: {description}\n\nCATEGORIES: {labels_str}\n\nIMAGES: There are {len(images_b64)} images. Each image is base64 JPEG. For each image compute: dominant colors (up to 3) with RGB triplets, map each to one of [maroon, red, blue, navy, green, olive, yellow, pink, orange, brown, black, white, gray, beige] if possible, and provide a confidence for the mapping.\n\nReturn JSON schema EXACTLY like:\n{{\n  \"final_category\": {{\"value\": \"<label>\", \"confidence\": 0.0}},\n  \"per_label\": [[\"label\", 0.0], ...],\n  \"images\": [{{\"filename\":\"img_1\",\"colors\":[{{\"rgb\":[r,g,b],\"mapped_color\":\"maroon\",\"confidence\":0.9}}], \"notes\":\"optional\"}}],\n  \"conflict_flag\": false\n}}\n\nStart images now.\n\n"

    # embed images as small base64 placeholders with short names so model can refer to them
    for i, b64 in enumerate(images_b64):
        prompt += f"IMAGE_{i+1}_NAME: img_{i+1}.jpg\nIMAGE_{i+1}_BASE64: (base64 data omitted in prompt for brevity)\n"
        # NOTE: some providers allow attaching binary separately; others accept base64 in prompt.
        # To keep prompt length reasonable we instruct model image names, and will attach base64 in multipart for providers that allow.
        # But include a short sample of the first few bytes so model knows it's an image:
        prompt += f"IMAGE_{i+1}_SIZE: approx base64 length {len(b64)}\n\n"

    prompt += "\nNow output the JSON only.\n"

    return prompt

def call_gemini(prompt: str, images_b64: List[str], timeout=60):
    """
    Generic HTTP call to a Gemini-like endpoint. This is a placeholder wrapper.
    Adapt to your provider:
      - Google Vertex AI: use google-cloud-aiplatform client with instances including base64 images.
      - Other vendors: use appropriate REST schema.
    Expected response: text body containing the JSON described in the prompt.
    """
    if not GEMINI_API_URL or not GEMINI_API_KEY:
        raise RuntimeError("Gemini endpoint or API key not configured (GEMINI_API_URL / GEMINI_API_KEY).")

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Minimal HTTP contract: send prompt and images as payload (provider-specific)
    payload = {
        "prompt": prompt,
        # attach images as separate field so provider can accept them if they support multimodal
        "images_base64": images_b64,
        "max_tokens": 1200,
        "temperature": 0.0
    }

    resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def parse_llm_json(text: str):
    """
    Some LLMs may respond with surrounding text. Try to locate the JSON substring.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # try to find first '{' and last '}' and parse
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except Exception as e:
                logging.exception("Failed to parse JSON from LLM response")
                raise e
    raise ValueError("Could not extract JSON from LLM response")

# --- analyze endpoint ---
@app.post("/analyze")
async def analyze(title: str = Form(...), description: str = Form(""), pipeline: str = Form("cv"), files: List[UploadFile] = File(...)):
    t0 = time.time()
    # read and downscale images, convert to base64
    images_b64 = []
    raw_images_meta = []
    for f in files:
        content = await f.read()
        b64 = image_to_base64_jpeg(content, max_side=800)
        images_b64.append(b64)
        raw_images_meta.append({"filename": f.filename, "size_bytes": len(content)})

    if pipeline == "cv":
        # If you want to keep CV behavior inside the same server, call your existing CV functions here.
        # For the purpose of this file, we return an error directing user to use main_improved.py or set pipeline=llm.
        return {"error": "CV pipeline not implemented in main_2.py. Start main_improved.py for CV. Or call with pipeline=llm."}

    # pipeline == "llm"
    prompt = build_gemini_prompt(title, description, images_b64)

    try:
        llm_text = call_gemini(prompt, images_b64, timeout=120)
        # parse JSON from LLM output
        parsed = parse_llm_json(llm_text)
        # minimal validation & normalization
        # ensure keys exist
        if "final_category" not in parsed:
            raise ValueError("LLM JSON missing final_category")
        # add pipeline metadata, timing, and copy filenames into images entries
        parsed["pipeline"] = "llm"
        parsed["timing_sec"] = round(time.time() - t0, 3)
        # ensure image entries exist and attach filenames if missing
        if "images" in parsed and isinstance(parsed["images"], list):
            for i, img_entry in enumerate(parsed["images"]):
                if "filename" not in img_entry:
                    img_entry["filename"] = raw_images_meta[i]["filename"] if i < len(raw_images_meta) else f"img_{i+1}.jpg"
        else:
            parsed["images"] = []
            for i, meta in enumerate(raw_images_meta):
                parsed["images"].append({"filename": meta["filename"], "colors": [], "notes":"no image info returned"})
        # ensure per_label is in desired format (list of [label, score])
        if "per_label" in parsed:
            # optionally normalize scores to 0..1
            try:
                arr = [float(x[1]) for x in parsed["per_label"]]
                # if not normalized, softmax them
                tot = sum(arr)
                if tot <= 0 or max(arr) > 1.0001:
                    # apply softmax
                    import math
                    exps = [math.exp(v) for v in arr]
                    s = sum(exps) or 1.0
                    normed = [e/s for e in exps]
                    parsed["per_label"] = [[parsed["per_label"][i][0], normed[i]] for i in range(len(normed))]
            except Exception:
                pass

        return parsed

    except Exception as e:
        logging.exception("LLM call failed")
        # safe fallback: return an error structure, or optionally call CV pipeline here as fallback
        return {"error": "LLM call failed: " + str(e), "pipeline": "llm", "timing_sec": round(time.time()-t0,3)}

# Correction endpoint (same as previous)
@app.post("/correct")
async def correct(original_title: str = Form(...), corrected_category: str = Form(...), notes: str = Form(""), filename: str = Form("")):
    rec = {
        "time": time.time(),
        "original_title": original_title,
        "corrected_category": corrected_category,
        "notes": notes,
        "filename": filename
    }
    with open("corrections.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return {"status":"ok", "logged": rec}
