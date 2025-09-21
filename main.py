# main_improved.py
import io
import os
import base64
import time
import json
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from sklearn.cluster import KMeans

import cv2
from skimage import color as skcolor

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Basic app + logging ---
app = FastAPI(title="Attribute Normalization (improved)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
logging.basicConfig(level=logging.INFO)

# --- Device & CLIP model load (one-time) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Canonical categories and palette ---
# Expand categories & synonyms (map to canonical)
CANONICAL = {
    "saree": ["saree", "sari", "half-saree"],
    "lehenga": ["lehenga", "lehnga"],
    "kurti": ["kurti", "kurta", "top", "anarkali"],
    "jeans": ["jeans"],
    "t-shirt": ["t-shirt", "tshirt", "tee"],
    "dress": ["dress"],
    "blouse": ["blouse"],
    "jacket": ["jacket"],
    "men's accessory": ["men's accessory", "wallet", "belt"]
}
CATEGORIES = list(CANONICAL.keys())

# palette name-> RGB for display and mapping
COLOR_PALETTE = {
    "maroon": (128, 0, 0), "red": (220,20,60), "blue": (30,144,255), "navy": (0,0,128),
    "green": (34,139,34), "olive": (128,128,0), "yellow": (255,215,0),
    "pink": (255,105,180), "orange": (255,140,0), "brown": (165,42,42),
    "black": (0,0,0), "white": (255,255,255), "gray": (128,128,128), "beige": (245,245,220)
}
# Precompute palette LAB values
_PALETTE_RGB_ARR = np.array([COLOR_PALETTE[k] for k in COLOR_PALETTE], dtype=np.uint8)[None, :, :]  # (1,M,3)
PALETTE_LAB = skcolor.rgb2lab(_PALETTE_RGB_ARR / 255.0)[0]  # (M,3)
PALETTE_NAMES = list(COLOR_PALETTE.keys())

# --- Prompt templates for label embeddings ---
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a product photo of a {}",
    "a studio photo of {}",
    "a garment: {}",
    "a picture of {}"
]

# Cache label embeddings computed with prompt templates
LABEL_EMBS = None

def compute_label_embeddings(labels: List[str]) -> np.ndarray:
    """Compute prompt-averaged CLIP text embeddings for each label (cacheable)."""
    global LABEL_EMBS
    if LABEL_EMBS is not None:
        return LABEL_EMBS
    all_prompts = []
    offsets = []
    for lbl in labels:
        start = len(all_prompts)
        for t in PROMPT_TEMPLATES:
            all_prompts.append(t.format(lbl))
        offsets.append((start, len(PROMPT_TEMPLATES)))
    # use processor+model to get text embeddings
    inputs = processor(text=all_prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        txt_feats = model.get_text_features(**inputs)  # (N_prompts, D)
        txt_feats = txt_feats.cpu().numpy()
    averaged = []
    for st, cnt in offsets:
        emb = txt_feats[st:st+cnt].mean(axis=0)
        emb = emb / np.linalg.norm(emb)
        averaged.append(emb)
    LABEL_EMBS = np.vstack(averaged)  # (num_labels, D)
    return LABEL_EMBS

# --- Image / text embedding helpers ---
def image_to_pil(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def get_clip_image_embedding(pil: Image.Image) -> np.ndarray:
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def get_clip_text_embedding(text: str) -> np.ndarray:
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

# --- GrabCut foreground (robust-ish) ---
def grabcut_foreground(pil_img: Image.Image, max_dim=1024, rect_margin_ratio=0.08, iter_count=5) -> Tuple[Image.Image, np.ndarray]:
    """
    Returns cropped foreground PIL image and mask (uint8: 1 foreground, 0 background),
    mask is aligned to cropped image coordinates.
    """
    img = np.array(pil_img)
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    rect = (
        int(w * rect_margin_ratio),
        int(h * rect_margin_ratio),
        int(w * (1 - rect_margin_ratio)) - int(w * rect_margin_ratio),
        int(h * (1 - rect_margin_ratio)) - int(h * rect_margin_ratio)
    )  # x,y,w,h

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        # bounding box of mask
        coords = np.column_stack(np.where(mask2 > 0))
        if coords.size == 0:
            # fallback: full image
            crop = Image.fromarray(img)
            full_mask = np.ones((h,w), dtype=np.uint8)
            return crop, full_mask
        y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0) + 1
        cropped = img[y0:y1, x0:x1]
        cropped_mask = mask2[y0:y1, x0:x1]
        return Image.fromarray(cropped), cropped_mask
    except Exception as e:
        # robust fallback: return whole image
        logging.exception("GrabCut failed: %s", e)
        full_mask = np.ones((h,w), dtype=np.uint8)
        return Image.fromarray(img), full_mask

# --- Masked LAB clustering for dominant colors ---
def get_dominant_colors_lab(pil_crop: Image.Image, mask: np.ndarray, n_colors=3, sample_limit=12000) -> List[Dict[str, Any]]:
    """
    Cluster foreground pixels in LAB space and map centers to canonical palette.
    Returns list of dicts: {"rgb": (r,g,b), "lab": (L,a,b), "mapped_color": name, "confidence": float}
    """
    arr = np.array(pil_crop)  # HxWx3
    h, w = arr.shape[:2]
    flat = arr.reshape(-1, 3)
    if mask is not None:
        m = mask.reshape(-1)
        pixels = flat[m==1]
        if pixels.size == 0:
            pixels = flat
    else:
        pixels = flat

    if pixels.shape[0] > sample_limit:
        idx = np.random.choice(pixels.shape[0], sample_limit, replace=False)
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels

    # convert to LAB
    lab = skcolor.rgb2lab((pixels_sample / 255.0).reshape(-1,3).reshape(-1,3))
    k = min(n_colors, max(1, pixels_sample.shape[0]//50))
    if k <= 0:
        k = 1
    kmeans = KMeans(n_clusters=k, random_state=0).fit(lab)
    centers_lab = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    order = np.argsort(-counts)
    out = []
    total = counts.sum()
    for i in order:
        center_lab = centers_lab[i]
        # convert back approx to rgb for output (through skimage inverse)
        rgb_arr = skcolor.lab2rgb(center_lab.reshape(1,1,3)).reshape(3,)
        rgb = tuple((rgb_arr * 255).clip(0,255).astype(int).tolist())
        # map to palette via LAB distance
        d = np.linalg.norm(PALETTE_LAB - center_lab.reshape(1,3), axis=1)
        idx_min = int(np.argmin(d))
        palette_name = PALETTE_NAMES[idx_min]
        # compute confidence as normalized closeness
        # maximum plausible LAB distance ~ 150; scale accordingly
        conf = max(0.0, 1.0 - (float(d[idx_min]) / 70.0))
        weight = float(counts[i]) / float(total)
        out.append({"rgb": rgb, "lab": center_lab.tolist(), "mapped_color": palette_name, "weight": weight, "confidence": float(conf * weight)})
    return out

# --- Heuristics: simple rules to help saree vs lehenga vs kurti ---
def compute_mask_heuristics(mask: np.ndarray) -> Dict[str, float]:
    """
    Given a binary mask (H,W) of the person+garment, compute simple heuristics:
      - mask_area_ratio: foreground area / total pixels
      - bbox_aspect_ratio: height/width of bounding box (tall narrow suggests saree)
      - bottom_contact_ratio: proportion of mask pixels touching bottom (saree tends to trail)
      - border_edge_density: edge density near mask border (sarees often have decorative border)
    Returns dict with heuristic scores for categories (0..1).
    """
    h, w = mask.shape[:2]
    total = h*w
    area = mask.sum()
    if area == 0:
        return {c: 0.0 for c in CATEGORIES}
    mask_area_ratio = area / total

    coords = np.column_stack(np.where(mask > 0))
    y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0) + 1
    bbox_h = max(1, y1 - y0); bbox_w = max(1, x1 - x0)
    bbox_aspect = bbox_h / bbox_w

    # bottom contact: fraction of mask pixels at bottom 10% of bbox (trailing)
    bottom_thresh = int(y1 - max(1, bbox_h * 0.1))
    bottom_contact = np.sum(mask[bottom_thresh:y1, x0:x1]) / max(1, np.sum(mask[y0:y1, x0:x1]))

    # border-edge density: compute Canny edges and measure edge density on a 6-px band on mask border
    edges = cv2.Canny((mask*255).astype(np.uint8), 50, 150)
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=6)
    border = mask.astype(np.uint8) - eroded
    border_area = border.sum()
    if border_area > 0:
        edge_density = edges[border==1].sum() / 255.0 / border_area
    else:
        edge_density = 0.0

    # heuristics to categories (hand tuned)
    heur = {}
    # saree likes tall bbox, high bottom_contact, and border_edge_density
    saree_score = 0.0
    saree_score += min(1.0, max(0.0, (bbox_aspect - 1.2) / 2.0)) * 0.6
    saree_score += min(1.0, bottom_contact * 4.0) * 0.3
    saree_score += min(1.0, edge_density * 3.0) * 0.1
    # lehenga: often wide skirt, lower bottom_contact but full lower mask
    lehenga_score = 0.0
    lehenga_score += min(1.0, max(0.0, (1.3 - bbox_aspect) / 1.5)) * 0.6
    lehenga_score += min(1.0, (mask_area_ratio * 4.0)) * 0.3
    lehenga_score += min(1.0, edge_density * 1.0) * 0.1
    # kurti/kurta: shorter vertical extent, smaller mask area
    kurti_score = 0.0
    kurti_score += min(1.0, max(0.0, (1.2 - bbox_aspect) / 1.2)) * 0.6
    kurti_score += min(1.0, (mask_area_ratio * 2.0)) * 0.3

    for c in CATEGORIES:
        if c == "saree":
            heur[c] = float(np.clip(saree_score, 0.0, 1.0))
        elif c == "lehenga":
            heur[c] = float(np.clip(lehenga_score, 0.0, 1.0))
        elif c in ["kurti", "kurta"]:
            heur[c] = float(np.clip(kurti_score, 0.0, 1.0))
        else:
            heur[c] = 0.0
    return heur

# --- softmax / calibration helpers ---
def sims_to_probs(sims: np.ndarray, temperature=0.07) -> np.ndarray:
    exps = np.exp(sims / temperature)
    probs = exps / (exps.sum() + 1e-9)
    return probs

def normalize_array(a: np.ndarray):
    a = np.array(a, dtype=float)
    s = a.sum()
    if s == 0:
        return (a * 0.0)
    return a / s

# --- utility: overlay mask on original image (base64) for frontend debugging ---
def mask_overlay_datauri(original_pil: Image.Image, mask: np.ndarray, alpha=0.45) -> str:
    # returns data URI PNG
    img = original_pil.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255,0,0,0))
    # mask may be for crop; pad to original size if needed. For simplicity, mask expected same size as image.
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    red = Image.new("RGBA", img.size, (255,0,0, int(255*alpha)))
    overlay.paste(red, (0,0), mask_img)
    out = Image.alpha_composite(img, overlay)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# --- logging corrections to file ---
CORRECTIONS_LOG = "corrections.jsonl"
def log_correction(record: dict):
    with open(CORRECTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# --- Main analyze endpoint ---
@app.post("/analyze")
async def analyze(title: str = Form(...), description: str = Form(""), files: List[UploadFile] = File(...)):
    t0 = time.time()
    label_embs = compute_label_embeddings(CATEGORIES)  # (L, D)
    # process images
    images_out = []
    image_img_probs = []  # store per-image CLIP-prob vectors for later aggregation

    for f in files:
        content = await f.read()
        orig_pil = image_to_pil(content)
        # Foreground crop + mask (try to crop for speed & focus)
        cropped_pil, cropped_mask = grabcut_foreground(orig_pil)
        # embedding and sims
        img_emb = get_clip_image_embedding(cropped_pil)
        sims = img_emb @ label_embs.T  # dot product since normalized -> cos sim
        img_probs = sims_to_probs(sims, temperature=0.07)
        image_img_probs.append(img_probs)

        # color extraction on cropped foreground
        colors = get_dominant_colors_lab(cropped_pil, cropped_mask, n_colors=3)

        # heuristics (mask should align to cropped coordinates)
        heur = compute_mask_heuristics(cropped_mask)

        # debug overlay for frontend (overlay cropped mask back onto original scale)
        # We need a mask in original image coords. For now, show overlay on cropped image (simpler).
        overlay_uri = mask_overlay_datauri(cropped_pil, cropped_mask, alpha=0.35)

        images_out.append({
            "filename": f.filename,
            "cropped_w": cropped_pil.width,
            "cropped_h": cropped_pil.height,
            "category_similarities": [(CATEGORIES[i], float(sims[i])) for i in np.argsort(-sims)[:6]],
            "category_probs": [(CATEGORIES[i], float(img_probs[i])) for i in np.argsort(-img_probs)[:6]],
            "colors": colors,
            "heuristics": heur,
            "mask_overlay": overlay_uri
        })

    # Aggregate images: average image probs across images (simple)
    if len(image_img_probs) == 0:
        return {"error": "no images"}
    avg_img_probs = normalize_array(np.mean(np.vstack(image_img_probs), axis=0))

    # text-based
    text_blob = (title or "") + " . " + (description or "")
    text_emb = get_clip_text_embedding(text_blob)
    text_sims = text_emb @ label_embs.T
    text_probs = sims_to_probs(text_sims, temperature=0.07)

    # Compute a combined score: weight image 0.5, text 0.3, heuristics averaged 0.15, color boost 0.05
    # heuristics averaged across images
    heur_avg = {c: 0.0 for c in CATEGORIES}
    for img in images_out:
        for c in CATEGORIES:
            heur_avg[c] += img["heuristics"].get(c, 0.0)
    for c in heur_avg:
        heur_avg[c] = heur_avg[c] / max(1, len(images_out))

    # color_strength: max color confidence across images (we use highest mapped color confidence)
    color_strength = {}
    for c in CATEGORIES:
        color_strength[c] = 0.0
    for img in images_out:
        for col in img["colors"]:
            mapped = col["mapped_color"]
            # if color name present provides small boost to clothing categories - simple heuristic
            for cat in CATEGORIES:
                # This is domain-specific; keep small boost
                color_strength[cat] = max(color_strength[cat], col.get("confidence", 0.0))

    # weights
    w_img, w_text, w_heur, w_color = 0.50, 0.30, 0.15, 0.05
    raw_scores = []
    for idx, c in enumerate(CATEGORIES):
        val = w_img * float(avg_img_probs[idx]) + w_text * float(text_probs[idx]) + w_heur * float(heur_avg[c]) + w_color * float(color_strength[c])
        raw_scores.append(val)
    raw_arr = np.array(raw_scores)
    final_probs = sims_to_probs(raw_arr, temperature=0.07)
    top_idx = int(np.argmax(final_probs))
    final_category = CATEGORIES[top_idx]
    final_conf = float(final_probs[top_idx])

    # determine conflict: text suggestion differs AND final_conf is low or gap small
    text_top_idx = int(np.argmax(text_probs))
    text_top = CATEGORIES[text_top_idx]
    gap = float(final_probs[top_idx] - final_probs[np.argsort(-final_probs)[1]])
    conflict = (final_category != text_top) and (final_conf < 0.6 or gap < 0.08)

    resp = {
        "title": title,
        "description": description,
        "final_category": {"value": final_category, "confidence": final_conf, "top_from_text": text_top, "text_score": float(text_probs[text_top_idx])},
        "per_label": [(CATEGORIES[i], float(final_probs[i])) for i in np.argsort(-final_probs)],
        "images": images_out,
        "conflict_flag": bool(conflict),
        "timing_sec": round(time.time() - t0, 3)
    }

    return resp

# --- correction endpoint for seller feedback (store logs) ---
@app.post("/correct")
async def correct(original_title: str = Form(...), corrected_category: str = Form(...), notes: str = Form(""), filename: str = Form("")):
    rec = {
        "time": time.time(),
        "original_title": original_title,
        "corrected_category": corrected_category,
        "notes": notes,
        "filename": filename
    }
    log_correction(rec)
    return {"status": "ok", "logged": rec}

if __name__ == "__main__":
    uvicorn.run("main_improved:app", host="0.0.0.0", port=8000, reload=True)

