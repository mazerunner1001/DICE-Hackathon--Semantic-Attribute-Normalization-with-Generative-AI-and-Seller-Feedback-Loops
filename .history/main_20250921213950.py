# main.py
import io
import uvicorn
import numpy as np
from PIL import Image
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import colorsys
import numpy as np
from skimage import color  # optional for conversion (pip install scikit-image)


app = FastAPI(title="Attribute Normalization Prototype")

# Allow local frontend to call
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CLIP model & processor once (will download on first run)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define canonical category labels and color palette
CATEGORIES = [
    "saree", "kurti", "jeans", "t-shirt", "men's accessory", "dress",
    "blouse", "lehenga", "kurta", "shorts", "top", "jacket"
]

# Fixed color palette (name -> RGB)
COLOR_PALETTE = {
    "black": (0,0,0), "white": (255,255,255), "red": (220,20,60),
    "maroon": (128,0,0), "blue": (30,144,255), "navy": (0,0,128),
    "green": (34,139,34), "olive": (128,128,0), "yellow": (255,215,0),
    "pink": (255,105,180), "orange": (255,140,0), "brown": (165,42,42),
    "gray": (128,128,128), "beige": (245,245,220)
}


PROMPT_TEMPLATES = [
    "a product photo of a {}",
    "a photo of a {} on a person",
    "a studio photo of a {}",
    "a close-up of a {}",
    "a picture of {}"
]


def cosine_sim(a: np.ndarray, b: np.ndarray):
    # a: (d,), b: (N, d) -> (N,)
    a_t = torch.tensor(a).to(device)
    b_t = torch.tensor(b).to(device)
    if a_t.ndim == 1:
        a_t = a_t.unsqueeze(0)
    sims = F.cosine_similarity(a_t, b_t)
    return sims.detach().cpu().numpy()

def image_to_pil(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def get_clip_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = model.get_image_features(**inputs)
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
    return img_emb.cpu().numpy()[0]

def get_clip_text_embeddings(texts: List[str]):
    inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        txt_emb = model.get_text_features(**inputs)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
    return txt_emb.cpu().numpy()

def get_top_labels_by_similarity(img_emb: np.ndarray, label_texts: List[str], top_k=3):
    label_embs = get_clip_text_embeddings(label_texts)
    sims = (img_emb @ label_embs.T)  # dot since normalized -> cos sim
    idx = np.argsort(-sims)[:top_k]
    return [(label_texts[i], float(sims[i])) for i in idx]

def get_dominant_colors(image: Image.Image, n_colors=3):
    arr = np.array(image)
    # downsample pixels to speed up
    h, w, _ = arr.shape
    sample = arr.reshape(-1, 3)
    # optional: sample subset if too big
    if sample.shape[0] > 10000:
        idx = np.random.choice(sample.shape[0], 10000, replace=False)
        sample = sample[idx]
    # cluster
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(sample)
    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    # rank by counts
    order = np.argsort(-counts)
    dominant = []
    for i in order:
        rgb = tuple(int(x) for x in centers[i])
        weight = float(counts[i]) / np.sum(counts)
        dominant.append((rgb, weight))
    return dominant

def rgb_to_palette_name(rgb):
    # find nearest palette color by Euclidean distance
    best = None
    best_dist = 1e9
    for name, prgb in COLOR_PALETTE.items():
        d = sum((rgb[i]-prgb[i])**2 for i in range(3))
        if d < best_dist:
            best_dist = d
            best = name
    # convert distance to confidence heuristic (normalize)
    max_dist = 255**2 * 3
    confidence = 1 - (best_dist / max_dist)
    return best, confidence

def grabcut_foreground(pil_img, rect_margin_ratio=0.08, iter_count=5):
    """
    Run GrabCut with an initial rectangle (central region) and return:
      - cropped PIL image of foreground (transparent background if you want),
      - binary mask (same size as original) where foreground=1.
    rect_margin_ratio: fraction of width/height to leave as padding (8% default).
    """
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    # initial rectangle around center (inset margins)
    left = int(w * rect_margin_ratio)
    top  = int(h * rect_margin_ratio)
    right = int(w * (1 - rect_margin_ratio))
    bottom = int(h * (1 - rect_margin_ratio))
    rect = (left, top, right - left, bottom - top)

    # masks for GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        # fallback: return full image as mask=1
        mask_fg = np.ones_like(mask, dtype=np.uint8)
        return pil_img, mask_fg

    # mask: probable/definite foreground -> 1, else 0
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')  # 0 background, 1 foreground

    # create RGBA image where background is transparent (optional)
    rgba = np.dstack((img, mask2*255)).astype('uint8')

    # crop to bounding rect of mask to speed downstream steps
    coords = np.column_stack(np.where(mask2 > 0))
    if coords.size == 0:
        # fallback: return original
        mask_fg = np.ones_like(mask, dtype=np.uint8)
        return pil_img, mask_fg
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped_rgb = img[y0:y1, x0:x1]
    cropped_mask = mask2[y0:y1, x0:x1]

    cropped_pil = Image.fromarray(cropped_rgb)
    return cropped_pil, cropped_mask

def get_dominant_colors(image_pil, mask=None, n_colors=3, sample_limit=10000):
    """
    image_pil: PIL crop (RGB)
    mask: 2D np.array same width/height as image crop (1 for fg, 0 for bg) OR None
    returns list of (rgb_tuple, weight)
    """
    arr = np.array(image_pil)  # HxWx3
    h, w = arr.shape[:2]
    pixels = arr.reshape(-1, 3)

    if mask is not None:
        # mask should align to this crop
        mask_flat = mask.reshape(-1)
        pixels = pixels[mask_flat == 1]
        if pixels.shape[0] == 0:
            # fallback: use whole image
            pixels = arr.reshape(-1, 3)

    # sample to speed up
    if pixels.shape[0] > sample_limit:
        idx = np.random.choice(pixels.shape[0], sample_limit, replace=False)
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels

    # KMeans
    kmeans = KMeans(n_clusters=min(n_colors, max(1, len(pixels_sample)//50)), random_state=0).fit(pixels_sample)
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels, minlength=len(centers))
    order = np.argsort(-counts)
    res = []
    total = counts.sum()
    for i in order:
        rgb = tuple(int(x) for x in centers[i])
        weight = float(counts[i]) / float(total)
        res.append((rgb, weight))
    return res


def get_label_embeddings_with_prompts(labels: List[str]):
    # for each label, build multiple prompt strings, compute embeddings and average
    all_prompts = []
    label_to_indices = []
    for lbl in labels:
        start_idx = len(all_prompts)
        for t in PROMPT_TEMPLATES:
            all_prompts.append(t.format(lbl))
        label_to_indices.append((start_idx, len(PROMPT_TEMPLATES)))
    emb = get_clip_text_embeddings(all_prompts)  # same function you already have
    # average per label
    avg_embs = []
    for start, count in label_to_indices:
        avg = emb[start:start+count].mean(axis=0)
        avg = avg / np.linalg.norm(avg)
        avg_embs.append(avg)
    return np.vstack(avg_embs)  # shape (num_labels, dim)



def rgb_to_lab(rgb):
    # rgb: (R,G,B) 0-255
    arr = np.array([[rgb]], dtype=np.uint8)
    lab = color.rgb2lab(arr / 255.0)
    return lab[0,0]  # (L,a,b)


def sims_to_probs(sims, temperature=0.05):
    # sims: numpy array of cosine similarities
    exps = np.exp(sims / temperature)
    probs = exps / exps.sum()
    return probs


@app.post("/analyze")
async def analyze(
    title: str = Form(...),
    description: str = Form(""),
    files: List[UploadFile] = File(...)
):
    # For each image: compute image embedding and color clusters
    images = []
    for f in files:
        content = await f.read()
        pil = image_to_pil(content)
        images.append({"filename": f.filename, "pil": pil})

    # Precompute label embeddings once (text labels)
    category_label_embs = get_clip_text_embeddings(CATEGORIES)

    results = []
    # Image-wise processing
    for img in images:
        pil = img["pil"]
        # img_emb = get_clip_image_embedding(pil)   vector

        cropped_pil, cropped_mask = grabcut_foreground(pil)
        img_emb = get_clip_image_embedding(cropped_pil)
        # category scores
        cat_sims = (img_emb @ category_label_embs.T)
        cat_scores = [(CATEGORIES[i], float(cat_sims[i])) for i in np.argsort(-cat_sims)[:5]]

        # dominant colors
        dom = get_dominant_colors(pil, n_colors=3)
        color_tags = []
        for rgb, weight in dom:
            cname, cconf = rgb_to_palette_name(rgb)
            # final confidence = cluster weight * palette match confidence
            color_tags.append({"rgb": rgb, "mapped_color": cname, "confidence": float(weight * cconf)})

        results.append({
            "filename": img["filename"],
            "category_scores": cat_scores,
            "colors": color_tags
        })

    # Text-based: compute similarity of title+description to categories
    text_blob = title + " . " + description
    text_embs = get_clip_text_embeddings([text_blob])  # 1 x d
    text_emb = text_embs[0]
    text_cat_sims = (text_emb @ category_label_embs.T)
    text_cat_scores = [(CATEGORIES[i], float(text_cat_sims[i])) for i in np.argsort(-text_cat_sims)[:5]]

    # Aggregate per-listing normalization & conflict detection
    # Choose top image category across images by max image score
    best_image_cat = None
    best_image_score = -1
    for r in results:
        top_cat, top_score = r["category_scores"][0]
        if top_score > best_image_score:
            best_image_score = top_score
            best_image_cat = top_cat

    top_text_cat, top_text_score = text_cat_scores[0]

    # Decide final category: prioritize image if strong enough
    IMAGE_THRESHOLD = 0.20  # cosine sims from CLIP are between -1..1; tuned low for demo
    final_category = top_text_cat
    conflict = False
    if best_image_score >= IMAGE_THRESHOLD:
        final_category = best_image_cat
        if best_image_cat != top_text_cat and top_text_score >= 0.12:
            conflict = True
    else:
        # fallback to text
        final_category = top_text_cat

    # Build normalized JSON
    json_out = {
        "title": title,
        "description": description,
        "final_category": {"value": final_category, "image_score": best_image_score, "text_score": top_text_score},
        "images": results,
        "text_category_scores": text_cat_scores,
        "conflict_flag": conflict
    }
    return json_out


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
