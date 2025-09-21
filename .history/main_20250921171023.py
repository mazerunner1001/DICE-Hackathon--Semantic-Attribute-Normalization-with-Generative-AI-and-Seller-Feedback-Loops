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
        img_emb = get_clip_image_embedding(pil)  # vector
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
