# 🛍️ Meesho Dice Hackathon – Attribute Normalization Prototype

## 📖 Overview  
This project was built as part of the **Meesho Dice Hackathon (Data Science Track)**.  
It tackles a key pain point in **search and filters experience** on Meesho:

- Sellers often upload products with inconsistent **titles, descriptions, or images**.  
- Current search results may misclassify products (e.g., saree tagged as lehenga) or surface irrelevant items when filters (like color) are applied.  
- This hurts both **buyers** (poor discovery, mistrust) and **sellers** (lost sales).  

**Our solution:**  
👉 **Attribute Normalization Pipeline** using **Computer Vision (CV), NLP, and optional LLM (Gemini)**.  
👉 Ensures **products are categorized correctly**, colors mapped to a fixed palette, and conflicts flagged for sellers.  
👉 Helps Meesho advance its mission of **democratizing internet commerce** by improving discoverability and fairness for all sellers.  

---

## 🚀 Features  

- **Dual Analysis Pipelines**  
  - 🔍 **CV + NLP Mode (default):** Lightweight, client-friendly pipeline for extracting product category, color, and attributes.  
  - 🤖 **LLM Mode (Gemini):** Optional, richer attribute extraction using Google’s Gemini LLM for complex cases.  

- **Foreground Segmentation**  
  - Uses GrabCut to isolate product foreground, avoiding background colors being picked up.  

- **Color Normalization**  
  - Extracts dominant colors via K-Means clustering.  
  - Maps to a fixed set of ~15 normalized fashion colors (e.g., maroon, red, green).  

- **Text Attribute Extraction**  
  - Uses DistilBERT (or TinyBERT) for structured tagging from seller-provided titles/descriptions.  
  - Example: *“half-saree” → normalized category `saree`*.  

- **Conflict Resolution**  
  - If CV and NLP disagree, image-based attributes are prioritized.  
  - Seller is flagged with feedback: *“Image shows saree, description says kurti.”*  

- **Multi-Variant Support**  
  - For multi-color packs, each variant image is tagged.  
  - When filtering by color (e.g. maroon), the **correct variant image** is surfaced in search results.  

- **Feedback Loop**  
  - Customer reviews (text & images) are embedded via BERT.  
  - Seller gets structured feedback: *“Customers report size mismatch”* or *“Color in image differs from received product.”*  

---

## 🏗️ Tech Stack  

### Backend  
- **FastAPI** – REST API backend  
- **Computer Vision** – OpenCV (GrabCut), CLIP embeddings, KMeans  
- **NLP** – Hugging Face Transformers (DistilBERT / TinyBERT)  
- **LLM Option** – Google Gemini (`generativelanguage.googleapis.com`)  
- **Python libs** – `torch`, `transformers`, `scikit-learn`, `Pillow`, `requests`  

### Frontend  
- **React (Vite)** – Meesho-style UI  
- **Tailwind CSS** – Styling and responsiveness  
- **Axios** – API calls  

### Deployment  
- Local dev via `uvicorn` (backend) and `npm run dev` (frontend)  
- Extensible to client-side CV with TensorFlow Lite / ONNX Runtime  

---

## ⚙️ Setup & Installation  

### 1. Clone Repo  
```bash
git clone https://github.com/<your-username>/meesho-attribute-normalization.git
cd meesho-attribute-normalization
```
### 2. Backend Setup
```bash
cd backend
python -m venv venv
```
# Windows:
```bash
venv\Scripts\activate
```
# Linux / macOS:
```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Run FastAPI server (CV + NLP pipeline):

```bash
uvicorn main_improved:app --reload --port 8000
```

Run FastAPI server (LLM pipeline with Gemini):

```bash
# PowerShell
$env:GEMINI_API_KEY="your_api_key"
$env:GEMINI_API_URL="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
```
```bash
uvicorn main_2_debug:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```


Visit: http://localhost:5173

## 🖼️ Demo Walkthrough

- Upload product images (max 6)
- Enter title/description
- Select pipeline:
  - CV + NLP (fast, lightweight)
  - LLM (Gemini) (rich but slower/costlier)
- Click Analyze → see:
  - Predicted category with confidence
  - Normalized colors
  - Conflict flag (if mismatch)
  - Variant image mapping (for filters)
- Download JSON output
- If incorrect, submit a seller correction

## 📊 Example JSON Output
```bash
{
  "final_category": { "value": "saree", "confidence": 0.92 },
  "colors": [
    { "value": "maroon", "confidence": 0.85 },
    { "value": "red", "confidence": 0.10 }
  ],
  "conflict_flag": false,
  "images": [
    {
      "filename": "test.jpg",
      "category_scores": [["saree", 0.92], ["lehenga", 0.05]],
      "colors": [{ "rgb": [93, 13, 39], "mapped_color": "maroon", "confidence": 0.85 }]
    }
  ]
}
```

## 📈 Impact

- **Customers** → Accurate search & filters → better discovery + trust
- **Sellers** → Fewer misclassifications, structured feedback → improved sales
- **Meesho** → Higher engagement, reduced returns, democratized commerce

## 🏅 Hackathon Evaluation Alignment

- **Problem Identification** → Misclassification & filter mismatches clearly defined
- **Solution Quality** → Detailed CV/NLP/LLM pipeline + feedback loop
- **Feasibility & Technical Rigour** → Lightweight, scalable, client-side possible
- **Presentation** → Interactive React UI + JSON outputs + correction flow

## 📌 Future Improvements

- Fine-tune CV classifier on Indian ethnic wear datasets (DeepFashion2, Meesho samples)
- Client-side deployment with TensorFlow Lite
- Advanced review sentiment analysis
- Integration with search ranking personalization

## 👥 Team


Built with ❤️ for the Meesho Dice Hackathon
Data Science Role – Search & Filters Enhancement Track
