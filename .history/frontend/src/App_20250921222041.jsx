// src/App.jsx
import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

/*
  Meesho-themed Demo Frontend
  - Upload images + title/description
  - Sends to backend /analyze
  - Displays per-image mask overlay + attributes
  - Allows corrections to be submitted to /correct
  - Tuning sliders for weights and temperature (sends only to UI; edit backend to use)
*/

function PriceTag() {
  return <div className="sale-tag">SALE<br/><span>UPTO 80% OFF</span></div>;
}

function Header({ search, setSearch }) {
  return (
    <header className="mh-header">
      <div className="mh-topbar">
        <div className="mh-logo">meesho</div>
        <div className="mh-search">
          <svg width="18" height="18" viewBox="0 0 24 24" className="search-icon"><path fill="#6b4b8a" d="M21 20l-5.6-5.6A7.5 7.5 0 1 0 18.5 18.5L24 24z"></path></svg>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Try Saree, Kurti or Search by Product Code"
          />
        </div>
        <div className="mh-actions">
          <div className="link">Become a Supplier</div>
          <div className="link">Profile</div>
          <div className="cart">Cart</div>
        </div>
      </div>

      <nav className="mh-nav">
        <div className="nav-scroll">
          {["Women Ethnic","Women Western","Men","Kids","Home & Kitchen","Beauty & Health","Jewellery & Accessories","Bags & Footwear","Electronics"].map((c) => (
            <button key={c} className="nav-chip">{c}</button>
          ))}
        </div>
      </nav>
    </header>
  );
}

function CategoryCarousel() {
  // static demo tiles (visual flair)
  const tiles = [
    { title: "Footwear", brand: "FLITE / Bata", img: "https://images.unsplash.com/photo-1519864600265-abb23847ef2c?auto=format&fit=crop&w=400&q=80" },
    { title: "Personal Care", brand: "mamaearth", img: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=400&q=80" },
    { title: "Health & Wellness", brand: "Patanjali", img: "https://images.unsplash.com/photo-1515378791036-0648a3ef77b2?auto=format&fit=crop&w=400&q=80" },
    { title: "Women Innerwear", brand: "Lyra", img: "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?auto=format&fit=crop&w=400&q=80" },
    { title: "Men Innerwear", brand: "Dollar", img: "https://images.unsplash.com/photo-1465101046530-73398c7f28ca?auto=format&fit=crop&w=400&q=80" },
    { title: "Electronics", brand: "Hoppup", img: "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=400&q=80" }
  ];
  return (
    <div className="carousel">
      {tiles.map((t,i)=>(
        <div className="carousel-card" key={i}>
          <div className="carousel-art">
            <PriceTag />
            <div className="hero-circle" />
            <img alt={t.title} src={t.img} />
          </div>
          <div className="carousel-footer">
            <div className="brand-pill">{t.brand}</div>
            <div className="cat-text">{t.title}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function TuningPanel({ weights, setWeights, temperature, setTemperature }) {
  return (
    <div className="tuning card">
      <h4>Tuning (demo only)</h4>
      <div className="tuning-row">
        <label>Image weight</label>
        <input type="range" min="0" max="1" step="0.01" value={weights.img} onChange={e=>setWeights({...weights, img: parseFloat(e.target.value)})}/>
        <div className="tval">{weights.img}</div>
      </div>
      <div className="tuning-row">
        <label>Text weight</label>
        <input type="range" min="0" max="1" step="0.01" value={weights.text} onChange={e=>setWeights({...weights, text: parseFloat(e.target.value)})}/>
        <div className="tval">{weights.text}</div>
      </div>
      <div className="tuning-row">
        <label>Heuristics weight</label>
        <input type="range" min="0" max="1" step="0.01" value={weights.heur} onChange={e=>setWeights({...weights, heur: parseFloat(e.target.value)})}/>
        <div className="tval">{weights.heur}</div>
      </div>
      <div className="tuning-row">
        <label>Color boost</label>
        <input type="range" min="0" max="1" step="0.01" value={weights.color} onChange={e=>setWeights({...weights, color: parseFloat(e.target.value)})}/>
        <div className="tval">{weights.color}</div>
      </div>
      <div className="tuning-row">
        <label>Temperature</label>
        <input type="range" min="0.01" max="0.2" step="0.01" value={temperature} onChange={e=>setTemperature(parseFloat(e.target.value))}/>
        <div className="tval">{temperature}</div>
      </div>
      <div className="hint">Changes are demo-only; update backend weights for persistent effect.</div>
    </div>
  );
}

export default function App() {
  const [search, setSearch] = useState("");
  const [title, setTitle] = useState("");
  const [desc, setDesc] = useState("");
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [weights, setWeights] = useState({img:0.5, text:0.3, heur:0.15, color:0.05});
  const [temperature, setTemperature] = useState(0.07);
  const [correctionSelection, setCorrectionSelection] = useState("");
  const [debugMode, setDebugMode] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const previewRef = useRef([]);

  useEffect(() => {
    return () => {
      // revoke object URLs on unmount
      previews.forEach(u => URL.revokeObjectURL(u));
    };
  }, [previews]);

  function handleFiles(e) {
    const f = Array.from(e.target.files).slice(0,6); // limit demo to 6 images
    setFiles(f);
    const urls = f.map(file => URL.createObjectURL(file));
    setPreviews(urls);
  }

  async function handleAnalyze() {
    if (!title && !desc) { setErrorMsg("Add a title or description for better text signals"); }
    if (files.length === 0) { alert("Please upload at least one image"); return; }
    setLoading(true); setResp(null); setErrorMsg("");
    const fd = new FormData();
    fd.append("title", title);
    fd.append("description", desc);
    files.forEach(f => fd.append("files", f, f.name));
    // optional: include tuning params to suggest backend use them (backend must accept)
    fd.append("temperature", temperature);
    fd.append("weights", JSON.stringify(weights));
    try {
      const r = await axios.post("http://localhost:8000/analyze", fd, { headers: {"Content-Type":"multipart/form-data"}, timeout: 180000 });
      setResp(r.data);
      setCorrectionSelection(r.data.final_category?.value || "");
    } catch (err) {
      console.error(err);
      setErrorMsg("Analysis failed: " + (err.response?.data || err.message));
    } finally { setLoading(false); }
  }

  async function submitCorrection() {
    if (!correctionSelection) { alert("Select a category to correct"); return; }
    const fd = new FormData();
    fd.append("original_title", title || "");
    fd.append("corrected_category", correctionSelection);
    fd.append("notes", "Corrected via demo UI");
    fd.append("filename", files[0]?.name || "");
    try {
      await axios.post("http://localhost:8000/correct", fd);
      alert("Correction logged. Thank you!");
    } catch (err) {
      console.error(err);
      alert("Failed to submit correction");
    }
  }

  function downloadJson() {
    if (!resp) return;
    const blob = new Blob([JSON.stringify(resp, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "analysis_result.json"; a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app">
      <Header search={search} setSearch={setSearch} />
      <main className="main">
        <CategoryCarousel />
        <div className="layout">
          <div className="left-col">
            <section className="card upload-card">
              <h3>List a product (Demo)</h3>
              <p className="muted">Upload images for the product you want to list. The demo will analyze images and title to extract normalized attributes.</p>
              <input type="text" placeholder="Title (e.g. Elegant maroon saree with golden border)" value={title} onChange={e=>setTitle(e.target.value)} />
              <textarea placeholder="Description (optional)" value={desc} onChange={e=>setDesc(e.target.value)} />
              <div className="file-row">
                <input type="file" accept="image/*" multiple onChange={handleFiles} />
                <button className="primary" onClick={handleAnalyze} disabled={loading}>{loading ? "Analyzing…" : "Analyze"}</button>
              </div>
              <div className="thumbs-grid">
                {previews.map((u,i)=>(
                  <div className="thumb-wrap" key={i}>
                    <img src={u} alt={"preview-"+i} className="thumb" />
                    <div className="thumb-meta">{files[i]?.name}</div>
                  </div>
                ))}
              </div>

              <div className="small-row">
                <label className="small-label"><input type="checkbox" checked={debugMode} onChange={e=>setDebugMode(e.target.checked)} /> Show debug overlays</label>
                <button className="link-btn" onClick={downloadJson} disabled={!resp}>Download JSON</button>
              </div>

              {errorMsg && <div className="error">{errorMsg}</div>}
            </section>

            <TuningPanel weights={weights} setWeights={setWeights} temperature={temperature} setTemperature={setTemperature} />

            <section className="card help">
              <h4>Demo notes</h4>
              <ol>
                <li>We run foreground extraction (GrabCut), CLIP zero-shot with prompt averaging, masked LAB color clustering and heuristics. The backend returns mask overlay images for debug view.</li>
                <li>Use the tuning sliders to experiment - changes are for demo only; backend weights are authoritative for production.</li>
                <li>When conflict is detected, please use the correction box to log fixes for retraining later.</li>
              </ol>
            </section>
          </div>

          <div className="right-col">
            {!resp && <div className="card placeholder">No analysis yet — upload images and click <b>Analyze</b>.</div>}

            {resp && (
              <div className="card result">
                <div className="result-head">
                  <div>
                    <h3 className="final-cat">{resp.final_category.value}</h3>
                    <div className="meta">Confidence: {(resp.final_category.confidence*100).toFixed(1)}% • Text top: {resp.final_category.top_from_text} ({(resp.final_category.text_score*100).toFixed(1)}%)</div>
                  </div>
                  <div>
                    <button className="primary" onClick={downloadJson}>Download JSON</button>
                  </div>
                </div>

                {resp.conflict_flag && <div className="conflict-flag">⚠️ Conflict detected between image & text. Suggest correction below.</div>}

                <div className="labels-row">
                  <h4>Top Predictions</h4>
                  <div className="label-strip">
                    {resp.per_label.slice(0,8).map(([l, s], idx) => (
                      <div key={idx} className="label-pill">
                        <div className="lp-name">{l}</div>
                        <div className="lp-score">{(s*100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="images-results">
                  {resp.images.map((img,i)=>(
                    <div key={i} className="img-result-card">
                      <div className="img-left">
                        <div className="img-stack">
                          <img src={previews[i]} alt={"orig-"+i} className="orig" />
                          {debugMode && <img src={img.mask_overlay} className="overlay" alt="overlay" />}
                          {!debugMode && <img src={img.mask_overlay} className="overlay blurred" alt="overlay" />}
                        </div>
                      </div>
                      <div className="img-right">
                        <div className="row">
                          <div className="col">
                            <strong>Top image labels</strong>
                            {img.category_probs.slice(0,4).map((p, k)=>(
                              <div key={k} className="prob"><span className="pname">{p[0]}</span> <span className="pscore">{(p[1]*100).toFixed(1)}%</span></div>
                            ))}
                          </div>
                          <div className="col">
                            <strong>Colors (masked)</strong>
                            <div className="color-chip-row">
                              {img.colors.map((c, idx)=>(
                                <div key={idx} className="color-chip">
                                  <div className="color-swatch" style={{background:`rgb(${c.rgb[0]},${c.rgb[1]},${c.rgb[2]})`}} />
                                  <div className="color-meta"><div className="cname">{c.mapped_color}</div><div className="cconf">{(c.confidence*100).toFixed(0)}%</div></div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="row heur">
                          <div><strong>Heuristics</strong></div>
                          <div className="heur-list">
                            {Object.entries(img.heuristics).slice(0,4).map(([k,v], idx)=>(<div key={idx} className="heur-item">{k}: {(v*100).toFixed(0)}%</div>))}
                          </div>
                        </div>

                        <div className="row actions">
                          <button className="secondary" onClick={()=>{ navigator.clipboard.writeText(JSON.stringify(img, null, 2)); alert("Image JSON copied"); }}>Copy image JSON</button>
                        </div>

                      </div>
                    </div>
                  ))}
                </div>

                {resp.conflict_flag && (
                  <div className="correction-box">
                    <h4>Seller Correction</h4>
                    <div className="corr-row">
                      <select value={correctionSelection} onChange={(e)=>setCorrectionSelection(e.target.value)}>
                        {["saree","lehenga","kurti","jeans","t-shirt","dress","blouse","jacket","men's accessory"].map(c=> <option key={c} value={c}>{c}</option>)}
                      </select>
                      <button className="primary" onClick={submitCorrection}>Submit Correction</button>
                    </div>
                  </div>
                )}

                <details className="json-details"><summary>Raw JSON</summary><pre>{JSON.stringify(resp, null, 2)}</pre></details>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        <div>© Demo • Meesho-style UI • Attribute Normalization Prototype</div>
      </footer>
    </div>
  );
}
