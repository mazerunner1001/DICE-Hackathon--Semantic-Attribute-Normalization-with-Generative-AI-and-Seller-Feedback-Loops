// src/App.jsx
import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

/*
  Meesho-themed Demo Frontend (updated)
  - Pipeline toggle: "CV + NLP" (default) or "LLM (Gemini)"
  - Sends 'pipeline' param to backend
  - Hides tuning sliders for LLM
  - Shows LLM warnings (cost/latency)
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
  const tiles = [
    { title: "Footwear", brand: "FLITE / Bata" },
    { title: "Personal Care", brand: "mamaearth" },
    { title: "Health & Wellness", brand: "Patanjali" },
    { title: "Women Innerwear", brand: "Lyra" },
    { title: "Men Innerwear", brand: "Dollar" },
    { title: "Electronics", brand: "Hoppup" }
  ];
  return (
    <div className="carousel">
      {tiles.map((t,i)=>(
        <div className="carousel-card" key={i}>
          <div className="carousel-art">
            <PriceTag />
            <div className="hero-circle" />
            <img alt={t.title} src={`https://via.placeholder.com/220x160.png?text=${encodeURIComponent(t.title)}`} />
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
      <div className="hint">Changes are demo-only; update backend to persist.</div>
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
  const [pipeline, setPipeline] = useState("cv"); // "cv" or "llm"
  const previewRef = useRef([]);

  useEffect(() => {
    return () => {
      previews.forEach(u => URL.revokeObjectURL(u));
    };
  }, [previews]);

  function handleFiles(e) {
    const f = Array.from(e.target.files).slice(0,6);
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
    fd.append("pipeline", pipeline);
    files.forEach(f => fd.append("files", f, f.name));
    // include tuning params only for CV pipeline (backend may ignore)
    if (pipeline === "cv") {
      fd.append("temperature", temperature);
      fd.append("weights", JSON.stringify(weights));
    }
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
              <p className="muted">Upload images for the product you want to list. Select pipeline: CV+NLP (fast, cheaper) or LLM (Gemini) — slower & costlier.</p>

              <div className="pipeline-row">
                <label className={`pipe-btn ${pipeline==="cv" ? "active":""}`} onClick={()=>setPipeline("cv")}>CV + NLP</label>
                <label className={`pipe-btn ${pipeline==="llm" ? "active":""}`} onClick={()=>setPipeline("llm")}>LLM (Gemini)</label>
              </div>

              {pipeline === "llm" && (
                <div className="llm-note">
                  <strong>LLM Mode:</strong> The server will call Gemini (multimodal) to analyze images and text. Expect higher latency and potential costs.
                </div>
              )}

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

            {/* show tuning only when CV selected */}
            {pipeline === "cv" && <TuningPanel weights={weights} setWeights={setWeights} temperature={temperature} setTemperature={setTemperature} />}

            <section className="card help">
              <h4>Demo notes</h4>
              <ol>
                <li>CV+NLP mode uses GrabCut + CLIP + heuristics (fast).</li>
                <li>LLM mode sends images to Gemini to generate attributes (slower, better for ambiguous cases).</li>
                <li>Logged corrections are used as supervised labels for future training.</li>
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
                    <div className="meta">Confidence: {(resp.final_category.confidence*100).toFixed(1)}% • Time: {resp.timing_sec}s • Pipeline: {resp.pipeline || pipeline}</div>
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
                          {debugMode && img.mask_overlay && <img src={img.mask_overlay} className="overlay" alt="overlay" />}
                          {!debugMode && img.mask_overlay && <img src={img.mask_overlay} className="overlay blurred" alt="overlay" />}
                        </div>
                      </div>
                      <div className="img-right">
                        <div className="row">
                          <div className="col">
                            <strong>Top image labels</strong>
                            {img.category_probs?.slice(0,4).map((p, k)=>(
                              <div key={k} className="prob"><span className="pname">{p[0]}</span> <span className="pscore">{(p[1]*100).toFixed(1)}%</span></div>
                            ))}
                            {/* LLM mode might return 'category_scores' instead of 'category_probs' - handle both */}
                            {(!img.category_probs && img.category_scores) && img.category_scores.slice(0,4).map((p,k)=>(
                              <div key={k} className="prob"><span className="pname">{p[0]}</span> <span className="pscore">{(p[1]*100).toFixed(1)}%</span></div>
                            ))}
                          </div>
                          <div className="col">
                            <strong>Colors (masked)</strong>
                            <div className="color-chip-row">
                              {img.colors?.map((c, idx)=>(
                                <div key={idx} className="color-chip">
                                  <div className="color-swatch" style={{background:`rgb(${c.rgb[0]},${c.rgb[1]},${c.rgb[2]})`}} />
                                  <div className="color-meta"><div className="cname">{c.mapped_color}</div><div className="cconf">{(c.confidence*100).toFixed(0)}%</div></div>
                                </div>
                              )) || <div className="muted">No colors returned</div>}
                            </div>
                          </div>
                        </div>

                        <div className="row heur">
                          <div><strong>Heuristics</strong></div>
                          <div className="heur-list">
                            {img.heuristics ? Object.entries(img.heuristics).slice(0,4).map(([k,v], idx)=>(<div key={idx} className="heur-item">{k}: {(v*100).toFixed(0)}%</div>)) : <div className="muted">LLM may not return heuristics</div>}
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
