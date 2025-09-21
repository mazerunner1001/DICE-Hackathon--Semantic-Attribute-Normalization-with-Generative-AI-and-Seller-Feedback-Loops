// src/App.jsx
import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function ColorChip({ name, rgb }) {
  const bg = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
  return (
    <div className="color-chip" title={name}>
      <div className="color-dot" style={{ background: bg }} />
      <div className="color-name">{name}</div>
    </div>
  );
}

function App() {
  const [title, setTitle] = useState("");
  const [desc, setDesc] = useState("");
  const [files, setFiles] = useState([]);
  const [previewUrls, setPreviewUrls] = useState([]);
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [correction, setCorrection] = useState("");

  function handleFiles(e) {
    const f = Array.from(e.target.files);
    setFiles(f);
    setPreviewUrls(f.map(file => URL.createObjectURL(file)));
  }

  async function submit() {
    if (files.length === 0) { alert("Upload at least one image"); return; }
    setLoading(true);
    setResp(null);
    const fd = new FormData();
    fd.append("title", title);
    fd.append("description", desc);
    files.forEach(f => fd.append("files", f, f.name));
    try {
      const r = await axios.post("http://localhost:8000/analyze", fd, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000
      });
      setResp(r.data);
      // default correction to final category
      setCorrection(r.data.final_category?.value || "");
    } catch (err) {
      console.error(err);
      alert("Error: see console");
    } finally {
      setLoading(false);
    }
  }

  async function submitCorrection() {
    try {
      const fd = new FormData();
      fd.append("original_title", title);
      fd.append("corrected_category", correction);
      fd.append("notes", "Corrected via demo UI");
      fd.append("filename", files[0]?.name ?? "");
      await axios.post("http://localhost:8000/correct", fd);
      alert("Correction logged. Thanks!");
    } catch (err) {
      console.error(err);
      alert("Failed to log correction");
    }
  }

  return (
    <div className="app-root">
      <header className="header">
        <div className="logo">Meesho • Demo</div>
        <div className="header-right">Attribute Normalization Prototype</div>
      </header>

      <main className="container">
        <section className="card input-card">
          <h3>Upload product images</h3>
          <input type="text" placeholder="Title (e.g. Elegant maroon saree with golden border)" value={title} onChange={e => setTitle(e.target.value)} />
          <textarea placeholder="Description" value={desc} onChange={e => setDesc(e.target.value)} />
          <input type="file" accept="image/*" multiple onChange={handleFiles} />
          <div className="thumbs">
            {previewUrls.map((u, i) => <img key={i} src={u} alt="thumb" className="thumb" />)}
          </div>
          <div className="actions">
            <button onClick={submit} disabled={loading}>{loading ? "Analyzing…" : "Analyze"}</button>
          </div>
        </section>

        {resp && (
          <section className="card result-card">
            <div className="result-header">
              <div className="result-title">Final Category: <strong>{resp.final_category.value}</strong></div>
              <div className="result-meta">Confidence: {(resp.final_category.confidence * 100).toFixed(1)}% &nbsp;•&nbsp; Time: {resp.timing_sec}s</div>
            </div>

            {resp.conflict_flag && (
              <div className="conflict">⚠️ Conflict detected between text and image. Please verify category below.</div>
            )}

            <div className="per-labels">
              <h4>Per-label scores</h4>
              <div className="label-list">
                {resp.per_label.slice(0,8).map(([lab, sc], idx) => (
                  <div key={idx} className="label-item">
                    <div className="label-name">{lab}</div>
                    <div className="label-score">{(sc*100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="images-grid">
              {resp.images.map((img, idx) => (
                <div key={idx} className="img-card">
                  <div className="img-wrap">
                    <img src={previewUrls[idx] || ""} className="prod-img" alt="orig" />
                    <img src={img.mask_overlay} className="mask-overlay" alt="mask" />
                  </div>
                  <div className="img-attrs">
                    <div><strong>Top image categories</strong></div>
                    {img.category_probs.slice(0,3).map((p, i) => <div key={i}>{p[0]} — {(p[1]*100).toFixed(1)}%</div>)}
                    <div className="colors">
                      <strong>Colors</strong>
                      <div className="color-row">
                        {img.colors.map((c, i) => (
                          <div key={i} className="color-chip-compact">
                            <div style={{background:`rgb(${c.rgb[0]},${c.rgb[1]},${c.rgb[2]})`}} className="dot"></div>
                            <div className="cap">{c.mapped_color} {(c.confidence*100).toFixed(0)}%</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {resp.conflict_flag && (
              <div className="correction">
                <h4>Seller correction</h4>
                <select value={correction} onChange={e => setCorrection(e.target.value)}>
                  {["saree","lehenga","kurti","jeans","t-shirt","dress","blouse","jacket","men's accessory"].map(c=> <option key={c} value={c}>{c}</option>)}
                </select>
                <button onClick={submitCorrection}>Submit correction</button>
              </div>
            )}

            <pre className="json-view">{JSON.stringify(resp, null, 2)}</pre>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
