import React, { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setResult(data);
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Navbar */}
      <nav className="border-b shadow-sm flex items-center justify-between px-6 py-3">
        <h1 className="text-2xl font-bold text-pink-600">meesho</h1>
        <input
          type="text"
          placeholder="Try Saree, Kurti or Search by Product Code"
          className="border rounded-md px-4 py-2 w-1/2"
        />
        <div className="flex gap-4">
          <button className="text-gray-600">Profile</button>
          <button className="text-gray-600">Cart</button>
        </div>
      </nav>

      {/* Categories */}
      <div className="flex gap-6 px-6 py-4 border-b text-sm font-medium text-gray-700">
        <span>Women Ethnic</span>
        <span>Women Western</span>
        <span>Men</span>
        <span>Kids</span>
        <span>Home & Kitchen</span>
        <span>Beauty & Health</span>
        <span>Electronics</span>
      </div>

      {/* Upload Form */}
      <div className="flex justify-center mt-10">
        <form
          onSubmit={handleUpload}
          className="bg-white shadow-lg rounded-xl p-8 w-[500px] border"
        >
          <h2 className="text-xl font-semibold mb-4 text-gray-800">
            Upload Product for Analysis
          </h2>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files[0])}
            className="mb-4"
          />
          <button
            type="submit"
            className="bg-pink-600 hover:bg-pink-700 text-white px-4 py-2 rounded-lg w-full"
          >
            Analyze
          </button>
        </form>
      </div>

      {/* Results */}
      {result && (
        <div className="px-6 py-10">
          <div className="bg-purple-50 rounded-xl shadow p-6 max-w-2xl mx-auto">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">
              Final Category:{" "}
              <span className="text-pink-600">{result.final_category.value}</span>
            </h3>
            <p className="mb-2 text-gray-700">
              <strong>Title:</strong> {result.title}
            </p>
            <p className="mb-4 text-gray-700">
              <strong>Description:</strong> {result.description}
            </p>
            <div className="flex gap-2">
              {result.images[0].colors.map((c, i) => (
                <div
                  key={i}
                  className="w-8 h-8 rounded-full border"
                  style={{ backgroundColor: c.mapped_color }}
                  title={c.mapped_color}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
