import React, { useState, useEffect } from "react";
import { getHealth, postAnswer } from "./api.js";

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await postAnswer(question);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "2rem" }}>
      <h1>UTI Clinical Decision Support</h1>
      <p>Evidence-Grounded Clinical Question Answering</p>

      {health && (
        <div style={{ margin: "1rem 0", padding: "0.75rem", background: "#e8f5e9", borderRadius: 8 }}>
          <strong>System:</strong> {health.model} | Docs: {health.indexed_documents} | Top-K: {health.top_k}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Example: What antibiotics are recommended for men aged 16 years and over?"
          style={{ width: "100%", minHeight: 120, padding: "0.75rem", fontSize: 16 }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{ marginTop: "1rem", padding: "0.75rem 2rem", fontSize: 16, cursor: loading ? "wait" : "pointer" }}
        >
          {loading ? "Retrieving..." : "Get Evidence-Based Recommendation"}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: "1rem", padding: "0.75rem", background: "#ffebee", borderRadius: 8 }}>
          Error: {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: "2rem" }}>
          <h2>Status: {result.status}</h2>
          <p><strong>Confidence:</strong> {result.confidence}</p>
          <p><strong>Reason:</strong> {result.reason}</p>

          {result.status === "ANSWERED" && result.answer && (
            <div style={{ marginTop: "1rem" }}>
              <h3>Recommendation</h3>
              <pre style={{ whiteSpace: "pre-wrap", fontFamily: "inherit" }}>{result.answer}</pre>
            </div>
          )}

          {result.source && (
            <div style={{ marginTop: "1rem" }}>
              <h3>Source</h3>
              <p>ID: {result.source.source_id}</p>
              <p>Title: {result.source.title}</p>
              <p>Pages: {result.source.pages}</p>
            </div>
          )}

          {result.results && result.results.length > 0 && (
            <div style={{ marginTop: "1rem" }}>
              <h3>Retrieved Evidence ({result.results.length})</h3>
              {result.results.map((r, i) => (
                <div key={i} style={{ padding: "0.5rem", borderBottom: "1px solid #ddd" }}>
                  <strong>Rank {i + 1}:</strong> {r.metadata?.source_id} |
                  Similarity: {r.similarity?.toFixed(4)} |
                  Hybrid: {r.hybrid_score?.toFixed(4)}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
