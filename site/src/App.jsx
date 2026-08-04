import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [prompt, setPrompt] = useState('');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState('');
  const [error, setError] = useState('');
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");

  async function loadModels() {
    try {
      const response = await fetch("http://127.0.0.1:11435/api/tags");

      if (!response.ok) {
        throw new Error(`Could not load models: ${response.status}`);
      }

      const data = await response.json();
      const loadedModels = data.models ?? [];

      setModels(loadedModels);

      if (loadedModels.length > 0) {
        setModel(loadedModels[0].name);
      }
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    loadModels();
  }, []);

  async function generate(event) {
    event.preventDefault();

    if (!prompt.trim) {
      return;
    }

    setLoading(true);
    setError("");
    setOutput("");

    try {
      const response = await fetch("http://127.0.0.1:11435/api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          prompt,
          raw: true,
          stream: false,
          options: {
            num_predict: 80,
            temperature: 0.2,
          },
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json;
      setOutput(data.response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <h1>Brittain</h1>

      <form onSubmit={generate}>
        <label htmlFor="model">Model</label>

        <select
          id="model"
          value={model}
          onChange={(event) => setModel(event.target.value)}
          disable={models.length === 0}
        >
          {models.length === 0 && (
            <option>Loading models...</option>
          )}

          {models.map((item) => (
            <option key={item.name} value={item.name}>
              {item.name} - {item.details.parameter_size} - {item.context} ctx
            </option>
          ))}
        </select>
        
        <label htmlFor="prompt">Prompt</label>

        <textarea
          id="prompt"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder='def add(a,b)'
          row="10"
        />

        <button type="submit" disabled={loading}>
          {loading ? "Generating..." : "Generate"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {output && (
        <>
          <h2>Output</h2>
          <pre>{output}</pre>
        </>
      )}
    </main>
  )
}

export default App
