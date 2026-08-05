import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [prompt, setPrompt] = useState('');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [suffix, setSuffix] = useState('');

  async function loadModels() {
    try {
      const response = await fetch("https://fragility-devoutly-dazzling.ngrok-free.dev/api/tags");

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

  const selectedModel = models.find((item) => item.name === model);
  const mode = selectedModel?.mode ?? "raw";

  async function generate(event) {
    event.preventDefault();

    if (!prompt.trim()) {
      return;
    }

    setLoading(true);
    setError("");
    setOutput("");

    try {
      const response = await fetch("https://fragility-devoutly-dazzling.ngrok-free.dev/api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          prompt,
          // The server applies the correct format from this request shape.
          // FIM needs both sides of the cursor. Instruct checkpoints need the
          // server's Alpaca template, so they must not use raw mode.
          suffix: mode === "fim" ? suffix : undefined,
          raw: mode !== "instruct",
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

      const data = await response.json();
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
          disabled={models.length === 0}
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
          rows="10"
        />

        {mode === "fim" && (
          <>
            <label htmlFor="suffix">Code after cursor</label>

            <textarea
              id="suffix"
              value={suffix}
              onChange={(event) => setSuffix(event.target.value)}
              placeholder="Optional code after the missing section"
              rows="6"
            />
          </>
        )}

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
