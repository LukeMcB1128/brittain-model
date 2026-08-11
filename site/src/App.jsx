import { useState, useEffect } from 'react'
import './App.css'

const API_ORIGIN = 'https://fragility-devoutly-dazzling.ngrok-free.dev';

// ngrok's free tunnels send an HTML warning page with status 200 to browser
// requests unless this header is present. That page is not an API response and
// has no CORS headers, so fetch reports it as a CORS/network failure.
const apiHeaders = {
  'ngrok-skip-browser-warning': 'true',
};

async function readNdjson(response, onChunk) {
  if (!response.body) {
    throw new Error('The server did not return a response stream.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = '';

  while (true) {
    const { value, done } = await reader.read();
    pending += decoder.decode(value, { stream: !done });

    const lines = pending.split('\n');
    pending = lines.pop();

    for (const line of lines) {
      if (!line.trim()) continue;
      onChunk(JSON.parse(line));
    }

    if (done) break;
  }

  if (pending.trim()) {
    onChunk(JSON.parse(pending));
  }
}

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
      const response = await fetch(`${API_ORIGIN}/api/tags`, {
        headers: apiHeaders,
      });

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
      const response = await fetch(`${API_ORIGIN}/api/generate`, {
        method: "POST",
        headers: {
          ...apiHeaders,
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
          stream: true,
          options: {
            num_predict: 1024,
            temperature: 0.2,
          },
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      await readNdjson(response, (chunk) => {
        if (chunk.response) {
          setOutput((current) => current + chunk.response);
        }
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <h1>Brittain API Chat</h1>

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
              {item.name}
            </option>
          ))}
        </select>

        <div className="generation-area">

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
        </div>

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

      <div className="model-details">
        <strong>Model Details:</strong>
        <p>{selectedModel?.name}</p>
        <p>{selectedModel?.details?.parameter_size ?? "unknown"} parameters</p>
        <p>{selectedModel?.context} token context</p>
        <p>Languages supported: {selectedModel?.details.languages ?? "unknown"}</p>
        <p>Mode: {mode}</p>
        <p>Note: these models are not all Brittain models, just our current best models.</p>
      </div>
    </main>
  )
}

export default App
