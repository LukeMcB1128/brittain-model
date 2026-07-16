import sys, json, time, codecs
import torch, tiktoken, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import Brittain, GPTConfig
from sft_prompt import format_prompt

CKPT = sys.argv[1] if len(sys.argv) > 1 else "brittain_124m_sft.pt"
NAME, PORT = "BRITTAIN1:124M", 11435

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available() else
          torch.device("cpu"))
enc = tiktoken.get_encoding("gpt2")
ck = torch.load(CKPT, map_location=device)
model = Brittain(GPTConfig(**ck['cfg'])).to(device); model.load_state_dict(ck['model']);
model.eval()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def stream_pieces(prompt, max_new=400):
    ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for _ in range(max_new):
            ids = model.generate(ids, max_new_tokens=1, temperature=0.5, top_p=0.9, repetition_penalty=1.3)
            nxt = ids[0,-1].item()
            if nxt == enc.eot_token: break
            piece = utf8.decode(enc.decode_single_token_bytes(nxt))
            if piece: yield piece

@app.get("/api/tags")
def tags():
    return {"models": [{"name": NAME+":latest", "model": "BRITTAIN1", "modified_at": now(), "size": 0, "digest": NAME, "details": {"family": NAME, "parameter_size": "124M"}, "context": "1024"}]}

@app.get("/api/version")
def version():
    return {"version": "brittain-0.1"}

@app.post("/api/show")
async def show(req: Request):
    return {"details": {"family": NAME, "parameter_size": "124M"}, "capabilities": ["completion"]}

@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    msgs = body.get("messages", [])
    user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
    prompt = format_prompt(user)                     # single-turn: last user msg = instruction
    def gen():
        for p in stream_pieces(prompt):
            yield json.dumps({"model": NAME, "created_at": now(),
                "message": {"role": "assistant", "content": p}, "done": False}) + "\n"
        yield json.dumps({"model": NAME, "created_at": now(),
            "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}) + "\n"
    if body.get("stream", True):
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    text = "".join(stream_pieces(prompt))
    return JSONResponse({"model": NAME, "created_at": now(),
        "message": {"role": "assistant", "content": text}, "done": True})

@app.post("/api/generate")
async def generate(req: Request):
    body = await req.json()
    prompt = format_prompt(body.get("prompt", ""))
    def gen():
        for p in stream_pieces(prompt):
            yield json.dumps({"model": NAME, "created_at": now(), "response": p, "done": False}) + "\n"
        yield json.dumps({"model": NAME, "created_at": now(), "response": "", "done": True, "done_reason": "stop"}) + "\n"
    if body.get("stream", True):
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    return JSONResponse({"model": NAME, "created_at": now(), "response": "".join(stream_pieces(prompt)), "done": True})

if __name__ == "__main__":
    print(f"BRITTAIN serving on http://localhost:{PORT} as model '{NAME}'")
    uvicorn.run(app, host="127.0.0.1", port=PORT)