import os
import json
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, "../models/meta-llama-3-8b-instruct-abliterated-v3-q4_k_m.gguf"))

MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

app = FastAPI(
    title="Offline Investigative Assistant",
    description="Secure, air-gapped law enforcement note analysis engine powered by local Llama 3.",
    version="1.0"
)

#Flexible Model Initialization
def load_local_model():
    use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
    
    n_gpu_layers = -1 if use_gpu else 0
    n_threads = 4 if use_gpu else max(1, os.cpu_count() - 1)
    
    print(f"[*] Initializing model | Mode: {'NVIDIA CUDA (GPU)' if use_gpu else 'CPU-Only'} | Threads: {n_threads}")
    print(f"[*] Loading model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at expected path: {MODEL_PATH}")

    return Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=False
    )

# Load model into memory upon startup
llm = load_local_model()

class AnalysisRequest(BaseModel):
    raw_notes: str = Field(..., description="Raw text notes to be analyzed.")
    existing_case_json: Optional[Dict[str, Any]] = Field(None, description="Optional previous JSON state file to 'remind' the LLM of past analysis.")

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Investigative Assistant</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; background: #f4f4f9; color: #333; }
            h1 { color: #2c3e50; margin-bottom: 5px; }
            p { color: #666; margin-top: 0; }
            .container { display: flex; gap: 20px; margin-top: 20px; }
            .panel { flex: 1; display: flex; flex-direction: column; }
            
            /* Drag and Drop Zone */
            .drop-zone {
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                padding: 30px;
                text-align: center;
                background: #fff;
                cursor: pointer;
                transition: all 0.2s ease;
                margin-bottom: 15px;
            }
            .drop-zone.dragover {
                border-color: #2c3e50;
                background: #f1f5f9;
            }
            .drop-zone p { margin: 0; color: #475569; font-weight: 500; }
            
            textarea { width: 100%; height: 200px; padding: 12px; font-size: 14px; border: 1px solid #cbd5e1; border-radius: 6px; box-sizing: border-box; resize: vertical; background: #fff; }
            
            .btn-group { display: flex; gap: 10px; margin-top: 10px; }
            button { background: #2c3e50; color: white; border: none; padding: 12px 20px; font-size: 16px; border-radius: 6px; cursor: pointer; font-weight: bold; flex: 1; }
            button:hover { background: #34495e; }
            button.secondary { background: #475569; }
            button.secondary:hover { background: #334155; }
            button:disabled { background: #94a3b8; cursor: not-allowed; }
            
            pre { background: #1e293b; color: #e2e8f0; padding: 15px; border-radius: 6px; overflow-x: auto; height: 350px; margin-top: 0; font-family: monospace; font-size: 13px; box-sizing: border-box; }
        </style>
    </head>
    <body>
        <h1>Offline Investigative Assistant</h1>
        <p>Air-gapped Llama 3 Note Analyzer</p>
        
        <div class="container">
            <div class="panel">
                <h3>Input Notes</h3>
                <div id="dropZone" class="drop-zone">
                    <p>📂 Drag & Drop a <strong>.txt</strong> file here<br><span style="font-size: 12px; color: #94a3b8;">or click to browse</span></p>
                    <input type="file" id="fileInput" accept=".txt" style="display: none;">
                </div>
                <textarea id="rawNotes" placeholder="Or type/paste raw investigative notes manually here..."></textarea>
                <button onclick="analyzeNotes()">Analyze Notes</button>
            </div>
            
            <div class="panel">
                <h3>Structured Case JSON</h3>
                <pre id="output">Drop a file or click analyze to see results...</pre>
                <div class="btn-group">
                    <button id="exportBtn" class="secondary" onclick="exportJson()" disabled>Export JSON</button>
                </div>
            </div>
        </div>

        <script>
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('fileInput');
            let lastResultData = null;

            dropZone.addEventListener('click', () => fileInput.click());

            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });

            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('dragover');
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                if (e.dataTransfer.files.length) {
                    handleFile(e.dataTransfer.files[0]);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length) {
                    handleFile(e.target.files[0]);
                }
            });

            function handleFile(file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('rawNotes').value = e.target.result;
                };
                reader.readAsText(file);
            }

            async function analyzeNotes() {
                const notes = document.getElementById('rawNotes').value.trim();
                const output = document.getElementById('output');
                const exportBtn = document.getElementById('exportBtn');
                
                if (!notes) {
                    output.textContent = "Error: No notes provided. Please drop a .txt file or type notes.";
                    exportBtn.disabled = true;
                    return;
                }

                output.textContent = "Processing notes through local Llama 3 model...";
                exportBtn.disabled = true;
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ raw_notes: notes, existing_case_json: null })
                    });
                    const result = await response.json();
                    lastResultData = result;
                    output.textContent = JSON.stringify(result, null, 2);
                    exportBtn.disabled = false;
                } catch (error) {
                    output.textContent = "Error: " + error.message;
                    exportBtn.disabled = true;
                }
            }

            function exportJson() {
                if (!lastResultData) return;
                
                const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastResultData, null, 2));
                const downloadAnchor = document.createElement('a');
                downloadAnchor.setAttribute("href", dataStr);
                
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                downloadAnchor.setAttribute("download", `case_analysis_${timestamp}.json`);
                
                document.body.appendChild(downloadAnchor);
                downloadAnchor.click();
                downloadAnchor.remove();
            }
        </script>
    </body>
    </html>
    """

@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_notes_endpoint(payload: AnalysisRequest):
    case_context = payload.existing_case_json if payload.existing_case_json else {"status": "initialized"}
    
    user_prompt = f"""
[EXISTING_CASE_MEMORY]
{json.dumps(case_context, indent=2)}

[NEW_NOTES_TO_ANALYZE]
{payload.raw_notes}

Provide the updated comprehensive JSON object matching the standard case schema:
{{
  "case_metadata": {{"case_id": "string", "last_updated": "ISO-8601"}},
  "raw_inputs": [],
  "llm_analysis": {{
    "summary": "string",
    "key_entities": [{{"entity": "", "role": "", "details": ""}}],
    "chronology": [{{"time": "", "event": ""}}],
    "open_questions": []
  }}
}}
"""

    messages = [
        {"role": "system", "content": "You are an offline law enforcement analysis engine. Output raw valid JSON only with no markdown wrapping or conversational text."},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        raw_output = response["choices"][0]["message"]["content"].strip()
        
        try:
            parsed_json = json.loads(raw_output)
        except json.JSONDecodeError:
            if "```json" in raw_output:
                raw_output = raw_output.split("```json")[1].split("```")[0]
            elif "```" in raw_output:
                raw_output = raw_output.split("```")[1].split("```")[0]
            parsed_json = json.loads(raw_output.strip())
            
        return {"status": "success", "data": parsed_json}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")