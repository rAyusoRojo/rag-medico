"""
Punto de entrada HTTP: carga variables de entorno, registra rutas de la API y sirve una UI mínima
para probar /ask y el filtro por documentos sin necesidad de otro frontend.
"""
from dotenv import load_dotenv

# Cargar .env antes de importar módulos que lean configuración (p. ej. get_settings).
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.routes.ask import router as ask_router
from app.api.routes.documents import router as documents_router

app = FastAPI(title="RAG Medico API", version="0.1.0")
app.include_router(ask_router)
app.include_router(documents_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    # Página estática con fetch a /documents y /ask; el marcado y el JS van inline para desplegar un solo archivo.
    return """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Consultorio RAG-MTC</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 900px; }
    h1 { margin-bottom: 12px; }
    label { display: block; margin-bottom: 8px; font-weight: 600; }
    .row { display: flex; gap: 10px; align-items: center; }
    #question { flex: 1; padding: 10px; font-size: 15px; }
    button { padding: 10px 16px; cursor: pointer; }
    .disclaimer { margin-top: 12px; margin-bottom: 6px; color: #666; font-size: 14px; }
    #result { width: 100%; margin-top: 12px; min-height: 220px; padding: 10px; resize: vertical; }
    .hint { margin-top: 8px; color: #555; font-size: 13px; }
    .doc-panel { margin-top: 16px; border: 1px solid #ddd; border-radius: 6px; padding: 0; background: #fafafa; }
    .doc-panel summary {
      list-style: none;
      padding: 10px 12px;
      cursor: pointer;
      font-size: 15px;
      font-weight: 600;
      color: #444;
      user-select: none;
    }
    .doc-panel summary::-webkit-details-marker { display: none; }
    .doc-panel[open] summary { border-bottom: 1px solid #e8e8e8; }
    .doc-inner { padding: 12px; }
    .doc-actions { margin-bottom: 8px; font-size: 13px; }
    .doc-actions button { padding: 4px 10px; margin-right: 8px; cursor: pointer; font-size: 13px; }
    #docList { max-height: 180px; overflow-y: auto; font-size: 13px; color: #333; }
    #docList label { display: block; margin: 4px 0; font-weight: normal; cursor: pointer; }
  </style>
</head>
<body>
  <h1>Cuál es tú pregunta?</h1>
  <div class="row">
    <input id="question" type="text" placeholder="Escribe tu pregunta..." />
    <button id="askBtn" type="button">Consultar</button>
  </div>
  <details class="doc-panel" id="docPanel">
    <summary>Filtrar por documentos concretos (opcional)</summary>
    <div class="doc-inner">
      <p class="hint" style="margin-top:0">Por defecto la búsqueda usa <strong>todos</strong> los documentos indexados. Marca solo los que quieras acotar; si no marcas ninguno, se sigue buscando en todos.</p>
      <div class="doc-actions">
        <button type="button" id="selectAllBtn">Marcar todos</button>
        <button type="button" id="selectNoneBtn">Ninguno</button>
      </div>
      <div id="docList">Despliega este apartado para cargar la lista.</div>
    </div>
  </details>
  <div class="disclaimer">Esta informacion es de caracter informativo y no sustituye la consulta con un profesional sanitario.</div>
  <textarea id="result" readonly placeholder="Aquí aparecerá el resultado..."></textarea>
  <div class="hint">API: <code>/documents</code> y <code>/ask?question=...&sources=...</code></div>

  <script>
    const questionInput = document.getElementById("question");
    const askBtn = document.getElementById("askBtn");
    const resultBox = document.getElementById("result");
    const docList = document.getElementById("docList");
    const docPanel = document.getElementById("docPanel");
    let documentsLoaded = false;

    function getSelectedSources() {
      return [...docList.querySelectorAll("input.doc-cb:checked")].map((cb) => cb.value);
    }

    async function loadDocuments() {
      docList.textContent = "Cargando lista…";
      try {
        const response = await fetch("/documents");
        const data = await response.json();
        docList.innerHTML = "";
        const sources = data.sources || [];
        if (sources.length === 0) {
          docList.textContent = "No hay documentos indexados todavía.";
          return;
        }
        for (const src of sources) {
          const label = document.createElement("label");
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.className = "doc-cb";
          cb.value = src;
          label.appendChild(cb);
          label.appendChild(document.createTextNode(" " + src));
          docList.appendChild(label);
        }
      } catch (e) {
        docList.textContent = "No se pudo cargar la lista de documentos.";
      }
    }

    docPanel.addEventListener("toggle", () => {
      if (docPanel.open && !documentsLoaded) {
        documentsLoaded = true;
        loadDocuments();
      }
    });

    document.getElementById("selectAllBtn").addEventListener("click", () => {
      docList.querySelectorAll("input.doc-cb").forEach((cb) => { cb.checked = true; });
    });
    document.getElementById("selectNoneBtn").addEventListener("click", () => {
      docList.querySelectorAll("input.doc-cb").forEach((cb) => { cb.checked = false; });
    });

    async function askQuestion() {
      const question = questionInput.value.trim();
      if (!question) {
        resultBox.value = "Escribe una pregunta.";
        return;
      }

      askBtn.disabled = true;
      resultBox.value = "Consultando...";
      try {
        const params = new URLSearchParams();
        params.set("question", question);
        getSelectedSources().forEach((s) => params.append("sources", s));
        const response = await fetch(`/ask?${params.toString()}`);
        const data = await response.json();
        if (!response.ok) {
          resultBox.value = data?.detail || "Error en la consulta.";
          return;
        }

        // Respuesta tolerante a futuros campos en AskResponse.
        if (typeof data === "string") {
          resultBox.value = data;
        } else if (data.answer) {
          let text = data.answer;
          const cites = data.citations;
          const hasReferencias = /\\*\\*Referencias\\*\\*/i.test(text);
          if (Array.isArray(cites) && cites.length > 0 && !hasReferencias) {
            text += "\\n\\n---\\nLeyenda de citas (el [n] del texto coincide con estos fragmentos):\\n";
            for (const c of cites) {
              if (c && typeof c.ref === "number" && c.document) {
                const p = c.page != null ? " — pag. " + c.page : " — sin pagina";
                text += "[" + c.ref + "] " + c.document + p + "\\n";
              }
            }
          }
          const srcs = data.sources;
          if (Array.isArray(srcs) && srcs.length > 0 && (!cites || cites.length === 0)) {
            text += "\\n\\n---\\nFuentes:\\n";
            for (const s of srcs) {
              if (typeof s === "string") {
                text += "- " + s + "\\n";
              } else if (s && s.document) {
                const p = s.page != null ? " (p. " + s.page + ")" : "";
                text += "- " + s.document + p + "\\n";
              }
            }
          }
          resultBox.value = text;
        } else {
          resultBox.value = JSON.stringify(data, null, 2);
        }
      } catch (error) {
        resultBox.value = "No se pudo conectar con el servidor.";
      } finally {
        askBtn.disabled = false;
      }
    }

    askBtn.addEventListener("click", askQuestion);
    questionInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        askQuestion();
      }
    });
  </script>
</body>
</html>
"""
