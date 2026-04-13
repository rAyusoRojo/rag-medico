# Tareas (RAG Médico)

Convención: cada tarea lleva **fecha y hora de creación** y **fecha y hora de ejecución**. Hasta que no se complete (o se cancele explícitamente), la ejecución queda como `—`. El listado va **de la más antigua en crearse arriba** a la **más reciente abajo** (orden de aparición en la conversación / sesión).

---

## 1. Medir el retrieval actual y proponer enfoque de evaluación

- **Descripción:** Definir cómo cuantificar recuperación (Hit@K, MRR, precisión) frente a un golden set y trade-offs antes de implementar.
- **Creada:** 2026-04-13 09:00:00
- **Ejecutada:** 2026-04-13 09:45:00

---

## 2. Implementar suite de evaluación de retrieval

- **Descripción:** Scripts `app/services/eval_generate_golden.py` (generación sintética de preguntas) y `app/services/eval_retrieval.py` (métricas contra el golden).
- **Creada:** 2026-04-13 09:30:00
- **Ejecutada:** 2026-04-13 11:00:00

---

## 3. Optimizar parámetros de troceo (CHUNK_SIZE / CHUNK_OVERLAP)

- **Descripción:** A/B según informes de retrieval; p. ej. `CHUNK_SIZE=500`, `CHUNK_OVERLAP=150` en `.env` tras mejora de Hit@1.
- **Creada:** 2026-04-13 10:30:00
- **Ejecutada:** 2026-04-13 12:00:00

---

## 4. Mejorar el prompt del generador del golden dataset

- **Descripción:** Preguntas autocontenidas, sin referencias a «el texto» / «el fragmento», con término médico concreto, en `eval_generate_golden.py`.
- **Creada:** 2026-04-13 11:00:00
- **Ejecutada:** 2026-04-13 12:30:00

---

## 5. Evaluación end-to-end (LLM como juez)

- **Descripción:** Script `app/services/eval_end_to_end.py` (faithfulness, relevancia, correctness sobre pipeline completo).
- **Creada:** 2026-04-13 11:30:00
- **Ejecutada:** 2026-04-13 13:00:00

---

## 6. Analizar resultados de evaluación (retrieval vs generación)

- **Descripción:** Interpretar informes para localizar cuello de botella (recuperación frente a respuesta del LLM).
- **Creada:** 2026-04-13 12:00:00
- **Ejecutada:** 2026-04-13 13:30:00

---

## 7. Endurecer prompts del repositorio LLM (solo contexto recuperado)

- **Descripción:** Ajustes en `app/infrastructure/llm/openai_llm_repository.py` (system/user) para reducir conocimiento externo no citado.
- **Creada:** 2026-04-13 12:30:00
- **Ejecutada:** 2026-04-13 14:00:00

---

## 8. Corregir formato de `.env` (variables pegadas en una línea)

- **Descripción:** Separar p. ej. `VECTOR_TOP_K` y `CHUNK_SIZE` si quedaron concatenados (`VECTOR_TOP_K=5CHUNK_SIZE=500`).
- **Creada:** 2026-04-13 13:00:00
- **Ejecutada:** 2026-04-13 14:15:00

---

## 9. Preparar despliegue con Docker (bajo coste / reproducible)

- **Descripción:** `Dockerfile` (Python 3.12-slim, uvicorn sin reload, healthcheck), `docker-compose.yml` (puerto 8000, volúmenes Chroma y `data/documents`, `env_file`), `.dockerignore`.
- **Creada:** 2026-04-13 13:30:00
- **Ejecutada:** 2026-04-13 15:00:00

---

## 10. Mantener `.gitignore` alineado con secretos y artefactos pesados

- **Descripción:** `.env`, `.chroma/`, `data/eval/`, `data/logs/`, `data/documents/`, `venv/`, `__pycache__/`, `.cursor/`, `PDFSSSS/`, etc.
- **Creada:** 2026-04-13 14:00:00
- **Ejecutada:** 2026-04-13 15:15:00

---

## 11. Endpoint de salud HTTP GET `/health`

- **Descripción:** Comprobación liveness/readiness en `main.py` (`{"status": "ok"}`).
- **Creada:** 2026-04-13 14:15:00
- **Ejecutada:** 2026-04-13 15:20:00

---

## 12. Incluir `chunk_id` en fragmentos recuperados (Chroma)

- **Descripción:** Campo opcional en `RetrievedChunk` y relleno desde metadatos/ids en `app/db/vector_store.py` para trazabilidad en evaluación.
- **Creada:** 2026-04-13 14:30:00
- **Ejecutada:** 2026-04-13 15:25:00

---

## 13. Inicializar Git, commit y subir a GitHub

- **Descripción:** Repositorio local, commits, remoto y push (sin `gh` si no está instalado).
- **Creada:** 2026-04-13 14:45:00
- **Ejecutada:** 2026-04-13 15:45:00

---

## 14. Documento `publicacion.txt` (pasos de publicación) y excluirlo de Git

- **Descripción:** Guía en raíz (Docker, ingesta, comprobación, Cloudflare Tunnel, Hetzner) y línea `publicacion.txt` en `.gitignore`.
- **Creada:** 2026-04-13 15:00:00
- **Ejecutada:** 2026-04-13 16:10:00

---

## 15. Hoja de ruta — Limpiar el golden dataset

- **Descripción:** Paso 2 del plan (calidad/consistencia del JSON antes de más experimentos).
- **Creada:** 2026-04-13 15:30:00
- **Ejecutada:** —

---

## 16. Hoja de ruta — Probar cross-encoder adecuado

- **Descripción:** Paso 3 (p. ej. modelo multilingüe MMARCO); en sesión se priorizó coste/tiempo tras constatar que el cuello no era retrieval.
- **Creada:** 2026-04-13 15:35:00
- **Ejecutada:** 2026-04-13 16:00:00 *(omitida de forma explícita en la sesión)*

---

## 17. Hoja de ruta — Multi-query con LLM + RRF

- **Descripción:** Paso 4 de la hoja de ruta (expansión de consulta + fusión de rankings).
- **Creada:** 2026-04-13 15:40:00
- **Ejecutada:** —

---

## 18. Chroma en modo servidor + HttpClient

- **Descripción:** Montar Chroma como servicio dedicado y conectar la API y la ingesta con `HttpClient`, para no depender de `docker compose restart` tras cada ingesta (evitar el problema de `PersistentClient` con dos procesos).
- **Creada:** 2026-04-13 16:34:57
- **Ejecutada:** —

---

## 19. Rotar clave OpenAI (seguridad)

- **Descripción:** Rotar `OPENAI_API_KEY` en el panel de OpenAI si el `.env` pudo exponerse (chat, capturas, remoto).
- **Creada:** 2026-04-13 16:34:57
- **Ejecutada:** 2026-04-13 16:37:56

---

## 20. Diagnóstico de acceso a `localhost:8000` tras `docker compose up -d`

- **Descripción:** Comprobar Docker Desktop, `docker compose ps` / `logs`, conflicto de puerto, probar `127.0.0.1:8000`; no depende de haber ejecutado ingesta para ver la UI. *(En chat se entregó la guía de diagnóstico; cerrar el incidente en tu PC queda como verificación local.)*
- **Creada:** 2026-04-13 16:45:00
- **Ejecutada:** —

---

*Nota: Las horas de los ítems 1–17 y 20 son **orden lógico de la conversación** en el día 2026-04-13, no marcas de sistema reales. Los ítems 18–19 conservan las marcas de creación que ya tenías en este fichero (Chroma y rotación de clave). Para auditoría exacta, sustituye por timestamps de commits o del historial de Cursor.*
