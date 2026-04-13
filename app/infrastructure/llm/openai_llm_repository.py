"""
Adaptador OpenAI para el puerto LLM: prompts orientados a RAG con citas [n] y baja temperatura
para reducir invención cuando el contexto es el único sustento permitido.
"""
from typing import Any

from openai import OpenAI

from app.core.config import get_settings
from app.domain.repositories.llm_repository import LLMRepository


class OpenAILLMRepository(LLMRepository):
    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.openai_model
        self._api_key = settings.openai_api_key.strip()
        self._client = OpenAI(api_key=self._api_key) if self._api_key else None

    def generate(self, question: str, context: str, *, timing: Any = None) -> str:
        if self._client is None:
            # Permite arrancar la API sin clave (fallo explícito en la respuesta en lugar de excepción).
            return "No hay OPENAI_API_KEY configurada."

        system_prompt = (
            "Eres un asistente experto en medicina tradicional china, fitoterapia, dietetica, nutrición, anatomía, fisiología y patología para un sistema RAG. "
            "REGLA FUNDAMENTAL: responde EXCLUSIVAMENTE con informacion que aparezca de forma explicita en el CONTEXTO RECUPERADO. "
            "NO añadas datos, ejemplos, clasificaciones, descripciones anatomicas ni detalles que conozcas pero que NO esten escritos en los fragmentos proporcionados. "
            "Aunque sepas que un dato es correcto, si no esta en el contexto, NO lo incluyas. "
            "Si el contexto no contiene datos suficientes para responder completamente, indica de forma explicita que los fragmentos recuperados no cubren ese aspecto. "
            "Mantente conciso, en espanol, con lenguaje claro y profesional. "
            "Responde de forma estructurada. "
            "Rechaza educadamente cualquier pregunta fuera del contexto."
        )
        user_prompt = (
            f"Pregunta del usuario:\n{question}\n\n"
            "CONTEXTO RECUPERADO. Cada bloque tiene ref=[n], Documento y Pagina; usa SOLO esos datos para citar.\n"
            f"{context}\n\n"
            "Instrucciones de salida (obligatorio):\n"
            "1) En el cuerpo, cita con [n] segun el fragmento. Si varios fragmentos son del mismo Documento y misma Pagina y apoyan la misma idea, usa un solo [n] o [n][m] en una sola mencion; evita listar [1][3] separados si [1] y [3] son la misma pagina sin necesidad.\n"
            "2) Al final, una seccion titulada exactamente **Referencias**: una linea por cada par **unico** (Documento + Pagina). "
            "Si citaste [1] y [3] y ambos bloques comparten el mismo Documento y la misma Pagina, una sola linea: `[1][3] NombreExacto — pagina X` (refs en orden creciente). "
            "Si Documento o Pagina difieren, lineas distintas. Formato: `[n]` o `[n][m]...` NombreDelArchivo — pagina X (o «sin pagina» si el bloque lo indica).\n"
            "3) No inventes titulos de archivo ni paginas: copialos del campo Documento y Pagina del bloque con ese ref=[n].\n"
            "4) No sustituyas la seccion **Referencias** por un resumen tipo «Fuentes: varios pdf»; manten el listado como arriba.\n"
            "5) Si falta evidencia, indica limites de forma explicita.\n"
            "6) PROHIBIDO aportar informacion propia: si un fragmento menciona un concepto pero no lo detalla, NO completes con tu conocimiento. Limitate a lo que dice el fragmento."
        )

        # temperature baja: respuestas más deterministas y alineadas con el contexto citado.
        if timing is not None:
            with timing.step("openai_chat_completions"):
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    temperature=0.1,
                )
        else:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.1,
            )
        return response.choices[0].message.content or "No se pudo generar una respuesta."
