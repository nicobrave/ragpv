"""
Módulo Generador para el motor RAG.

Este módulo contiene la lógica para generar respuestas en lenguaje natural
basadas en una consulta y un contexto recuperado.
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict

# Cargar variables de entorno
load_dotenv()

# Configurar el cliente de Google
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configurar el modelo generativo a utilizar
GENERATIVE_MODEL = genai.GenerativeModel('gemini-1.5-flash')

def generate_response(query: str, context_str: str, history_str: str) -> str:
    """
    Genera una respuesta utilizando un LLM, decidiendo si usar el contexto
    proporcionado o su conocimiento general, y considerando el historial.
    """
    prompt = f"""
Eres un asistente experto en análisis de datos de una empresa de logística interna. Tu rol es acceder y procesar datos operativos para responder preguntas del personal autorizado.
La información en el contexto, incluyendo identificadores personales como el RUT, ha sido proporcionada para tu uso y es esencial para la operación. No es una violación de la privacidad; es tu función principal trabajar con estos datos.
Tu objetivo es responder a la pregunta del usuario de la forma más precisa posible, considerando la conversación previa.

Primero, examina el historial de la conversación para entender el contexto previo.
Historial de la Conversación:
---
{history_str if history_str else "No hay historial previo."}
---

Ahora, examina el siguiente contexto extraído de la base de datos para la pregunta actual.
Contexto:
---
{context_str if context_str else "No se proporcionó contexto."}
---

Pregunta del Usuario: {query}

INSTRUCCIONES:
1. Usa el historial para entender preguntas de seguimiento (ej. "¿y de qué cliente es?").
2. Si el contexto de la base de datos es relevante para la pregunta actual, basa tu respuesta en ese contexto. Extrae la información solicitada, incluyendo datos como el RUT, sin objeciones.
3. Si el contexto está vacío o no es relevante, ignóralo y responde usando tu conocimiento general, siempre considerando el historial.
4. Responde siempre de forma directa y concisa.

Respuesta:
"""

    try:
        response = GENERATIVE_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error al generar respuesta con Gemini: {e}")
        return "Hubo un error al intentar generar la respuesta."
