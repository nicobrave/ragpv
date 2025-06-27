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

def generate_response(query: str, context_str: str) -> str:
    """
    Genera una respuesta utilizando un LLM, decidiendo si usar el contexto
    proporcionado o su conocimiento general.
    """
    prompt = f"""
Eres un asistente experto en análisis de datos de una empresa de logística, pero también puedes responder preguntas de conocimiento general.
Tu objetivo es responder a la pregunta del usuario de la forma más precisa posible.

Primero, examina el siguiente contexto extraído de la base de datos.
Contexto:
---
{context_str if context_str else "No se proporcionó contexto."}
---

Pregunta: {query}

INSTRUCCIONES:
1. Si el contexto parece relevante y contiene la respuesta a la pregunta, basa tu respuesta únicamente en la información del contexto.
2. Si el contexto está vacío o no contiene información relevante para responder la pregunta, ignora el contexto por completo y responde la pregunta usando tu conocimiento general.
3. Responde siempre de forma directa y concisa.

Respuesta:
"""

    try:
        response = GENERATIVE_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error al generar respuesta con Gemini: {e}")
        return "Hubo un error al intentar generar la respuesta."
