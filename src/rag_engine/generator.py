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
    Genera una respuesta utilizando un LLM. La estrategia del prompt cambia
    dependiendo de si se encontró contexto relevante o no.
    """
    prompt = ""

    if context_str:
        # Si hay contexto, actuamos como un experto en logística basado en datos.
        prompt = f"""
Eres un asistente experto en análisis de datos de una empresa de logística.
Tu principal objetivo es responder basándote en el siguiente contexto extraído de la base de datos operativa.
El contexto está estructurado como una serie de registros, donde cada registro contiene campos en formato 'CLAVE: VALOR' separados por '|'.

Contexto:
{context_str}

Pregunta: {query}

Analiza el contexto, encuentra el registro relevante que coincida con la pregunta y extrae el valor del campo solicitado. Responde de forma concisa y directa.
Respuesta:
"""
    else:
        # Si no hay contexto, actuamos como un asistente de conocimiento general.
        prompt = f"""
Eres un asistente servicial. Responde la siguiente pregunta de la manera más directa y útil posible.

Pregunta: {query}

Respuesta:
"""

    try:
        response = GENERATIVE_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error al generar respuesta con Gemini: {e}")
        return "Hubo un error al intentar generar la respuesta."
