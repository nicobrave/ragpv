"""
Endpoints de la API para el agente RAG.

Este módulo contiene el router de FastAPI y define todas las rutas
relacionadas con la interacción y consulta del agente.
"""
from fastapi import APIRouter, HTTPException
from src.rag_engine.retriever import SupabaseRetriever
from src.rag_engine.generator import generate_response
from .schemas import QueryRequest, QueryResponse
from ..utils.helpers import documents_to_string
from supabase import create_client
import os
import re
import json
import logging

from google.generativeai import GenerativeModel, types as genai_types

# Crear una nueva instancia de APIRouter.
# Este objeto será usado para definir todas las rutas de este módulo.
router = APIRouter()

# Inicializamos el retriever. En una aplicación más compleja, esto se manejaría
# con un sistema de inyección de dependencias de FastAPI.
try:
    retriever = SupabaseRetriever()
except ValueError as e:
    # Si las variables de entorno no están configuradas, el retriever no se puede crear.
    retriever = None
    print(f"ADVERTENCIA: No se pudo inicializar el Retriever. {e}")

# Inicializar cliente de Supabase para guardar historial
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Función auxiliar RECURSIVA para convertir datos a un formato serializable
def _to_serializable(data):
    """
    Convierte de forma recursiva los datos de respuesta de Supabase, incluyendo
    objetos anidados MapComposite, a un formato serializable por JSON.
    """
    if data is None:
        return None
    if isinstance(data, list):
        return [_to_serializable(item) for item in data]
    if hasattr(data, 'keys'): # Trata objetos dict-like como MapComposite
        return {key: _to_serializable(value) for key, value in data.items()}
    return data

def consultar_bd(operacion: str, columna_regex: str = None, filtro_fragmento: str = None) -> any:
    """
    Ejecuta una consulta simplificada en Supabase y devuelve un resultado serializable.
    """
    logging.info(f"Ejecutando consultar_bd: operacion={operacion}, filtro_fragmento={filtro_fragmento}, regex={columna_regex}")

    # Construir los filtros si se proporciona el parámetro.
    # Las funciones RPC esperan un objeto JSON para los filtros.
    p_filtros = None
    if filtro_fragmento:
        # Se busca el texto en cualquier parte del fragmento.
        p_filtros = {'fragmento': f'%{filtro_fragmento}%'}

    # Operaciones que usan RPC para extraer con Regex
    if operacion.upper() in ["SUM", "AVG", "MAX", "MIN"]:
        if not columna_regex or not p_filtros:
            raise ValueError("SUM/AVG/MAX/MIN requieren columna_regex y filtro_fragmento.")
        response = supabase.rpc('agregar_columna', {
            'p_operacion': operacion.upper(),
            'p_columna_regex': columna_regex,
            'p_filtros': p_filtros
        }).execute()
        return _to_serializable(response.data)
    elif operacion.upper() == "SELECT DISTINCT":
        if not columna_regex or not p_filtros:
            raise ValueError("SELECT DISTINCT requiere columna_regex y filtro_fragmento.")
        response = supabase.rpc('seleccionar_distintos', {
            'p_columna_regex': columna_regex,
            'p_filtros': p_filtros
        }).execute()
        return _to_serializable(response.data)

    # Operaciones que usan la query builder estándar de Supabase
    query = supabase.table('documentos_embeddings')
    
    if operacion.upper() == "COUNT":
        query = query.select('*', count='exact')
        if filtro_fragmento:
            query = query.ilike('fragmento', f'%{filtro_fragmento}%')
        response = query.execute()
        return response.count
    
    if operacion.upper() == "SELECT":
        query = query.select('fragmento')
        if filtro_fragmento:
            query = query.ilike('fragmento', f'%{filtro_fragmento}%')
        
        # Se limita a 10 para no sobrecargar el contexto del LLM
        response = query.limit(10).execute()
        return _to_serializable(response.data)

    raise ValueError(f"Operación no soportada: {operacion}")

# Declaración de la herramienta para Gemini (versión simplificada)
consultar_bd_tool = {
    "name": "consultar_bd",
    "description": "Consulta la base de datos logística. Úsala para contar, agregar o recuperar información específica de la columna 'fragmento'.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "operacion": {"type": "STRING", "description": "La operación a realizar: SELECT, COUNT, SUM, AVG, MAX, MIN, SELECT DISTINCT."},
            "columna_regex": {"type": "STRING", "description": "Expresión regular (regex) de PostgreSQL para extraer un valor del 'fragmento'. OBLIGATORIO para SUM, AVG, MAX, MIN y SELECT DISTINCT."},
            "filtro_fragmento": {"type": "STRING", "description": "Texto a buscar en la columna 'fragmento' para filtrar los resultados. Úsalo para buscar por códigos de tractor, IDs de contenedor, etc."}
        },
        "required": ["operacion"]
    }
}

# La estructura de 'tools' espera una lista de diccionarios, donde cada diccionario
# representa una 'Tool' y contiene la clave 'function_declarations'.
tools_list = [{"function_declarations": [consultar_bd_tool]}]

# La forma moderna de pasar herramientas es directamente en la inicialización del modelo.
# Esto evita problemas de compatibilidad de versiones con la clase 'Tool'.
gemini_pro_model = GenerativeModel(
    "gemini-2.5-pro", 
    tools=tools_list,
    generation_config={"temperature": 0.0}
)
gemini_flash_model = GenerativeModel("gemini-1.5-flash") # Modelo para refinamiento

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_agent(request: QueryRequest):
    """
    Recibe una consulta, decide si usar function calling o RAG, y genera una respuesta.
    """
    try:
        # 1. Recuperar historial de conversación
        historial_str = ""
        if request.user_id:
            # Recuperar de Supabase
            historial_db = supabase.table("conversacion_historial").select("*").eq("id_usuario", request.user_id).order("creado_en", desc=True).limit(10).execute()
            if historial_db.data:
                historial_str = "\n".join([f"{msg['rol']}: {msg['contenido']}" for msg in reversed(historial_db.data)])

        # 2. Enriquecer el prompt con el esquema y operaciones
        prompt_con_esquema = f"""
Eres un asistente de base de datos. Tu objetivo es ayudar a los usuarios a consultar una base de datos de logística.
A continuación se describe el esquema y las operaciones disponibles:

# Esquema de la Base de Datos
- **Tabla principal**: `documentos_embeddings`
  - **Descripción**: Contiene fragmentos de texto de un Excel sobre operaciones logísticas.
  - **Columnas relevantes**: `fragmento`. Todos los datos están en esta columna.

## Herramienta disponible: `consultar_bd`
- **Descripción**: Ejecuta operaciones en la base de datos.
- **Parámetros**:
  - `operacion`: `SELECT`, `COUNT`, `SUM`, `AVG`, `MAX`, `MIN`, `SELECT DISTINCT`.
  - `columna_regex`: OBLIGATORIO para `SUM`, `AVG`, `MAX`, `MIN`, `SELECT DISTINCT`. Es una regex para extraer un número o texto.
  - `filtro_fragmento`: El texto a buscar en la base de datos para filtrar.

## Instrucciones Clave
1.  Analiza la pregunta del usuario y el historial. Si es una pregunta de seguimiento, usa el historial para completar los parámetros.
2.  **CRÍTICO**: Si la pregunta contiene un identificador (ej. un código de tractor 'T209', un ID de contenedor), DEBES usarlo en el parámetro `filtro_fragmento`. Las consultas sin filtro no son útiles.
3.  Usa `columna_regex` cuando necesites operar sobre un valor específico dentro del texto (ej. sumar los "Kilos").
4.  Si la pregunta es general y no contiene un filtro claro (ej. "¿quiénes son los conductores?"), NO llames a la función. En su lugar, responde que se necesita más información.

---
**Historial:**
{historial_str if historial_str else "No hay historial previo."}
---
**Pregunta:** {request.query}
"""
        
        # 3. Intentar function calling con Gemini 2.5 Pro
        response = gemini_pro_model.generate_content(prompt_con_esquema)
        
        # 4. Si Gemini decide llamar a una función
        if response.candidates and response.candidates[0].content.parts and response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            if function_call.name == "consultar_bd":
                
                logging.basicConfig(level=logging.INFO)
                
                # Convertir los argumentos de Gemini a un dict de Python estándar
                args_dict = _to_serializable(function_call.args)
                
                logging.info("--- INICIANDO LLAMADA A FUNCIÓN (SIMPLIFICADO) ---")
                logging.info(f"Argumentos recibidos de Gemini: {args_dict}")

                resultado_crudo = consultar_bd(**args_dict)

                logging.info(f"Resultado de consultar_bd (valor): {resultado_crudo}")
                logging.info("--- FIN DE LLAMADA A FUNCIÓN ---")

                # 5. Refinamiento de respuesta con Gemini 1.5 Flash
                prompt_refinamiento = f"""
                Basado en la pregunta del usuario y el resultado de la base de datos, genera una respuesta humana y conversacional.

                **Instrucciones de formato:**
                - Comienza la respuesta indicando que has realizado una consulta o cálculo (ej: "Tras consultar los registros...", "He calculado el total...", "Buscando en la base de datos, he encontrado...").
                - Si el resultado es una lista de ítems, formatéala usando viñetas (ej: `- Item 1\n- Item 2`).
                - No incluyas el resultado crudo en la respuesta final.
                
                Pregunta del usuario: '{request.query}'
                Resultado de la base de datos: '{resultado_crudo}'
                
                Respuesta pulida:
                """
                respuesta_final = gemini_flash_model.generate_content(prompt_refinamiento).text.strip()
                
                # Guardar en historial y devolver
                if request.user_id:
                    supabase.table("conversacion_historial").insert([
                        {"id_usuario": request.user_id, "rol": "user", "contenido": request.query},
                        {"id_usuario": request.user_id, "rol": "ai", "contenido": respuesta_final}
                    ]).execute()
                return QueryResponse(response=respuesta_final)
        
        # 6. Si no hay llamada a función, decidir entre RAG o conocimiento general
        relevant_docs = retriever.retrieve_context(request.query)
        
        if not relevant_docs:
            # No se encontraron documentos, es una pregunta de conocimiento general.
            # Se usa el modelo Flash para una respuesta conversacional, instruyéndolo
            # a ignorar el historial si el tema cambia.
            prompt_general = f"""
            Responde la pregunta del usuario de forma directa y conversacional. El historial de la conversación es sobre logística, pero la pregunta actual podría no estarlo. Ignora el tema anterior si no es relevante para la pregunta actual.
            
            Historial:
            {historial_str if historial_str else "No hay historial previo."}
            
            Pregunta: '{request.query}'
            
            Respuesta:
            """
            respuesta_final = gemini_flash_model.generate_content(prompt_general).text.strip()
        else:
            # Se encontraron documentos, usar el flujo RAG normal.
            context_str = documents_to_string(relevant_docs)
            respuesta_final = generate_response(request.query, context_str, historial_str)
        
        if request.user_id:
            supabase.table("conversacion_historial").insert([
                {"id_usuario": request.user_id, "rol": "user", "contenido": request.query},
                {"id_usuario": request.user_id, "rol": "ai", "contenido": respuesta_final}
            ]).execute()
            
        return QueryResponse(response=respuesta_final, source_documents=[doc.dict() for doc in relevant_docs])

    except Exception as e:
        # Loggear el error de forma explícita para depuración en Render
        logging.error(f"Error no controlado en query_agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
