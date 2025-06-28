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

def consultar_bd(tabla: str, operacion: str, p_columna_regex: str = None, filtros: dict = None) -> any:
    """
    Ejecuta una consulta en Supabase y devuelve un resultado serializable.
    """
    logging.info(f"Ejecutando consultar_bd: tabla={tabla}, operacion={operacion}, filtros={filtros}, regex={p_columna_regex}")

    # Operaciones que usan RPC para extraer con Regex
    if operacion.upper() in ["SUM", "AVG", "MAX", "MIN"]:
        if not p_columna_regex or not filtros:
            raise ValueError("SUM/AVG/MAX/MIN requieren p_columna_regex y filtros.")
        response = supabase.rpc('agregar_columna', {
            'p_operacion': operacion.upper(),
            'p_columna_regex': p_columna_regex,
            'p_filtros': filtros
        }).execute()
        return _to_serializable(response.data)
    elif operacion.upper() == "SELECT DISTINCT":
        if not p_columna_regex or not filtros:
            raise ValueError("SELECT DISTINCT requiere p_columna_regex y filtros.")
        response = supabase.rpc('seleccionar_distintos', {
            'p_columna_regex': p_columna_regex,
            'p_filtros': filtros
        }).execute()
        return _to_serializable(response.data)

    # Operaciones que usan la query builder estándar de Supabase
    query = supabase.table(tabla)
    
    if operacion.upper() == "COUNT":
        # Para COUNT, el filtro se aplica después de la selección
        count_query = query.select('*', count='exact')
        if filtros:
            for campo, valor in filtros.items():
                count_query = count_query.ilike(campo, f'%{valor}%')
        response = count_query.execute()
        return response.count
    
    if operacion.upper() == "SELECT":
        # Para SELECT, se aplica el filtro sobre la selección
        select_query = query.select('fragmento')
        if filtros:
            for campo, valor in filtros.items():
                select_query = select_query.ilike(campo, f'%{valor}%')
        
        # Se limita a 10 para no sobrecargar el contexto del LLM
        response = select_query.limit(10).execute()
        return _to_serializable(response.data)

    raise ValueError(f"Operación no soportada: {operacion}")

# Declaración de la herramienta para Gemini
consultar_bd_tool = {
    "name": "consultar_bd",
    "description": "Consulta la base de datos logística con filtros y operaciones flexibles. Úsala para contar, agregar, o recuperar información específica.",
    "parameters": {
        "type": "object",
        "properties": {
            "tabla": {"type": "string", "description": "Nombre de la tabla a consultar (siempre 'documentos_embeddings')."},
            "operacion": {"type": "string", "description": "Operación: SELECT, COUNT, SUM, AVG, MAX, MIN, SELECT DISTINCT."},
            "columna": {"type": "string", "description": "Regex para extraer un valor de la columna 'fragmento'. Requerido para SUM, AVG, MAX, MIN, SELECT DISTINCT."},
            "filtros": {"type": "object", "description": "Filtros a aplicar sobre la columna 'fragmento' (clave: valor)."}
        }
    }
}
tools = genai_types.Tool(function_declarations=[consultar_bd_tool])
gemini_pro_model = GenerativeModel("gemini-2.5-pro", tools=[tools])
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

## Esquema de la Base de Datos
- **Tabla principal**: `documentos_embeddings`
  - **Descripción**: Contiene fragmentos de texto de un Excel sobre operaciones logísticas.
  - **Columnas relevantes**: `fragmento`. Todos los datos están en esta columna.

## Operaciones disponibles:
- `SELECT`: para recuperar los fragmentos de texto completos que coinciden con los filtros. Úsalo para preguntas generales de información.
- `COUNT`: para contar filas.
- `SUM`, `AVG`, `MAX`, `MIN`: para operaciones numéricas.
- `SELECT DISTINCT`: para obtener listas de valores únicos.

## Instrucciones
1.  Analiza la pregunta del usuario y el historial.
2.  Si la pregunta requiere una operación en la BD, llama a la función `consultar_bd`.
3.  **Para el parámetro 'columna', proporciona un patrón de expresión regular (regex) de PostgreSQL para extraer el valor del texto del 'fragmento'. Este parámetro es OBLIGATORIO para SUM, AVG, MAX, MIN y SELECT DISTINCT.**
4.  Para la operación `SELECT` y `COUNT`, no necesitas proporcionar el parámetro 'columna'.
5.  **Usa siempre la tabla `documentos_embeddings`.**
6.  Aplica filtros de texto sobre la columna `fragmento`.

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
                
                # --- INICIO: CORRECCIÓN Y LOGS DE DEPURACIÓN ---
                logging.basicConfig(level=logging.INFO)
                
                # Convertir los argumentos de Gemini a un dict de Python estándar
                args_dict = _to_serializable(function_call.args)
                
                # Adaptar el nombre del parámetro para la llamada a la función Python
                if 'columna' in args_dict:
                    args_dict['p_columna_regex'] = args_dict.pop('columna')
                
                # Rellenar tabla por defecto si Gemini no la proporciona
                if 'tabla' not in args_dict:
                    args_dict['tabla'] = 'documentos_embeddings'

                # --- INICIO: MEDIDA DE SEGURIDAD ---
                # Forzar que todos los filtros de texto apunten a la columna 'fragmento'
                if 'filtros' in args_dict and args_dict['filtros']:
                    # Concatenamos todos los valores de los filtros en uno solo para buscar en 'fragmento'
                    search_value = " ".join(str(v) for v in args_dict['filtros'].values())
                    args_dict['filtros'] = {'fragmento': search_value}
                # --- FIN: MEDIDA DE SEGURIDAD ---

                logging.info("--- INICIANDO LLAMADA A FUNCIÓN (CORREGIDO) ---")
                logging.info(f"Argumentos recibidos de Gemini (tipo original): {type(function_call.args)}")
                logging.info(f"Argumentos convertidos a dict: {args_dict}")
                # --- FIN: CORRECCIÓN Y LOGS DE DEPURACIÓN ---

                resultado_crudo = consultar_bd(**args_dict)

                # --- INICIO: LOGS DE DEPURACIÓN ---
                logging.info(f"Resultado de consultar_bd (tipo): {type(resultado_crudo)}")
                logging.info(f"Resultado de consultar_bd (valor): {resultado_crudo}")
                logging.info("--- FIN DE LLAMADA A FUNCIÓN ---")
                # --- FIN: LOGS DE DEPURACIÓN ---

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
        # Loggear el error sería una buena práctica aquí
        raise HTTPException(status_code=500, detail=str(e))
