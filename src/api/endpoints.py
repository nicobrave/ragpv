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

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_agent(request: QueryRequest):
    """
    Recibe una consulta, recupera contexto relevante y genera una respuesta.
    """
    try:
        # Recuperar contexto basado en la consulta
        relevant_docs = retriever.retrieve_context(request.query)
        context_str = documents_to_string(relevant_docs)

        # Formatear el historial si existe
        history_str = ""
        if request.history:
            history_str = "\n".join([f"{msg.rol}: {msg.contenido}" for msg in request.history])

        # Generar una respuesta usando el contexto y el historial
        final_response = generate_response(request.query, context_str, history_str)

        return QueryResponse(
            response=final_response,
            source_documents=[doc.dict() for doc in relevant_docs]
        )
    except Exception as e:
        # Loggear el error sería una buena práctica aquí
        raise HTTPException(status_code=500, detail=str(e))
