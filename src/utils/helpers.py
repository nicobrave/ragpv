"""
Funciones de ayuda y utilidades para el proyecto.
"""
from typing import List
from langchain_core.documents import Document

def documents_to_string(documents: List[Document]) -> str:
    """
    Convierte una lista de objetos Document de LangChain en un solo string,
    formateando cada documento con su fuente y contenido.
    """
    if not documents:
        return ""
    
    formatted_docs = []
    for doc in documents:
        # Asumiendo que 'source' está en los metadatos del documento
        source = doc.metadata.get('source', 'Fuente desconocida')
        content = doc.page_content
        formatted_docs.append(f"Fuente: {source}\nContenido: {content}\n---")
        
    return "\n".join(formatted_docs)
