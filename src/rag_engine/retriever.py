"""
Módulo Retriever para el motor RAG.

Este módulo contiene la lógica para recuperar información relevante
de la base de datos vectorial (Supabase).
"""
import os
import re
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Dict, Optional
from langchain_core.documents import Document

# Cargar variables de entorno para obtener las credenciales
load_dotenv()

# Expresión regular para detectar un ID de contenedor (ej. ABCD123456, ABCD 123456)
# 4 letras mayúsculas, un espacio opcional, y 6 dígitos.
container_id_pattern = re.compile(r'([A-Z]{4})\s?(\d{6})')

class SupabaseRetriever:
    """
    Una clase para recuperar documentos de texto relevantes desde Supabase
    basándose en la similitud de embeddings (búsqueda semántica) o
    búsqueda de texto directo para IDs (búsqueda por palabra clave).
    """
    def __init__(self):
        """
        Inicializa el cliente de Supabase y configura la API de Google.
        """
        # Cargar variables de entorno
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        google_api_key = os.environ.get("GOOGLE_API_KEY")

        if not all([supabase_url, supabase_key, google_api_key]):
            raise ConnectionError("Las variables de entorno de Supabase y Google deben estar configuradas.")

        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Configurar la API de Google Gemini
        genai.configure(api_key=google_api_key)
        self.embed_model = "models/embedding-001"

    def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Crea un embedding para un texto dado usando Google Gemini."""
        try:
            # Especificar el tipo de tarea es crucial para la precisión de la búsqueda
            result = genai.embed_content(
                model=self.embed_model,
                content=text,
                task_type="RETRIEVAL_QUERY"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error al crear embedding: {e}")
            return None

    def retrieve_context(self, query: str, match_count: int = 5, match_threshold: float = 0.65) -> List[Document]:
        """
        Recupera el contexto relevante para una consulta utilizando una estrategia híbrida.
        Siempre devuelve una lista de objetos Document.
        """
        clean_query = query.strip().upper()
        results_data = []

        # Estrategia 1: Búsqueda por palabra clave si se encuentra un ID en CUALQUIER LUGAR de la consulta.
        match = container_id_pattern.search(clean_query) # Usamos search() en lugar de match()
        if match:
            prefix, serial_number = match.groups()
            print(f"--- INFO: ID de contenedor detectado en la consulta. Buscando prefijo '{prefix}' y número de serie '{serial_number}' ---")
            
            # Ejecutar la consulta de texto directo, buscando el prefijo Y el número por separado
            # para ser robusto a formatos como 'DRYU 1234567-8'
            response = self.supabase.table('documentos_embeddings') \
                .select('fragmento, fuente') \
                .ilike('fragmento', f'%{prefix}%') \
                .ilike('fragmento', f'%{serial_number}%') \
                .execute()
            
            if response.data:
                print(f"--- INFO: Encontrados {len(response.data)} fragmentos por búsqueda directa. ---")
                results_data = response.data
        
        # Estrategia 2: Búsqueda semántica (fallback si no se encuentra ID o si la búsqueda por ID no da resultados)
        if not results_data:
            print(f"--- INFO: No se detectó ID o no hubo resultados. Usando búsqueda semántica para '{query}' ---")
            query_embedding = self._create_embedding(query)
            if not query_embedding:
                return []

            # Llamar a la función RPC en Supabase
            response = self.supabase.rpc('match_documentos', {
                'query_embedding': query_embedding,
                'match_threshold': match_threshold,
                'match_count': match_count,
            }).execute()

            if response.data:
                results_data = response.data

        # Estandarizar la salida a una lista de objetos Document
        documents = [
            Document(
                page_content=item.get('fragmento', ''), 
                metadata={
                    'source': item.get('fuente', 'desconocido'), 
                    'id': item.get('id', 0),
                    'similarity': item.get('similarity', None)
                }
            ) for item in results_data
        ]
        
        return documents
