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

# Cargar variables de entorno para obtener las credenciales
load_dotenv()

# Patrón de regex para extraer las partes clave de un ID de contenedor en cualquier parte del texto:
# 1. Cuatro letras (el prefijo)
# 2. Seis o siete dígitos (el número de serie)
# No usamos el ancla '^' para que pueda encontrarlo en cualquier lugar.
CONTAINER_ID_PATTERN = re.compile(r'([A-Z]{4})\s?(\d{6,7})')

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

    def retrieve_context(self, query: str, match_count: int = 5, match_threshold: float = 0.65) -> List[Dict]:
        """
        Recupera el contexto relevante para una consulta utilizando una estrategia híbrida.
        """
        clean_query = query.strip().upper()

        # Estrategia 1: Búsqueda por palabra clave si se encuentra un ID en CUALQUIER LUGAR de la consulta.
        match = CONTAINER_ID_PATTERN.search(clean_query) # Usamos search() en lugar de match()
        if match:
            prefix, serial_number = match.groups()
            # Priorizamos la búsqueda por ID si se detecta uno.
            print(f"--- INFO: ID de contenedor detectado en la consulta. Buscando prefijo '{prefix}' y número de serie '{serial_number}' ---")
            
            response = self.supabase.table('documentos_embeddings') \
                .select('id, fragmento, fuente') \
                .like('fragmento', f'%{prefix}%') \
                .like('fragmento', f'%{serial_number}%') \
                .limit(match_count) \
                .execute()
            
            if response.data:
                # Si encontramos un resultado con el ID, lo devolvemos directamente.
                for item in response.data:
                    item['similarity'] = 1.0
                return response.data
        
        # Estrategia 2: Búsqueda semántica (fallback si no se encuentra ID o si la búsqueda por ID no da resultados)
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
            return response.data
        return []
