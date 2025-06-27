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
import logging
from itertools import combinations

# Cargar variables de entorno para obtener las credenciales
load_dotenv()

# Expresión regular para detectar un ID de contenedor (ej. ABCD1234567, TCNU 5754568)
# 4 letras mayúsculas, un espacio opcional, y 6 o 7 dígitos.
container_id_pattern = re.compile(r'([A-Z]{4})\s?(\d{6,7})')

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

    def extractar_valores_relevantes(self, query: str) -> dict:
        """
        Extrae posibles valores relevantes de la consulta para las columnas principales.
        Puedes ampliar fácilmente agregando nuevos patrones.
        """
        resultados = {
            'numero': [],
            'hr': [],
            'cliente': [],
            'contenedor': [],
            'tipo': [],
            'tracto': [],
            'trailer': [],
            'conductor': [],
            'rut': [],
            'estado': [],
            'modalidad': [],
            'origen_nombre': [],
            'destino_nombre': [],
            'fecha_emision': [],
            'produccion': [],
            'rut_conductor': [],
            'fecha_viaje': [],
            'usuario': [],
            'faena': [],
            'eta_edt': [],
            'area_negocio': [],
            'tipo_contenedor': [],
            'sistema': [],
            'descripcion': [],
            'id_contenedor': [],
            'id_conductor': [],
            'piso_chasis': []
        }
        texto = query.upper()

        # NUMERO: secuencia de 6-7 dígitos
        resultados['numero'] = re.findall(r'\b\d{6,7}\b', texto)

        # HR: código alfanumérico terminado en C (ej: 798887C)
        resultados['hr'] = re.findall(r'\b\d{6,7}C\b', texto)

        # CLIENTE: lista de clientes conocidos
        clientes_conocidos = ['GOODYEAR', 'P&G', 'FALABELLA', 'ARCOR']
        for cliente in clientes_conocidos:
            if cliente in texto:
                resultados['cliente'].append(cliente)

        # CONTENEDOR: 4 letras + espacio + 6-7 dígitos + guion + dígito
        resultados['contenedor'] = re.findall(r'[A-Z]{4} \d{6,7}-\d', texto)

        # TIPO: IMPO/EXPO
        resultados['tipo'] = re.findall(r'IMPO|EXPO', texto)

        # TRACTO: T seguido de 2-3 dígitos, opcional paréntesis y placa
        resultados['tracto'] = re.findall(r'T\d{2,3}(?: \([A-Z0-9]+\))?', texto)

        # TRAILER: 6 caracteres alfanuméricos
        resultados['trailer'] = re.findall(r'\b[A-Z0-9]{6}\b', texto)

        # CONDUCTOR: secuencia de 2+ palabras en mayúsculas (excluyendo clientes conocidos)
        nombres = re.findall(r'([A-ZÁÉÍÓÚÑ]{2,}(?: [A-ZÁÉÍÓÚÑ]{2,}){1,})', texto)
        resultados['conductor'] = [n for n in nombres if n not in clientes_conocidos]

        # RUT: números con guion y dígito verificador
        resultados['rut'] = re.findall(r'\b\d{7,8}-[\dkK]\b', texto)

        # ESTADO: lista de estados conocidos
        estados_conocidos = ['CARGA CLIENTE', 'DEVOLUCION VACIO', 'DESCARGA CLIENTE', 'INTERMEDIA']
        for estado in estados_conocidos:
            if estado in texto:
                resultados['estado'].append(estado)

        # MODALIDAD: Empty/Full
        modalidades = ['EMPTY', 'FULL']
        for modalidad in modalidades:
            if modalidad in texto:
                resultados['modalidad'].append(modalidad)

        # ORIGEN_NOMBRE y DESTINO_NOMBRE: lista de lugares frecuentes
        lugares_frecuentes = ['CCTI', 'PLANTA MAIPU', 'P&G MACUL', 'DYC SCL', 'SAN ANTONIO', 'VALPARAÍSO', 'SANTIAGO', 'CD ARCOR', 'CONTOPSA SCL']
        for lugar in lugares_frecuentes:
            if lugar in texto:
                resultados['origen_nombre'].append(lugar)
                resultados['destino_nombre'].append(lugar)

        # FECHA_EMISION y FECHA_VIAJE: fechas YYYY-MM-DD
        fechas = re.findall(r'\d{4}-\d{2}-\d{2}', texto)
        resultados['fecha_emision'] = fechas
        resultados['fecha_viaje'] = fechas

        # PRODUCCION: números grandes (ejemplo simple)
        resultados['produccion'] = re.findall(r'\b\d{4,6}\b', texto)

        # RUT_CONDUCTOR: igual que RUT
        resultados['rut_conductor'] = resultados['rut']

        # USUARIO: nombres propios (puedes ampliar con lista de usuarios conocidos)
        usuarios_conocidos = ['BARBARA RUIZ']
        for usuario in usuarios_conocidos:
            if usuario in texto:
                resultados['usuario'].append(usuario)

        # FAENA: palabras clave conocidas
        faenas = ['LOCALEROS']
        for faena in faenas:
            if faena in texto:
                resultados['faena'].append(faena)

        # ETA_EDT: fechas YYYY-MM-DD
        resultados['eta_edt'] = fechas

        # AREA_NEGOCIO: palabras clave conocidas
        areas_negocio = ['CONTENEDORES']
        for area in areas_negocio:
            if area in texto:
                resultados['area_negocio'].append(area)

        # TIPO_CONTENEDOR: ejemplos
        tipos_contenedor = ['40 HC']
        for tipo_c in tipos_contenedor:
            if tipo_c in texto:
                resultados['tipo_contenedor'].append(tipo_c)

        # SISTEMA: ejemplos
        sistemas = ['TMS']
        for sistema in sistemas:
            if sistema in texto:
                resultados['sistema'].append(sistema)

        # DESCRIPCION: palabras clave conocidas (puedes ampliar)
        descripciones = ['NO']
        for desc in descripciones:
            if desc in texto:
                resultados['descripcion'].append(desc)

        # ID_CONTENEDOR: números grandes (ejemplo simple)
        resultados['id_contenedor'] = re.findall(r'\b\d{6}\b', texto)

        # ID_CONDUCTOR: números grandes (ejemplo simple)
        resultados['id_conductor'] = re.findall(r'\b\d{4,6}\b', texto)

        # PISO_CHASIS: números pequeños (ejemplo simple)
        resultados['piso_chasis'] = re.findall(r'\b\d{1,3}\b', texto)

        logging.info(f"[EXTRACCIÓN] Consulta: '{query}' => {resultados}")
        return resultados

    def retrieve_context(self, query: str, match_count: int = 5, match_threshold: float = 0.65) -> List[Document]:
        """
        Recupera el contexto relevante para una consulta utilizando una estrategia híbrida.
        Siempre devuelve una lista de objetos Document.
        """
        # Loguear extracción de valores relevantes
        valores = self.extractar_valores_relevantes(query)

        clean_query = query.strip().upper()
        results_data = []

        # Estrategia 1: Búsqueda directa flexible por campos relevantes
        def build_and_execute(condiciones):
            query_builder = self.supabase.table('documentos_embeddings').select('fragmento, fuente')
            for columna, valor in condiciones:
                # Si es fecha, busca solo año-mes-día con wildcard
                if columna == 'fragmento' and re.match(r'\d{4}-\d{2}-\d{2}', valor):
                    query_builder = query_builder.ilike(columna, f'%{valor}%')
                else:
                    query_builder = query_builder.ilike(columna, f'%{valor}%')
            query_builder = query_builder.limit(10)  # Limita a 10 resultados
            response = query_builder.execute()
            if response.data:
                print(f"--- INFO: Encontrados {len(response.data)} fragmentos por búsqueda directa flexible con condiciones: {condiciones} ---")
            else:
                print(f"--- INFO: Sin resultados por búsqueda directa flexible con condiciones: {condiciones} ---")
            return response.data if response.data else []

        # Mapeo de campo extraído a columna en la tabla
        mapeo_columnas = {
            'tracto': 'fragmento',
            'cliente': 'fragmento',
            'contenedor': 'fragmento',
            'conductor': 'fragmento',
            'rut': 'fragmento',
            'estado': 'fragmento',
            'modalidad': 'fragmento',
            'origen_nombre': 'fragmento',
            'destino_nombre': 'fragmento',
            'fecha_emision': 'fragmento',
            'faena': 'fragmento',
            'tipo': 'fragmento',
            'trailer': 'fragmento',
            'hr': 'fragmento',
            'numero': 'fragmento',
            'produccion': 'fragmento',
            'usuario': 'fragmento',
            'eta_edt': 'fragmento',
            'area_negocio': 'fragmento',
            'tipo_contenedor': 'fragmento',
            'sistema': 'fragmento',
            'descripcion': 'fragmento',
            'id_contenedor': 'fragmento',
            'id_conductor': 'fragmento',
            'piso_chasis': 'fragmento',
        }
        # Solo usar campos con valores extraídos
        condiciones = []
        for campo, columna in mapeo_columnas.items():
            for valor in valores.get(campo, []):
                # Si es fecha, extrae solo año-mes-día
                if campo.startswith('fecha') and len(valor) >= 10:
                    condiciones.append((columna, valor[:10]))
                else:
                    condiciones.append((columna, valor))

        # 1. Intentar con todos los campos extraídos
        if condiciones:
            print(f"--- INFO: Probando búsqueda directa flexible con todos los campos: {condiciones} ---")
            results_data = build_and_execute(condiciones)

        # 2. Si no hay resultados, probar con combinaciones de los dos campos más relevantes
        if not results_data:
            # Definir prioridad de campos
            prioridad = ['tracto', 'cliente', 'fecha_emision', 'contenedor', 'conductor']
            valores_relevantes = [(campo, mapeo_columnas[campo], valores[campo][0]) for campo in prioridad if valores.get(campo)]
            # Probar todas las combinaciones de 2 campos
            for combo in combinations(valores_relevantes, 2):
                condiciones_combo = [(c[1], c[2]) for c in combo]
                print(f"--- INFO: Probando búsqueda directa flexible con combinación: {condiciones_combo} ---")
                results_data = build_and_execute(condiciones_combo)
                if results_data:
                    break

        # Estrategia 3: Búsqueda semántica (fallback si la directa no da resultados)
        if not results_data:
            print(f"--- INFO: No se detectó coincidencia directa o no hubo resultados. Usando búsqueda semántica para '{query}' ---")
            query_embedding = self._create_embedding(query)
            if not query_embedding:
                return []
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
