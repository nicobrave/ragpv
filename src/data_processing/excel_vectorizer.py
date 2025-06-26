import os
import pandas as pd
import tiktoken
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv
import time
from typing import List, Dict, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class ExcelVectorizer:
    def __init__(self, chunk_size: int = 10, tokens_per_chunk: int = 500):
        # Configuración de clientes
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not all([self.supabase_url, self.supabase_key, self.google_api_key]):
            raise ValueError("Las variables de entorno de Supabase y Google deben estar configuradas.")

        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        genai.configure(api_key=self.google_api_key)
        
        # Configuración de embedding y chunking
        self.embed_model = "models/embedding-001"
        self.chunk_size = chunk_size
        self.tokens_per_chunk = tokens_per_chunk
        # Usaremos tiktoken para estimar, aunque no sea 100% preciso para Gemini, es una buena aproximación.
        self.encoding = tiktoken.encoding_for_model("gpt-4")

    def _read_excel(self, file_path: str) -> Optional[pd.DataFrame]:
        """Lee el archivo Excel."""
        try:
            df = pd.read_excel(file_path)
            logger.info(f"Archivo Excel cargado: {len(df)} filas, {len(df.columns)} columnas")
            return df
        except Exception as e:
            logger.error(f"Error al leer Excel: {e}")
            return None

    def _format_row_to_text(self, row: pd.Series) -> str:
        """Convierte una fila del DataFrame a texto estructurado."""
        texto_fila = []
        for col, valor in row.items():
            if pd.isna(valor):
                valor = "N/A"
            elif isinstance(valor, (int, float)):
                valor = str(valor)
            elif isinstance(valor, pd.Timestamp):
                valor = valor.strftime('%Y-%m-%d %H:%M:%S')
            texto_fila.append(f"{col}: {valor}")
        return " | ".join(texto_fila)

    def _create_chunks(self, df: pd.DataFrame) -> List[Dict]:
        """Divide el DataFrame en chunks de texto."""
        chunks = []
        for i in range(0, len(df), self.chunk_size):
            chunk_df = df.iloc[i:i + self.chunk_size]
            texto_chunk = "\n".join([self._format_row_to_text(row) for _, row in chunk_df.iterrows()])
            
            tokens = self.encoding.encode(texto_chunk)
            if len(tokens) > self.tokens_per_chunk:
                sub_chunks = self._split_chunk_by_tokens(texto_chunk)
                for j, sub_chunk_text in enumerate(sub_chunks):
                    chunks.append({
                        'text': sub_chunk_text,
                        'chunk_id': f"chunk_{i}_{j}",
                        'metadata': {'filas_inicio': i, 'filas_fin': min(i + self.chunk_size, len(df))}
                    })
            else:
                chunks.append({
                    'text': texto_chunk,
                    'chunk_id': f"chunk_{i}",
                    'metadata': {'filas_inicio': i, 'filas_fin': min(i + self.chunk_size, len(df))}
                })
        return chunks

    def _split_chunk_by_tokens(self, text: str) -> List[str]:
        """Divide un texto por tokens si es demasiado largo."""
        tokens = self.encoding.encode(text)
        sub_chunks = []
        for i in range(0, len(tokens), self.tokens_per_chunk):
            sub_tokens = tokens[i:i + self.tokens_per_chunk]
            sub_chunks.append(self.encoding.decode(sub_tokens))
        return sub_chunks

    def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Crea embedding para un texto usando Google Gemini."""
        try:
            # Especificar el tipo de tarea es crucial para la precisión de la búsqueda
            result = genai.embed_content(
                model=self.embed_model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error al crear embedding con Google: {e}")
            return None

    def _insert_into_supabase(self, chunk_data: Dict, embedding: List[float], file_name: str):
        """Inserta el chunk y su embedding en Supabase."""
        try:
            data_to_insert = {
                "fuente": file_name,
                "chunk_id": chunk_data['chunk_id'],
                "fragmento": chunk_data['text'],
                "embedding": embedding,
                "metadata": {**chunk_data['metadata'], "tipo_datos": "operacional"}
            }
            self.supabase.table("documentos_embeddings").insert(data_to_insert).execute()
            return True
        except Exception as e:
            logger.error(f"Error al insertar en Supabase: {e}")
            return False

    def process_file(self, file_path: str):
        """Procesa un archivo Excel completo y genera embeddings."""
        df = self._read_excel(file_path)
        if df is None:
            logger.error("No se pudo cargar el archivo Excel, deteniendo proceso.")
            return

        logger.info(f"Creando chunks de {self.chunk_size} filas cada uno...")
        chunks = self._create_chunks(df)
        logger.info(f"Se crearon {len(chunks)} chunks para procesar.")

        file_name = os.path.basename(file_path)
        successful_inserts, failed_inserts = 0, 0

        for i, chunk in enumerate(chunks):
            if i > 0 and i % 10 == 0:
                logger.info(f"Pausa para evitar rate limiting... ({i}/{len(chunks)})")
                time.sleep(2)

            embedding = self._create_embedding(chunk['text'])
            if not embedding:
                logger.warning(f"No se pudo crear embedding para chunk {chunk['chunk_id']}")
                failed_inserts += 1
                continue

            if self._insert_into_supabase(chunk, embedding, file_name):
                successful_inserts += 1
                logger.info(f"[{i+1}/{len(chunks)}] Chunk {chunk['chunk_id']} procesado exitosamente.")
            else:
                failed_inserts += 1
                logger.warning(f"Error al insertar chunk {chunk['chunk_id']}")

        logger.info(f"\nProcesamiento completado:\n- Total chunks: {len(chunks)}\n- Exitosos: {successful_inserts}\n- Fallidos: {failed_inserts}")


def main():
    """Función principal para ejecutar el proceso de vectorización."""
    logger.info("Iniciando proceso de ingesta de datos de Excel...")
    
    file_to_process = "data/BD_Contenedores_Completo_2025.xlsx"
    if not os.path.exists(file_to_process):
        logger.error(f"Archivo no encontrado: {file_to_process}")
        return

    try:
        vectorizer = ExcelVectorizer(chunk_size=10, tokens_per_chunk=500)
        vectorizer.process_file(file_to_process)
        logger.info("¡Proceso de vectorización completado!")
    except ValueError as e:
        logger.error(f"Error de configuración: {e}")
    except Exception as e:
        logger.error(f"Ha ocurrido un error inesperado: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario.")
