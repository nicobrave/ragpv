# RAGPV - Agente de Análisis de Datos Logísticos

Este proyecto es un agente de IA basado en el patrón RAG (Retrieval-Augmented Generation) diseñado para responder preguntas sobre datos operativos de una empresa de logística, utilizando un archivo Excel como fuente de conocimiento.

## Tech Stack

- **Backend:** FastAPI
- **Base de Datos Vectorial:** Supabase (con la extensión `pgvector`)
- **Modelos de IA:** Google Gemini (para embeddings y generación de texto)
- **Despliegue:** Render

## Configuración del Entorno Local

Para ejecutar el proyecto localmente, es necesario crear un archivo `.env` en la raíz del proyecto con las siguientes variables:

```
GOOGLE_API_KEY="tu_api_key_de_google"
SUPABASE_URL="la_url_de_tu_proyecto_supabase"
SUPABASE_KEY="la_anon_key_de_tu_proyecto_supabase"
```

## Ejecución Local

1.  Asegúrate de tener Python 3.11 y `pip` instalados.
2.  Crea y activa un entorno virtual:
    ```bash
    python3.11 -m venv venv
    source venv/bin/activate
    ```
3.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
4.  Ejecuta el servidor de desarrollo:
    ```bash
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    El servidor estará disponible en `http://localhost:8000`.

## Ingesta de Datos

Para poblar la base de datos vectorial, ejecuta el script `excel_vectorizer.py`:

```bash
python src/data_processing/excel_vectorizer.py
```
Asegúrate de que el archivo `data/BD_Contenedores_Completo_2025.xlsx` exista.

## Uso de la API

El endpoint principal para realizar consultas es `/api/query`.

### Ejemplo de Consulta con `curl`

```bash
curl -X 'POST' \
  'https://ragpv-api.onrender.com/api/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "¿Cual es el estado del contenedor TCNU5754568?"
  }'
```

La respuesta será un JSON con el texto generado por la IA.
