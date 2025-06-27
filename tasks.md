# Tareas y Mejoras Pendientes del Proyecto RAGPV

Este archivo documenta las futuras funcionalidades y mejoras planificadas para el agente.

## 1. Implementar Generación de Gráficos

- **Objetivo:** Permitir que el agente responda a preguntas analíticas con gráficos (imágenes PNG) en lugar de texto a través de WhatsApp.
- **Ejemplo de Consulta:** "Muéstrame un gráfico con la evolución de clientes en calidad de entregado en el último mes."

### Plan de Implementación Propuesto

#### Backend (FastAPI)
1.  **Detección de Intención:** Mejorar el endpoint `/api/query` para que pueda diferenciar entre una pregunta de texto (RAG) y una solicitud de gráfico (Analítica).
2.  **Tabla de Datos Estructurada:** Crear una nueva tabla en Supabase (ej. `viajes`) que almacene los datos del Excel con tipos de datos correctos (`TIMESTAMP` para fechas, `VARCHAR` para texto, `INTEGER` para números).
3.  **Lógica de Consulta SQL:** Implementar una función que, ante una solicitud de gráfico, ejecute una consulta SQL analítica sobre la nueva tabla estructurada (ej. `GROUP BY`, `COUNT`, `WHERE fecha_viaje > ...`).
4.  **Generación de Imagen:** Usar una librería como `Matplotlib` o `Seaborn` para crear un gráfico (`.png`) a partir de los resultados de la consulta SQL.
5.  **Almacenamiento de Imagen:** Subir la imagen generada al Storage de Supabase para obtener una URL pública y de acceso temporal.
6.  **Respuesta de API Modificada:** El endpoint deberá devolver un JSON que indique el tipo de respuesta (texto o imagen) y el contenido (el texto o la URL de la imagen).

#### Frontend (n8n)
1.  **Lógica Condicional:** Añadir un nodo `IF` en el flujo de n8n que verifique el tipo de respuesta recibido desde la API.
2.  **Envío de Imagen:** Si el tipo es "imagen", el flujo deberá usar el nodo "Send Media" o "Send Image" de WhatsApp, pasándole la URL del gráfico.
3.  **Envío de Texto:** Si el tipo es "texto", continuará usando el nodo "Send Text" actual.

## 2. Implementar Ingesta de Datos Incremental

- **Objetivo:** Optimizar el proceso de actualización de datos para que solo se procesen y vectoricen las filas nuevas o modificadas del archivo Excel, evitando el reprocesamiento completo.
- **Método Propuesto:** Utilizar un hash de fila (ej. SHA-256) como identificador único para detectar cambios.

### Plan de Implementación Propuesto

1.  **Base de Datos (Supabase):**
    -   Añadir una nueva columna `row_hash` de tipo `VARCHAR(64)` a la tabla `documentos_embeddings`.
    -   Añadir una restricción `UNIQUE` a la columna `row_hash` para garantizar la integridad y evitar duplicados.

2.  **Script de Ingesta (`excel_vectorizer.py`):**
    -   Al leer cada fila del Excel, concatenar todos sus valores y calcular su hash SHA-256.
    -   Antes de procesar la fila, ejecutar una consulta rápida a Supabase (`SELECT id FROM documentos_embeddings WHERE row_hash = :hash`).
    -   **Si el hash ya existe:** Ignorar la fila y continuar con la siguiente.
    -   **Si el hash no existe:** Proceder con la creación del fragmento, la vectorización y la inserción en la base de datos, guardando también el `row_hash` calculado. 