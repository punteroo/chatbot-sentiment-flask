# Chatbot de Análisis de Reseñas

Este proyecto implementa un chatbot basado en Flask para clasificar reseñas en español según su sentimiento: "Excelente" (positivo), "Regular" (neutral) o "Pobre" (negativo). Utiliza un modelo de aprendizaje automático entrenado con un conjunto de datos de reseñas en español para predecir el sentimiento de una reseña ingresada por el usuario a través de una interfaz web.

Este producto fue realizado como un trabajo práctico para la cátedra de **Inteligencia Artificial 2025** en la UTN FRVM.

El dataset fue generado con **Google Gemini 2.5 Pro** con múltiples indicaciones de variabilidad para poder lograr una variación grande entre distintos tipos de reseñas, puede contener material ofensivo.

## Integrantes
- MAZA BIANCHI, Lucas
- MAIRONE, Nicolás Nahuel
- PEDRAZA RUBIANO, Santiago Manuel
- MARTINI, Leopoldo
- MOSCA, Sebastián Jesús
- RIVERA, María Paula

## Docente
- Ing. FUENTE, Claudio

## Requisitos

- Python 3.11 o superior
- Bibliotecas de Python:
  - `flask`
  - `scikit-learn`
  - `nltk`
  - `pandas`
  - `numpy`

## Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   ```

2. **Crear un entorno virtual** (opcional, pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install flask scikit-learn nltk pandas numpy
   ```

4. **Agregar el conjunto de datos**:
   - Coloque el archivo `dataset_sentiment_analisys.csv` en la carpeta `model/dataset/`.
   - Asegúrese de que el archivo tenga las columnas `review` y `sentimiento` con valores "positivo", "neutral" o "negativo".

## Despliegue

1. **Entrenar el modelo**:
   Desde el directorio raíz del proyecto, ejecute el script de entrenamiento para generar los archivos `sentiment_model.pkl` y `vectorizer.pkl`:
   ```bash
   python model/train.py
   ```
   Esto preprocesa el conjunto de datos, entrena un modelo y guarda los archivos del modelo en la carpeta `model/`. Revise la salida en la consola para verificar la precisión del modelo (idealmente >65%).

2. **Correr el servidor Flask**:
   Inicie la aplicación Flask con el siguiente comando:
   ```bash
   flask --app app.py run
   ```
   Esto inicia un servidor local en `http://127.0.0.1:5000`. Abra esta URL en un navegador web.

## Uso

1. **Interfaz web**:
   - Acceda a `http://127.0.0.1:5000` en su navegador.
   - Ingrese una reseña en español en la caja de texto proporcionada (por ejemplo, "El sistema es muy eficiente" o "El software tiene muchos errores").
   - Haga clic en el botón "Analizar Reseña".

2. **Salida**:
   - La aplicación clasificará la reseña y mostrará el sentimiento como:
     - **Sentimiento: Excelente** (fondo verde, para reseñas positivas).
     - **Sentimiento: Regular** (fondo amarillo, para reseñas neutrales).
     - **Sentimiento: Pobre** (fondo rojo, para reseñas negativas).
   - La reseña ingresada se mostrará debajo del resultado para referencia.
   - Puede ingresar una nueva reseña en la caja de texto sin necesidad de recargar la página.

## Ejemplo

- **Entrada**: "El sistema es muy rápido y fácil de usar"
- **Salida**: 
  - Sentimiento: Excelente (en fondo verde)
  - Reseña ingresada: El sistema es muy rápido y fácil de usar

- **Entrada**: "El software es lento y se cuelga a menudo"
- **Salida**: 
  - Sentimiento: Pobre (en fondo rojo)
  - Reseña ingresada: El software es lento y se cuelga a menudo