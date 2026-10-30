# Predicción del precio de casas en California (servicio web)

Servicio web hecho con FastAPI que usa un árbol de decisión (scikit-learn) para estimar
el precio medio de las casas de una zona de California (`MedHouseVal`, en cientos de miles de dólares).

Autores: <Daniel Gracia>, <David Lopez>

## Contenido del repositorio
- `model_training.py`: entrena el modelo (Práctica 1) y genera `modelo_california.pkl`.
- `modelo_california.pkl`: el modelo ya entrenado.
- `app.py`: la API, con `GET /` (health-check) y `POST /predict` (predicción).
- `requirements.txt`: las dependencias mínimas para producción.
- `medir_tiempos.py`: mide el tiempo de respuesta del servicio desplegado.

## Pasos del despliegue
1. Creamos una cuenta gratuita en <https://render.com> iniciando sesión con GitHub.
2. Subimos el proyecto a un repositorio público de GitHub (`<usuario>/<repositorio>`).
3. En Render: **New > Web Service**, conectamos el repositorio y elegimos el plan **Free**.
4. Configuración: Language `Python 3`, Branch `main`, Build Command `pip install -r requirements.txt`.
5. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
6. Pulsamos **Deploy Web Service** y esperamos a que el log muestre "Your service is live".
7. Si el `.pkl` no está en local al arrancar, `app.py` lo descarga desde GitHub (`MODEL_URL`).

## URL final
<https://dawm2dopia.onrender.com>
