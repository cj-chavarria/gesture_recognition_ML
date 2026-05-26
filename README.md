<div align="center">

# Reconocimiento de Gestos con Deep Learning

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.14-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.35-00BCD4?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge)](https://github.com/astral-sh/uv)

*Sistema de reconocimiento de gestos en tiempo real que combina estimación de pose de mano con MediaPipe y una Red Neuronal Convolucional 1D para clasificación de secuencias temporales.*

</div>

---

## Descripción general

Este proyecto implementa un pipeline completo para reconocer gestos de mano desde una cámara web en vivo. Captura secuencias de landmarks de mano normalizados, codifica la dinámica del movimiento mediante ingeniería de características, y clasifica los gestos usando una CNN ligera entrenada sobre secuencias temporales — todo en tiempo real.

El sistema se estructura en tres etapas secuenciales:

```
┌──────────────────┐     ┌────────────────────────┐     ┌──────────────────────┐
│  1. Captura de   │────▶│  2. Entrenamiento      │────▶│  3. Predicción en    │
│     datos        │     │  (CNN 1D, 50 épocas)   │     │     tiempo real      │
└──────────────────┘     └────────────────────────┘     └──────────────────────┘
```

---

## Arquitectura de IA

### Etapa 1 — Estimación de pose de mano (MediaPipe)

El modelo **MediaPipe Hand Landmarker** de Google detecta **21 puntos clave** de la mano en cada fotograma, produciendo coordenadas 3D normalizadas `(x, y, z)` para cada articulación desde la muñeca hasta las puntas de los dedos. Es un modelo pre-entrenado que corre eficientemente en CPU.

Los landmarks crudos se procesan con ingeniería de características personalizada:

- **Normalización relativa a la muñeca** — todos los landmarks se restan por la posición de la muñeca, haciendo la representación invariante a traslación.
- **Normalización por tamaño de mano** — las coordenadas se dividen por la distancia muñeca-MCP del dedo medio, eliminando la dependencia de escala.
- **Características de velocidad** — se añaden los deltas entre fotogramas consecutivos `(Δx, Δy)`, codificando explícitamente la dinámica del movimiento.

Cada fotograma queda representado como un **vector de 44 dimensiones** (20 landmarks × 2 coordenadas + 4 componentes de velocidad).

```
Frame t   ──▶  [21 landmarks] ──▶ normalizar ──▶ [40 coords] + [4 Δ velocidad] = 44 features
Frame t+1 ──▶  [21 landmarks] ──▶ normalizar ──▶ [40 coords] + [4 Δ velocidad] = 44 features
    ⋮
Frame t+59 ──▶  ...
                                                  ─────────────────────────────────────────────
                                                          Forma de la secuencia: (60, 44)
```

### Etapa 2 — Clasificación de secuencias temporales (CNN 1D)

La secuencia de 60 fotogramas se pasa a una **Red Neuronal Convolucional 1D** que extrae patrones temporales locales a lo largo del movimiento del gesto.

```
Entrada: (60, 44)
       │
       ▼
 ┌──────────────────────────────┐
 │  Conv1D — 64 filtros, k=3    │  ← detección de patrones de movimiento locales
 │  MaxPooling1D (pool=2)       │
 └────────────┬─────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │  Conv1D — 128 filtros, k=3  │  ← características temporales de alto nivel
 │  MaxPooling1D (pool=2)      │
 └────────────┬────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │  Flatten                    │
 │  Dense (64, ReLU)           │
 │  Dropout (0.5)              │
 └────────────┬────────────────┘
              │
              ▼
 ┌─────────────────────────────┐
 │  Dense (5, Softmax)         │  ← probabilidades por clase de gesto
 └─────────────────────────────┘

Parámetros totales: 140,101  (~547 KB)
```

**¿Por qué CNN 1D en lugar de RNN/LSTM?**

Las CNNs 1D tienen un sesgo inductivo natural para detectar patrones temporales locales (por ejemplo, el cierre de un dedo que ocurre en ~5–10 fotogramas) y son significativamente más rápidas de entrenar e inferir que las alternativas recurrentes. Para datos de gestos ya representados como secuencias de longitud fija, entregan precisión competitiva con un costo de entrenamiento y latencia mucho menor — ideal para este tamaño de dataset y el requisito de inferencia en tiempo real.

### Etapa 3 — Inferencia en tiempo real

Durante la captura en vivo se mantiene un **deque deslizante de 60 fotogramas**. Una vez que el buffer está lleno, se genera una predicción con cada nuevo fotograma entrante usando la ventana actual de landmarks, proporcionando clasificación continua de gestos a la tasa de fotogramas de la cámara.

---

## Clases de gestos

| Etiqueta | Gesto | Descripción |
|----------|-------|-------------|
| `hello` | 👋 Saludo | Mano abierta saludando de lado a lado |
| `calling` | 🤙 Llámame | Pulgar y meñique extendidos |
| `pointing` | ☝️ Señalar | Dedo índice extendido hacia arriba |
| `so_so` | 🤚 Más o menos | Mano inclinada horizontalmente |
| `no_hand` | — | Ninguna mano detectada en el fotograma |

---

## Dataset

Las secuencias de gestos fueron recolectadas directamente desde una cámara web con **MediaPipe** procesando cada fotograma en tiempo real.

| Propiedad | Valor |
|-----------|-------|
| Secuencias por clase | 80 |
| Total de secuencias | 400 |
| Fotogramas por secuencia | 60 |
| Características por fotograma | 44 |
| Forma de la secuencia | `(60, 44)` |
| División entrenamiento / prueba | 80% / 20% |
| Formato de almacenamiento | arreglos `.npy` |
| Tamaño total | ~9.5 MB |

El dataset está **balanceado** entre las 5 clases, recolectado bajo condiciones controladas con una sola cámara web.

---

## Estructura del proyecto

```
gesture_recognition_ML/
│
├── 1-create_dataset.py     # Captura por cámara web → secuencias .npy
├── 2-training.ipynb        # Definición del modelo, entrenamiento y evaluación
├── 3-live_predict.py       # Inferencia en tiempo real desde cámara web
│
├── utils.py                # Configuración compartida (N_FRAMES, GESTURES) + extracción de features
├── main.py                 # Punto de entrada
│
├── dataset/                # Secuencias de gestos (400 × archivos .npy)
│   ├── hello/
│   ├── calling/
│   ├── pointing/
│   ├── so_so/
│   └── no_hand/
│
├── models/
│   ├── trained_model.keras      # Clasificador CNN 1D entrenado (1.7 MB)
│   └── hand_landmarker.task     # Modelo de landmarks de MediaPipe (7.5 MB)
```

---

## Dependencias

Gestionado con [uv](https://github.com/astral-sh/uv). Requiere **Python 3.12**.

| Paquete | Versión | Rol |
|---------|---------|-----|
| `mediapipe` | 0.10.35 | Detección de landmarks de mano |
| `opencv-python` | 4.13.0.92 | Captura de cámara y procesamiento de fotogramas |
| `tensorflow` | 2.21.0 | Framework de deep learning |
| `keras` | 3.14.1 | API de alto nivel para modelos |
| `numpy` | 2.4.5 | Operaciones numéricas sobre arreglos |
| `scikit-learn` | 1.8.0 | División de datos y métricas de evaluación |
| `matplotlib` | 3.10.9 | Curvas de entrenamiento y matriz de confusión |
| `jupyter` | 1.1.1 | Notebook interactivo de entrenamiento |

> **Nota:** TensorFlow no soporta aceleración GPU en Windows nativo. La inferencia por CPU es suficiente para predicción en tiempo real a 30 FPS dado el tamaño reducido del modelo (~547 KB).

---

## Configuración de entrenamiento

| Hiperparámetro | Valor |
|----------------|-------|
| Épocas | 50 |
| Tamaño de batch | 16 |
| Optimizador | Adam |
| Función de pérdida | Sparse categorical crossentropy |
| Dropout | 0.5 |
| División de validación | 20% de los datos de entrenamiento |

---

## Flujo de trabajo

```
1. Recolectar datos — 1-create_dataset.py
   └─ Realizar cada gesto frente a la cámara web
   └─ Se capturan 80 secuencias × 60 fotogramas por clase
   └─ Guarda en → dataset/<gesto>/<idx>.npy

2. Entrenar — 2-training.ipynb
   └─ Carga todas las secuencias, aplica división 80/20
   └─ Entrena la CNN 1D durante 50 épocas
   └─ Genera curvas de accuracy/pérdida + matriz de confusión
   └─ Guarda en → models/trained_model.keras

3. Predecir — 3-live_predict.py
   └─ Carga el modelo entrenado
   └─ Captura video de la cámara en tiempo real
   └─ Muestra el gesto predicho y su confianza en pantalla
```

---

<div align="center">

*Desarrollado como parte del currículo del 10° semestre — Ingeniería Física*

</div>
