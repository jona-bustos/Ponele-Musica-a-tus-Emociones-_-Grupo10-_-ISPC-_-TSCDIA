<h1 align="center">
  GRUPO 10 - Ponele Música a Tus Emociones - ISPC
</h1>

<p align="center">
  <img src="https://github.com/ClaudiaMetz/ISPC---PP1---Grupo-10/blob/main/CD.png" width="100%" title="Intro Card" alt="Intro Card">
</p>
<h2 align="center">Integrantes: 👩🏾‍💻 </h2>
<h5 align="center">Hilgemberg Maria Sol<br>
<b>Soria Julio Ezequiel</b><br>
<b>Bustos Jonathan</b><br>
<b>Metz Claudia</b><br>
<b>Quiroga Horacio Eduardo</b><br>
<b>Meier Ivan Didier</b><br>
<b>Muñoz Mariel</b><br></h5>

---




# Ponele Música a Tus Emociones-Grupo10-ISPC-TSCDIA

---

<b>Idea Principal:</b> Detectar la Emoción en una foto y Recomendar una Canción
<br>
<p><b>Este trabajo combina técnicas de Procesamiento de Imágenes y Machine Learning para crear una experiencia única: El usuario suministra una fotografía por medio de la computadora, luego la aplicación web analiza la emoción predominante en la foto utilizando un modelo de Aprendizaje Profundo. Luego se selecciona una canción de una base de datos o conjunto de temas musicales y la expresa con un audio. <br>
Para su desarrollo utilizamos técnicas de Minería de Datos y Aprendizaje Automático. <br>
Se utilizó el dataset FER2013 [Dataset 🌟](https://drive.google.com/file/d/1vtWxb5LioAATFb5Qypqa0zavj4NPoj06/view?usp=drive_link). (Facial Expression Recognition 2013) contiene imágenes junto con categorías que describen la emoción de la persona en ellas. <br>

Este proyecto busca aprovechar el deep learning y el procesamiento de imágenes para las personas puedan revivir momentos especiales o simplemente para explorar nuevas sensaciones. Y en cada nota, en cada imagen, encontrarán un eco de su propia historia.
<b><br></p>
<p align="center">
  <img src="[https://github.com/ClaudiaMetz/ISPC---PP1---Grupo-10/blob/main/CD.png](https://drive.google.com/file/d/1GFpOeiYm3X5lxhz8wbGrPh-VdwFfp-mK/view?usp=drive_link)" width="100%" title="Historia" alt="Historia">
</p>

---

# Objetivo General
Desarrollar un sistema automatizado basado en deep learning para detectar emociones mediante fotografías que el usuario suministre, con el fin de obtener una canción que refleje el resultado obtenido.

# Objetivos Específicos
Recolección y Preparación de Datos
o Obtener un Dataset, que se adapte al campo del reconocimiento de expresiones faciales.
FER2013 [Dataset 🌟](https://drive.google.com/file/d/1vtWxb5LioAATFb5Qypqa0zavj4NPoj06/view?usp=drive_link)

o Corroborar una muestra del conjunto de imágenes con las etiquetas para crear un conjunto de datos de entrenamiento consistente.

Desarrollo del Modelo de Deep Learning

o Diseñar y entrenar un modelo de aprendizaje profundo que pueda identificar emociones, basándose en características visuales de los rostros.

o Evaluar el rendimiento del modelo por medio de metrícas adecuadas y realizar ajustes necesarios para optimizar su desempeño.

Implementación del Sistema de Detección

o Implementar una página web donde los usuarios puedan cargas sus fotos y la aplicación detecte la emoción y se obtenga la canción.

Validación y Verificación
o Realizar pruebas y validación para asegurar la precisión y robustez del sistema.

o Ajustar el modelo y el sistema en base a los resultados obtenidos para mejorar su confiabilidad y precisión.

Metodología
Análisis de Requisitos:
o Definir claramente los requisitos técnicos y funcionales del sistema.

o Establecer los criterios de éxito y las métricas de evaluación.

Desarrollo Iterativo:
o Seguir un enfoque ágil para el desarrollo del sistema, permitiendo iteraciones rápidas y ajustes continuos basados en retroalimentación.

Colaboración y Feedback:
o Mantener una comunicación constante con los usuarios de la aplicación para asegurar que el sistema desarrollado satisfaga sus necesidades.

Monitoreo y Mejora Continua:
o Implementar un sistema de monitoreo para evaluar continuamente el rendimiento del sistema en producción.

o Planificar actualizaciones y mejoras basadas en nuevos datos y nuevas tecnologías.

<p align="center">
  <img src="[https://github.com/ClaudiaMetz/ISPC---PP1---Grupo-10/blob/main/CD.png](https://drive.google.com/file/d/1GFpOeiYm3X5lxhz8wbGrPh-VdwFfp-mK/view?usp=drive_link)" width="100%" title="CRISP" alt="CRISP">
</p>
---

Modelo de Referencia CRISP-DM
1. Comprensión del Negocio
Objetivos y requisitos del Proyecto, definición del problema y plan preliminar diseñado.

📌 Desarrollar un sistema automatizado que use:

 Fotografías que el usuario cargue

 Algoritmos de aprendizaje profundo, entrenados para reconocer rasgos faciales y detectar emociones.

📌 Objetivo del sistema automatizado

El objetivo es que la aplicación pueda identificar emociones a través de los rostros de imágenes cargadas. Éste sistema podría ayudar a detectar cambios de ánimo y emociones que presenta una imágen o persona.

📌 Definición del Problema y Plan Preliminar Diseñado

El problema a resolver es el reconocimiento facial automatizado en imágenes para luego buscar una frase que permita obtener una canción relacionada con la emoción detectada. Para abordar este problema, se propone un procedimiento completo de aprendizaje profundo que incluye:

Lectura de datos

Se obtendrán imágenes de un dataset previamente descargado, que serán utilizadas como base de datos principal.
Utilizaremos el dataset FER2013, un recurso muy utilizado en el campo del reconocimiento de expresiones faciales y que puede proporcionarnos una buena variedad de datos para entrenar un modelo, asegurando que la información sea correctamente procesada y preparada para el entrenamiento del modelo.

Entrenamiento de modelos

Se entrenan modelos de aprendizaje profundo, específicamente redes neuronales convolucionales(CNN), para reconocer las características distintivas de los rostros de las personas en las imágenes.

El entrenamiento del modelo 

Realización de predicciones

Una vez que el modelo esté entrenado , se ingresarán nuevas imágenes de rostros para identificar las emociones.
Las predicciones serán validadas y ajustadas para mejorar la precisión del sistema. Esto incluye la evaluación del rendimiento del modelo utilizando métricas como la precisión , el recall y el F1-score, y realizando ajustes basados en los resultados de estas evaluaciones.


📌 Desarrollo del Sistema Automatizado

Fotografías provistas por el usuario
El sistema utiliza imágenes con rostros para identificar las emociones. La calidad y resolución de estas imágenes son esenciales para la precisión modelo. 

Algoritmo de Aprendizaje Profundo
Se entrenan algoritmos de aprendizaje profundo, específicamente diseñados para recorrer formas en las imágenes. El entrenamiento incluirá técnicas de argumentación de datos para mejorar la robustez del modelo......

📌 Identificación de las partes interesadas y sus necesidades

Partes interesadas:

Equipo de desarrollo del proyecto: Incluye a Claudia Metz y Jonathan Bustos, quienes están a cargo de desarrollar y entrenar el modelo de deep learning.

Usuarios finales: Cualquier persona o empresa que estén interesadas en contar con una aplicación que convine IA y Música.

Necesidades:

Automatización y precisión: Desarrollar un sistema automatizado que utilice imágenes de rostros y algoritmos de aprendizaje profundo para identificar las emociones.

Visualización y evaluación: Capacidad para visualizar los resultados de la segmentación semántica y evaluar la precisión del modelo.

2. Comprensión de los Datos
Recopilación , primeros conocimientos de los datos
El primer paso en la fase de comprensión de los datos del modelo CRISP-DM implica la identificación y adquisición de las fuentes de datos relevantes. Este proceso es fundamental para asegurar que se cuenta con la información adecuada para abordar el problema planteado.

1. Identificación de Fuentes de Datos:
Dataset: Para este proyecto, las imágenes fueron obtenidad desde Keagle.com, del dataset FER2013 (Facial Expression Recognition 2013).
Fotografías: Fotografías que el usuario cargue en la aplicación.

2. Análisis Exploratorios de las Imágenes:
Visualización de Imágenes: Se realiza una visualización inicial de las imágenes para comprender mejor su contenido y características visuales.
Resolución y Metadatos: Es esencial entender la resolución espacial de las imágenes, que determina el nivel de detalle que se puede observar. Además, se revisan los metadatos para obtener información adicional.

Instalación de Librerías y Configuración del Entorno
A continuación se observa las importaciones de bibliotecas de Python que son útiles para procesamiento de imágenes, visualización de datos, y construcción de modelos de aprendizaje automático y profundo:
NumPy (import numpy as np): Biblioteca fundamental para computación científica en Python. Proporciona soporte para grandes matrices multidimensionales y una colección de funciones matemáticas de alto nivel.
SciPy (from scipy import misc): Biblioteca que complementa a NumPy y proporciona funciones adicionales para computación científica. Aquí se importa el submódulo misc, que contiene funciones de utilidad misceláneas.
PIL (Python Imaging Library) (from PIL import Image): Biblioteca para la manipulación de imágenes.
glob (import glob): Biblioteca para encontrar todos los nombres de ruta que coinciden con un patrón especificado, útil para manejar múltiples archivos.
Matplotlib (import matplotlib.pyplot as plt y from matplotlib.pyplot import imshow): Biblioteca para crear gráficos y visualizaciones.
IPython.display (from IPython.display import SVG): Biblioteca para mostrar contenido interactivo en Jupyter Notebooks.
OpenCV (import cv2): Biblioteca de visión por computadora para procesar imágenes y videos.
Seaborn (import seaborn as sn): Biblioteca para visualización de datos basada en Matplotlib, proporciona una interfaz de alto nivel para dibujar gráficos estadísticos atractivos y informativos.
Pandas (import pandas as pd): Biblioteca para manipulación y análisis de datos, proporciona estructuras de datos flexibles y eficientes.
pickle (import pickle): Biblioteca para serializar y deserializar estructuras de objetos de Python.
Keras (import keras y varios submódulos): Biblioteca de alto nivel para crear y entrenar modelos de aprendizaje profundo. Las importaciones incluyen capas, modelos, optimizadores, funciones de pérdida, y utilidades para trabajar con datos de imágenes.
TensorFlow (import tensorflow as tf y submódulos de Keras dentro de TensorFlow): Plataforma de aprendizaje automático de extremo a extremo. Aquí se importa para usar junto con Keras, ya que Keras es una API de alto nivel que puede funcionar sobre TensorFlow.
Scikit-learn (from sklearn.metrics import confusion_matrix, classification_report): Biblioteca para aprendizaje automático en Python. Aquí se importa para evaluar modelos con matrices de confusión y reportes de clasificación.
ImageDataGenerator (from tensorflow.keras.preprocessing.image import ImageDataGenerator): Utilidad de Keras para generar nuevas imágenes mediante técnicas de aumento de datos.
VGG (from keras.applications import vgg16, vgg19): Modelos de redes neuronales convolucionales preentrenados, útiles para tareas de clasificación de imágenes y transferencia de aprendizaje.

Estas importaciones proporcionan un conjunto robusto de herramientas para el procesamiento y análisis de imágenes, construcción y entrenamiento de modelos de aprendizaje profundo, y evaluación del rendimiento de los modelos en Python.

Cargar en Google Drive: El dataset se cargó en Google Drive para permitir un fácil acceso y manipulación desde cualquier lugar.

Análisis exploratorio de los datos

Como se puede apreciar el dataset consiste de 3 columnas:
emotion: Contiene la etiqueta que define la emoción de la imágen y es nuestra variable objetivo. (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
pixels: Corresponden a los valores de cada uno de los pixeles. De acuerdo con las instrucciones de la competencia, las imágenes son de 48x48.
Usage: Corresponde al set de datos correspondientes: Train, PublicTest, Private Test.

Se puede ver que existen cerca de 28700 imágenes que son utilizadas para entrenamiento. Estas imágenes contienen una etiqueta por lo que podían ser utilizadas para entrenar.

Adicionalmente se pueden ver dos grupos, Public y Private Test que correspondían a los Sets de Validación. El set público era el que se usaba para evaluar los resultados al subirlos a la plataforma, mientras que el Private correspondía a los datos ocultos que se liberan sólo al finalizar la competencia para decidir a los ganadores.


## Características del Proyecto

Objetivo: Utilizar el procesamiento de imágenes y machine learning para detectar emociones en rostros de personas.
 
Calidad y Configuración de Datos
Dataset: FER2013 (Facial Expression Recognition 2013 - kaggle.com)
Preprocesamiento: Esta etapa incluyó la normalización de las imágenes, conversión de etiquetas a categóricas.

Calidad del Código
Estructura: El código está bien organizado, con comentarios explicativos y secciones claramente definidas.
Modelo de Deep Learning: Utilizamos un modelo DenseNet121, el cual introduce conexiones directas entre todas las capas. Su principal ventaja es la capacidad de determinar características y patrones más discriminativas gracias a su mayor flujo de información. Además, el modelo reduce el problema de desvanecimiento del gradiente. 
Sin embargo, la profundidad del modelo, con el número de capas que conlleva, y las conexiones entre ellas hace de este modelo muy pesado computacionalmente a la hora de su entrenamiento.

3. Preparación de los Datos
Seleccion de tablas, registros y atributos, transformación y limpieza de datos.

Preparar los datos para el modelado implica la limpieza de las imágenes, la evaluación del dataset , la eliminación de las partes no deseadas y la preparación de las etiquetas para el entrenamiento del modelo, afortunadamente en el dataset utilizado, los datos ya se encontraban tabulados casi de la forma que los necesitábamos


🧪 Entrenamiento del modelo 'resnet18'

Carga del Modelo Preentrenado:

Se configura el modelo para el preprocesamiento de datos (conversión de imágenes  de un canal a tres canales, convertir etiquetas a categóricas, deivir el dataset en entranamiento, validación y prueba).
Se aplica técnica de argumentación de datos para mejorar la robustez del modelo.
Se definen las entradas del modelo.
Se redimensionan las entradas que coincidan con el modelo preentrenado
Se carga un modelo DenseNet121 utilizando tensorflow.keras.applications
Se agregan capas personalizadas: 
    o	GlobalAveragePooling2D: Reduce las dimensiones espaciales del tensor (de (7, 7, ...) a (1, 1, ...)).
    o	Capas densas (Dense):
    	1024, 512 y 256 neuronas con activación ReLU.
    	BatchNormalization: Normaliza los valores para mejorar la estabilidad del entrenamiento.
    	Dropout(0.5): Reduce el sobreajuste al desactivar aleatoriamente el 50% de las neuronas.
    o	Capa final (Dense(7, activation="softmax")):
    	Produce una probabilidad para cada una de las 7 clases.

Entrenamiento del Modelo:

 Se inicia el entrenamiento del modelo con los conjuntos de datos preparados.
 Se realizan iteraciones (épocas) donde el modelo aprende a identificar características faciales y emociones.

🧪 Preprocesar y Transformar los datos para su uso en el modelo .................


4. Modelado
- [Ver modelo 🌟](FotoEmocion__Cancion.ipynb)
Selección y aplicación de varias técnicas de modelado

📝 Seleccionar el modelo de aprendizaje profundo (Densenet121)

Utilizamos un modelo DenseNet121, el cual introduce conexiones directas entre todas las capas. Su principal ventaja es la capacidad de determinar características y patrones más discriminativas gracias a su mayor flujo de información. Además, el modelo reduce el problema de desvanecimiento del gradiente. 
Sin embargo, la profundidad del modelo, con el número de capas que conlleva, y las conexiones entre ellas hace de este modelo muy pesado computacionalmente a la hora de su entrenamiento.


📝 Configurar el modelo y los parámetros de entrenamiento

Configuración de Parámetros del Modelo: Los parámetros del modelo, como el número de capas y neuronas, así como las funciones de activación, se configuran de acuerdo con las necesidades de la tarea.

Configuración de Parámetros de Entrenamiento: Los parámetros de entrenamiento, como el número de épocas, el tamaño del lote y la tasa de aprendizaje, se establecen. Estos parámetros pueden tener un gran impacto en la eficacia del entrenamiento.

📝 Entrenar el modelo con los datos de entrenamiento

El modelo se entrena utilizando los datos de entrenamiento, ajustando los pesos y los parámetros internos de la red para minimizar el error.
Se inicia el entrenamiento del modelo con los conjuntos de datos preparados.
Se realizan iteraciones (epoch) donde el modelo aprende a expresiones faciales en las imágenes.
Se configuran los callback y el data augmentation para:
   Optimizar el entrenamiento.
   Ahorrar tiempo y recursos.
   Mejorar la generalización.
   Evitar el sobreajuste.
 
En conjunto, los callbacks y el data augmentation sirven para mejorar la eficiencia y la calidad del entrenamiento de un modelo de aprendizaje profundo. Aquí está el propósito principal de cada uno:

Callbacks:
Optimizar el entrenamiento:

EarlyStopping previene el sobreajuste deteniendo el entrenamiento cuando el modelo deja de mejorar en los datos de validación.
ReduceLROnPlateau ajusta automáticamente el learning rate para permitir que el modelo refine su aprendizaje cuando el progreso se desacelera.
Ahorrar tiempo y recursos:

Detener el entrenamiento antes de completar todas las épocas evita desperdiciar tiempo en épocas improductivas.
Reducir el learning rate ayuda al modelo a converger más eficientemente hacia una solución óptima.
Data Augmentation:
Mejorar la generalización:

Genera datos nuevos a partir de los existentes aplicando transformaciones aleatorias (como rotaciones, desplazamientos o cambios de brillo).
Esto aumenta la diversidad del conjunto de entrenamiento sin necesidad de recolectar más datos.
Evitar el sobreajuste:

Al exponer al modelo a variaciones de los datos de entrenamiento, se hace menos probable que memorice ejemplos específicos y más probable que aprenda características generales.
Beneficio combinado:
Los callbacks garantizan un entrenamiento eficiente, ajustando dinámicamente el proceso según el desempeño del modelo.
El data augmentation mejora la calidad del aprendizaje, ayudando al modelo a ser más robusto y generalizable.
Ambos trabajan juntos para producir un modelo que no solo se entrena de manera eficiente, sino que también funciona bien con datos nuevos (no vistos).

5. Evaluación
Evaluación del modelo y revisión de los pasos ejecutados La evaluación del rendimiento del modelo permite comprender cómo se comporta el modelo en datos no vistos durante el entrenamiento .Este análisis se realiza utilizando el conjunto de datos de validación y se mide mediante diversas métricas de evaluación como la precisión, la exhaustividad (recall), el puntaje F1 y el AUC-ROC.

🔬 Propósitos de la Evaluación Evaluar el rendimiento del modelo con los datos de validación es crucial para varios propósitos:

Generalización: Evaluar cómo el modelo se comporta en datos no vistos durante el entrenamiento. Los datos de validación proporcionan una estimación realista del rendimiento en situaciones del mundo real.
Ajustes de Hiperparametros: Durante la validación, podemos ajustar los hiper parámetros del modelo (como la tasa de aprendizaje, el tamaño del lote, etc) para obtener un mejor rendimiento.
Selección de Modelos: Comparamos diferentes modelos o arquitecturas utilizando los datos de validación para elegir el mejor.
Evitar el sobreajuste: Si el modelo tiene un rendimiento excelente en los datos de entrenamiento, pero no en los de validación, podría ser sobre ajustado. La validación ayuda a detectar esto.
En resumen, la evaluación con datos de validación nos permite comprender como nuestro modelo se desempeñará en el mundo real y tomar decisiones informadas para mejorarlo.

-- Resultados de Evaluación del Modelo
Después de ejecutar el código para evaluar el modelo, obtuvimos los siguientes resultados:

Test Loss (Pérdida en el conjunto de prueba): 0.920417308807373
Test Accuracy (Precisión en el conjunto de prueba): 0.6625185608863831

1. Test Loss (Pérdida en el conjunto de prueba)

Valor: 0.920417308807373
La "pérdida" es una métrica que mide el error del modelo en el conjunto de datos de prueba. Un valor más bajo de pérdida indica que el modelo está haciendo predicciones más precisas. En este caso, una pérdida de aproximadamente 0.92 indica que el modelo tiene un error moderado al predecir las emociones en las imágenes del conjunto de prueba. Sin embargo, este valor por sí solo no proporciona una imagen completa del rendimiento del modelo, ya que depende del contexto y del valor de la función de pérdida utilizada durante el entrenamiento.
2. Test Accuracy (Precisión en el conjunto de prueba)

Valor: 0.6625185608863831
La precisión es la proporción de predicciones correctas realizadas por el modelo en el conjunto de datos de prueba. En este caso, una precisión de aproximadamente 66.25% significa que el modelo predice correctamente las emociones en el 66.25% de las imágenes de prueba.

--- INGRESAR GRAFICO

Los gráficos muestran la evolución de la precisión (accuracy) y la pérdida (loss) durante el entrenamiento y la validación de un modelo de aprendizaje automático a lo largo de varias épocas.

Gráfico 1: Training Accuracy vs Validation Accuracy

• Eje Y: Precisión (Accuracy). 
• Eje X: Número de épocas (Num of Epochs). 
• Línea Roja: Precisión en el conjunto de entrenamiento (Train Accuracy). 
• Línea Verde: Precisión en el conjunto de validación (Validation Accuracy).

Observaciones:
Precisión del entrenamiento (línea roja): Aumenta constantemente a medida que el modelo se entrena, alcanzando cerca del 90% hacia la última época.
Precisión de la validación (línea verde): Aumenta inicialmente pero se estabiliza y fluctúa alrededor del 70% después de unas pocas épocas.

Interpretación:
• El modelo está aprendiendo bien en el conjunto de entrenamiento, lo que se refleja en el aumento constante de la precisión de entrenamiento. 
• La precisión de validación se estabiliza y no mejora mucho después de unas pocas épocas, lo cual sugiere que el modelo puede estar sobreajustándose (overfitting) al conjunto de entrenamiento, ya que la brecha entre la precisión de entrenamiento y de validación se ensancha con el tiempo.

Gráfico 2: Training Loss vs Validation Loss

• Eje Y: Pérdida (Loss). 
• Eje X: Número de épocas (Num of Epochs). 
• Línea Roja: Pérdida en el conjunto de entrenamiento (Train Loss). 
• Línea Verde: Pérdida en el conjunto de validación (Validation Loss).

Observaciones:

Pérdida del entrenamiento (línea roja): Disminuye constantemente a medida que el modelo se entrena.
Pérdida de la validación (línea verde): Disminuye inicialmente pero luego se estabiliza e incluso muestra un ligero aumento hacia el final.

Interpretación:

• La disminución continua de la pérdida de entrenamiento indica que el modelo está mejorando en su capacidad de predecir los datos de entrenamiento. 
• La pérdida de validación disminuye al principio, lo que sugiere que el modelo mejora, pero luego se estabiliza e incluso aumenta ligeramente, lo cual es una señal de sobreajuste. Esto significa que el modelo está memorizando los datos de entrenamiento en lugar de generalizar bien a datos nuevos.


Conclusión General
Los resultados y las observaciones de los gráficos sugieren que el modelo está aprendiendo eficazmente del conjunto de entrenamiento, como lo indican la alta precisión de entrenamiento y la disminución constante de la pérdida de entrenamiento. Sin embargo, la precisión de validación que se estabiliza y la pérdida de validación que aumenta ligeramente indican que el modelo no generaliza bien a datos nuevos, un signo claro de sobreajuste.

Recomendaciones para mitigar el sobreajuste:
Regularización: Aplicar técnicas de regularización como L2 o dropout.
Aumento de datos (Data Augmentation): Incrementar la variabilidad del conjunto de datos de entrenamiento mediante técnicas de aumento de datos.
Uso de un conjunto de validación más grande: Aumentar el tamaño del conjunto de validación para obtener una mejor estimación de la capacidad de generalización del modelo.
Early Stopping: Implementar early stopping para detener el entrenamiento cuando la pérdida de validación ya no mejora.
Estas estrategias pueden ayudar a mejorar la capacidad del modelo para generalizar a datos nuevos y, por ende, mejorar el rendimiento en el conjunto de test.



LISTADO DE ALGORITMOS, FRAMEWORKS Y HERRAMIENTAS PRE ENTRENADAS UTILIZADOS EN EL PROYECTO
El proyecto de detección de piletas mediante imágenes satelitales utiliza varios algoritmos y técnicas para lograr su objetivo. A continuación, se presentan algunos de los más relevantes:
Redes Neuronales Convolucionales (CNN):

Estas redes son fundamentales para el aprendizaje profundo en visión por computadora. Se entrenaron para reconocer patrones y características en imágenes, como formas y texturas. En este proyecto, se utilizan para identificar piscinas en las imágenes satelitales.


Para mejorar la robustez del modelo, se aplicaron técnicas de argumentación de datos, como ......... la rotación, el cambio de brillo y la ampliación/reducción de imágenes.
---


