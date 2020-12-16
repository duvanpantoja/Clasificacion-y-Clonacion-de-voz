# Clasificacion-y-Clonacion-de-voz
En este repositorio proporcionamos la información esencial como datasets y códigos utilizados para realizar clasificación y clonación de la voz.
Clasificacion
Se planteo realizar el proceso de clasificacion entre varias personas utilizando las imagenes del espectro en tiempo frecuencia de su voz, estas imagenes se obtuviron para 6 transformadas, STFT: Hamming, Blackman, Kaiser y CWT: Bump, Morse, Amor. Este proceso se realizo en el entorno de programacion MATLAB, los parametros y codigo en general se encuentran en el archivo llamado 'Transformadas_Audios_clasificacion.m' con el cual se obtiene la imagen de cada transformada.
¡[](images\Transformadas.jpg)
Para el proceso de clasificacion proporcionamos una tecnica llamada Trasnfer-Learning que consiste en tomar modelos pre-entrenados y adaptarlos a los problemas de clasificacion especificos que necesites, puedes acceder a estos modelos proporcionados por Keras en Python.
Las imgenes de clasificacion utilizadas tanto para el entrenamiento y prubea 


