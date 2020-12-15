from google.colab import drive                #Realiza la conexion con Google Drive para poder acceder a las rutas de las carpetas que contienen el Dataset
drive.mount('/content/drive')

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import *
from mlxtend.evaluate import confusion_matrix
import matplotlib.image as mpimg
import pandas as pd
import sys 
import os


#Ruta datos de entrada
train_data = '/content/drive/My Drive/Dataset Nariñense/Morse/train'     #Ruta donde se encuentran los datos de entrenamiento, la carpeta Train contiene tres subcarpetas que 
                                                                                #seran las etiquetas de cada clase y en cada una de ella sus espectros correspondientes 

test_data = '/content/drive/My Drive/Dataset Nariñense/Morse/prueba'     #Ruta donde se encuetrar los datos de Prueba, en esta carpeta se encuentran espectros de cada etiqueta
                                                                                #de manera aleatoria, se ingresan estos datos para observar si la red es capaz de clasificarlos

#PARAMETROS 
epocas = 20                     #Cantidad de iteraciones que dará el algoritmo de entrenamiento
altura, longitud = 256, 256     #Altura y longitud a la cual serán redimensionadas las imágenes 
batch_size = 32                #Numero de imagenes que procesa en cada uno de los pasos
pasos = 1000
filtrosConv1 = 32               #Numero de filtros que se aplican tras la primera convolucion 
filtrosConv2 = 64               #Numero de filtros que se aplican tras la segunda convolucion
tam_filtro1 = (3,3)             #Tamaño para la primera convolucion
tam_filtro2 = (2,2)             #Tamaño para la segunda convolucion
tam_pool = (2,2)                #Para mejorar el avance de la convolucion 
clases = 20                      #Numero de clases o etiquetas que la red va a clasificar, en este caso [Desconocidos,Duvan,Oscar] 
lr = 0.0005                     #Tasa de aprendizaje, se recomienda valores pequeños

#PRE-PROCESAMIENTO DE IMAGENES
entrenamiento_datagen = ImageDataGenerator(rescale = 1./255)           #Se normaliza las imagenes 
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    train_data,                                                        #Datos de entrenamiento 
    target_size = (altura, longitud),                                  #Redimensiona a los tamaños establecidos
    batch_size = batch_size,                                           #Batch size anteriormente establecido
    class_mode = 'categorical',
    color_mode="grayscale")                                        #Categorical porque vamos a clasificar entre 3 o mas clases                                     
print(imagen_entrenamiento.class_indices)

#CREACION RED NEURONAL CONVOLUCIONAL
cnn = Sequential()    #Varias capas apiladas entre ellas                

cnn.add(Convolution2D(filtrosConv1, tam_filtro1, input_shape = (256,256,1), padding='same', activation = 'relu'))  #Se agrega la primera capa convolucional 
cnn.add(MaxPooling2D(pool_size = tam_pool))   #Se realiza un MaxPooling para extraer las caracteristicas mas importantes de la imagen y disminuir su tamaño

cnn.add(Convolution2D(filtrosConv2, tam_filtro2, padding='same'))  #Se agrega la segunda capa convolucional 
cnn.add(MaxPooling2D(pool_size = tam_pool))         

cnn.add(Flatten())                                  #Se realiza lo que se conoce como aplanamiento, transforma una matriz bidimensional de características en un vector que puede alimentar a un clasificador            
cnn.add(Dense(100,activation = 'relu'))             #Se agrega una capa que contiene N neuronas con una activacion Relu
cnn.add(Dropout(0.5))                               #El dropout lo que hace es que deshabilita el porcentaje deseado de neuronas en cada paso, para evitar un sobreajuste 
cnn.add(Dense(clases, activation = 'softmax'))      #Por ultimo se agrega una capa que contiene tantas neuronas como clases se tenga, con una activacion softmax
                                                    #Indica que tanta probabilidad tiene cada clase y por ende saber cual tiene la mayor probabilidad

#cnn.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(lr = lr), metrics = ['accuracy'])    #Se compila el modelo con una funcion de perdida Adam con tasa de apredizaje lr
cnn.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

H = cnn.fit(imagen_entrenamiento , epochs = epocas, batch_size = batch_size, verbose = 1)                       #Se entrena el modelo 
#print(imagen_entrenamiento.class_indices) 
#cnn.summary()

#GRAFICAS DE LA PERDIDA Y PRECISION DEL MODELO
N = epocas
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["accuracy"], label="train_acc")
plt.title("Entrenamiento Loss y Accuracy ")
plt.xlabel("Epoch #")
plt.xlabel("Loss/Accuracy")
plt.legend(loc="lower left")

#PRE-PROCESO DE LOS DATOS DE PRUEBA 
test_datagen = ImageDataGenerator(rescale = 1./255)       #Se normaliza las imagenes
test_generator = test_datagen.flow_from_directory(
    test_data,                                            #Datos de entrenamiento
    target_size = (altura, longitud),                     #Redimensiona a los tamaños establecidos
    color_mode="grayscale",                               #Tipo de imagen RGB
    batch_size = 1,                                       #Batch size anteriormente establecido
    class_mode = None,                                    #En esta ocasion sera un lote de imagenes por este motivo esta desactivado    
    shuffle=False,
    seed=42)

#PREDICCIONES DEL MODELO 
pred = cnn.predict(test_generator)        #Realiza la predicción del lote de datos de prueba 
predicted=np.argmax(pred,axis=1)          #Retorna el indice del valor de probabilidad mas alto que nos de la prediccion de un vector de tamaño N clases 
print(predicted)

filenames = test_generator.filenames      
result=pd.DataFrame({"Filename":filenames,"Predictions":predicted}) #Se agrega el nombre del archivo y su respectiva prediccion 

#TARGET
real_class_indices=[]                     #Se obtiene el Target de cada imagen 
for i in range(0, len(filenames)):        #Por medio del nombre de cada imagen se la identifica y se le da el valor de su indice en el vector de predicciones 
  if ("_A_" in filenames[i]):               #Tenemos como resultado la etiqueta real, a quien pertenece la imagen para posteriarmente hacer una comparacion 
    real_class_indices.append(0)
  if ("_B_" in filenames[i]):
    real_class_indices.append(11)
  if ("_C_" in filenames[i]):
    real_class_indices.append(13)
  if ("_D_" in filenames[i]):
    real_class_indices.append(14)
  if ("_E_" in filenames[i]):
    real_class_indices.append(15)
  if ("_F_" in filenames[i]):
    real_class_indices.append(16)
  if ("_G_" in filenames[i]):
    real_class_indices.append(17)
  if ("_H_" in filenames[i]):
    real_class_indices.append(18)
  if ("_I_" in filenames[i]):
    real_class_indices.append(19)
  if ("_J_" in filenames[i]):
    real_class_indices.append(1)
  if ("_K_" in filenames[i]):
    real_class_indices.append(2)
  if ("_L_" in filenames[i]):
    real_class_indices.append(3)
  if ("_M_" in filenames[i]):
    real_class_indices.append(4)
  if ("_N_" in filenames[i]):
    real_class_indices.append(5)
  if ("_Z_" in filenames[i]):
    real_class_indices.append(6)
  if ("_O_" in filenames[i]):
    real_class_indices.append(7)
  if ("_P_" in filenames[i]):
    real_class_indices.append(8)
  if ("_Q_" in filenames[i]):
    real_class_indices.append(9)
  if ("_R_" in filenames[i]):
    real_class_indices.append(10)
  if ("_S_" in filenames[i]):
    real_class_indices.append(12)
print(real_class_indices)

from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

data = pd.DataFrame({"Predic":predicted,"Originales":real_class_indices})   #Se crea un Dataframe con los indices originales(Target) y sus predicciones 
matriz = pd.crosstab (data ['Predic'], data ['Originales'], rownames = ['Predicciones'], colnames = ['Originales']) #Crea una matriz de confusión a partir del anterior Dataframe 
plt.figure(figsize=(10,7))
sn.heatmap (matriz, annot = True, fmt=" ", annot_kws={"size": 13})
#xticklabels=['Des','Duvan','Oscar'], yticklabels=['Des','Duvan','Oscar'])

data['Err'] = np.where(data['Predic'] == data['Originales'], 0, 1)
Error = data.sum()
Error[2]

from sklearn.metrics import classification_report
real = np.array(real_class_indices)
print(classification_report(real_class_indices,predicted))

"""# **Guardar y Cargar Modelo**"""

cnn.save('/content/drive/My Drive/Dataset Nariñense/Morse/modelos/Duv')  #Se guarda el modelo en Google Drive en la ruta deseada
