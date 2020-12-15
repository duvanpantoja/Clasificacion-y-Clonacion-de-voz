clc
close all
clear all

myPath= 'C:\Users\USUARIO\Desktop\tesis\HABLANTES NARIÑENSE\PRUEBA 4 Y 5\HABLANTE 4\5 CLON'; %Ruta donde se encuentran los Audios que se quiere procesar 
fileNames = dir(fullfile(myPath, '*.wav'));  

for (var=1:1:length(fileNames)) 
    a=struct2table(fileNames(var))        %Se cambia de tipo estructura a tabla los nombres de los archivos para poder leerlos con audioread   
    [data2 fs] = audioread(a.name);       %Se obtiene los valores y la frecuencia de muestreo de cada uno de los audios 
    data=VADPAL(data2,fs);                %se aplica Voice Activity Detection para obtener un espectro sincronizado en un mismo instante de tiempo
    
    x=cwt(data(:,1),'Amor',fs);            %Se aplica la tranformada wavelet Amor, se obtienen sus coeficientes y su frec 
    
    nom=a.name;                           %se extrae el nombre del archivo de audio que ingreso al ciclo

   b=imread([nom(1:6),'.jpg']);           %lee la imagen que debe tener el mismo nombre del audio

    
    [L,A]=size(x);                      %se guarda el tamaño de la matriz resultante al aplicar la wavelet
    I=rgb2gray(b);                      %se convierte la imagen de RGB a BW
    I2=rescale(I);                      %Se reescala la imagen a valores de (0,1)
    y2=imresize(I2,[L A]);              %se redimensiona la imagen a los valores de la matriz de la wavelet
    z=10.^(y2/50)-1;                    %Se hace la operacion inversa de la operacion logaritmo
    ph = angle(x);                      %se obtiene la fase del audio de entrada a traves de su wavelet
    p = z.* exp(i*ph);                  %se aplica la fase a la imagen de entrada 
    c = icwt(p);                        %se realiza la transformada inversa wavelet para obtener un audio
    
    audiowrite([nom(1:6),'_C_5','.wav'],c,fs);  %Se guarda el audio con la frecuencia de muestreo inicial

    
end



