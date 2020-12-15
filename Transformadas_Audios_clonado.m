clc
close all
clear all

myPath= 'C:\Users\USUARIO\Desktop\tesis\CLONADO'; %Ruta donde se encuentran los Audios que se quiere procesar 
fileNames = dir(fullfile(myPath, '*.wav'));    
 a=struct2table(fileNames(1));%Agrega a una estructura todas las caracteristicas de los archivos de tipo 
 %%                                                                        %(.wav .mp3 .m4a) o el que se requiera 
for (i=1:1:length(fileNames))           %Ciclo que contiene todo el procesamiento de imagenes, trabaja hasta procesar todos los audios cargados 
                                        %Se recomienda realizar el proceamiento seccion por seccion cuando se trabajo con gran cantodad de audios debido a que consumen recursos
                                        %computacionales considerables 
    a=struct2table(fileNames(i));       %Se cambia de tipo estructura a tabla los nombres de los archivos para poder leerlos con audioread   
    [data2, fs] = audioread(a.name);    
    data=VADPAL(data2,fs);
    x = length(data)/fs;                %Parametro importante para la creacion del vector de Tiempo 
    time = linspace(0,x,length(data));  %Creacion del vector de tiempo 
    num=num2str(i);                     %Convierte el contador a str, este se agrega al nombre de la imagen para evitar que se sobrescriban al guardarla
    let = a.name;
%% Wavelets

% %% Wavelet Amor BW
    x=cwt(data(:,1),'Amor',fs);      %Se aplica la tranformada wavelet Amor, se obtienen sus coeficientes y su frec 
    y=imresize(abs(x),[512 512]);    %Se realiza un resize
    I=50*log10(y+1);                 %Se realiza un re-dimensionamiento de la imagen a 512x512
    I2 = cat(3,I,I,I);               %opcional si se quiere la imagen en 3 capas (RGB), por defecto una capa (BW)
    imwrite(I2,[let(1:5),'.jpg']);   %se guarda la imagen en la ruta donde esta el archivo matlab

    clear data fs
    close all 
    
end
