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
    [data, fs] = audioread(a.name);
    data = data(:,1);%Se obtiene los valores y la frecuencia de muestreo de cada uno de los audios 
    %sound(data,fs);                    %Permite escuchar el audio que esta siendo procesado  
    x = length(data)/fs;                %Parametro importante para la creacion del vector de Tiempo 
    time = linspace(0,x,length(data));  %Creacion del vector de tiempo 
    num=num2str(i);                     %Convierte el contador a str, este se agrega al nombre de la imagen para evitar que se sobrescriban al guardarla
    let = a.name;
%% Wavelets
% %% Wavelet Bump BW
    x=cwt(data(:,1),'bump',fs);      %Se aplica la tranformada wavelet Bump, se obtienen sus coeficientes y su frec 
    y=imresize(abs(x),[512 512]);    %Se realiza un re-dimensionamiento de la imagen a 512x512
    I=50*log10(y+1);                 %se aplica una operacion logaritmo para acentuar la imagen
    %I2 = cat(3,I,I,I);               %opcional si se quiere la imagen en 3 capas (RGB), por defecto una capa (BW)
    imwrite(I,['Bump_',let(1),'_',num,'.jpg']);  %se guarda la imagen en la ruta donde esta el archivo matlab
% %% Wavelet Amor BW
    x=cwt(data(:,1),'Amor',fs);      %Se aplica la tranformada wavelet Amor, se obtienen sus coeficientes y su frec 
    y=imresize(abs(x),[512 512]);    %Se realiza un resize
    I=50*log10(y+1);                 %Se realiza un re-dimensionamiento de la imagen a 512x512
    %I2 = cat(3,I,I,I);               %opcional si se quiere la imagen en 3 capas (RGB), por defecto una capa (BW)
    imwrite(I2,[let(1:5),'.jpg']);   %se guarda la imagen en la ruta donde esta el archivo matlab
% %% Wavelet Morse BW
    x=cwt(data(:,1),fs);              %Se aplica la tranformada wavelet Morse, se obtienen sus coeficientes y su frec 
    y=imresize(abs(x),[512 512]);     %Se realiza un re-dimensionamiento de la imagen a 512x512 
    I=50*log10(y+1);                  %se aplica una operacion logaritmo para acentuar la imagen
    imwrite(I,['Morse_',let(1),'_',num,'.jpg']);  %se guarda la imagen en la ruta donde esta el archivo matlab

 %%   
    % STFT (short-time-fourier-transform)
% Ventana Hamming BW
    wlen = 256;                             %Tamaño de la Ventana
    hop = wlen/4;                             %Tamaño de salto
    nfft = 512;                               %Numero de puntos fft
    win = hamming(wlen, 'periodic');          %Ventana Hamming
    x=stft(data(:,1),fs,'Window',win,'OverlapLength',hop,'FFTLength',nfft);  %Se aplica stft con ventana hamming, se obtienen sus coeficientes y su frec 
    I=imresize(abs(x),[512 512]);               %Se realiza un re-dimensionamiento de la imagen a 512x512
    I=50*log10(I+1);                            %se aplica una operacion logaritmo para acentuar la imagen
    %I2 = cat(3,I,I,I);                          %opcional si se quiere la imagen en 3 capas (RGB), por defecto una capa (BW)
    imwrite(I2,[let(1:5),'.jpg']);              %se guarda la imagen en la ruta donde esta el archivo matlab
%% Ventana Blackman BW
    wlen = 256;                             %Tamaño de la Ventana
    hop = wlen/4;                             %Tamaño de salto
    nfft = 512;                               %Numero de puntos fft
    win = blackman(wlen, 'periodic');          %Ventana Blackman 
    x=stft(data(:,1),fs,'Window',win,'OverlapLength',hop,'FFTLength',nfft);  %Se aplica stft con ventana blackman, se obtienen sus coeficientes y su frec 
    I=imresize(abs(x),[512 512]);               %Se realiza un re-dimensionamiento de la imagen a 512x512
    I=50*log10(I+1);                            %se aplica una operacion logaritmo para acentuar la imagen
    imwrite(I,['Black_',let(1),'_',num,'.jpg']);  %se guarda la imagen en la ruta donde esta el archivo matlab
% %% Ventana Kaiser BW
    wlen = 256;                             %Tamaño de la Ventana
    hop = wlen/4;                             %Tamaño de salto
    nfft = 512;                               %Numero de puntos fft
    win = kaiser(wlen,5);                   %Ventana Kaiser
    x=stft(data(:,1),fs,'Window',win,'OverlapLength',hop,'FFTLength',nfft);  %Se aplica stft con ventana kaiser, se obtienen sus coeficientes y su frec 
    I=imresize(abs(x),[512 512]);           %Se realiza un re-dimensionamiento de la imagen a 512x512
    I=50*log10(I+1);                        %se aplica una operacion logaritmo para acentuar la imagen
    imwrite(I,['Kaiser_',let(1),'_',num,'.jpg']);  %se guarda la imagen en la ruta donde esta el archivo matlab
%%
    clear data fs
    close all 
    
end

%%
% [audioIn,fs] = audioread('DF1 (1).wav');
% S=mfcc(audioIn,fs); 
% figure()
% plot(abs(S))

%%
% [audioIn,fs] = audioread('DF1 (1).wav');
% melSpectrogram(audioIn,fs)

%%
% [audioIn,fs] = audioread('DF1 (1).wav');
% S = melSpectrogram(audioIn,fs);
% I=imresize(abs(S),[512 512]); 
% I=50*log10(I+1);  
% imshow(I)

% I=imresize(S,[512 512]); 
% I2=50*log10(I+1);
% imshow(I2)
% wlen = 256;                             %Tamaño de la Ventana
% hop = wlen/4;                             %Tamaño de salto
% nfft = 512;                               %Numero de puntos fft
% win = hamming(wlen, 'periodic');          %Ventana Hamming
% x=stft(data(:,1),fs,'Window',win,'OverlapLength',hop,'FFTLength',nfft);
% I=imresize(abs(x),[512 512]); 
% I=50*log10(I+1);                   
% imshow(I)

