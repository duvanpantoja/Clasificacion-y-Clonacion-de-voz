function data3=VADPAL(data,fs)
[x,y]=detectSpeech(data,fs); %Se aplica voice activity detection proporcionado por matlab en donde registra donde hay actividad vocal
data2=data(x(1):x(end));        %se obtiene la muestra especifica del audio
data3=zeros(6*22051,1);         %Se crea un nuevo audio con 22051 muestras por cada 0.5 segundos en total 3 segundos
data3(22052:(22051+length(data2)))=data2; %el nuevo audio contiene los 0.5 segundos inciales de silencio, despues la muestra de audio y lo que reste del audio sera silencio
end
