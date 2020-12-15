clear all
close all 
clc 

Modelos = categorical({'hablanteA','hablanteB','ResNet152','InceptionV3','InceptionResNetV2','DenseNet201','Xception','ResNet101','MobileNet',...
                        'MobileNetV2','NasNetMobile','EfficientNetB0','EfficientNetB3','EfficientNetB7'});
Amor = [93.4 95.2 92.8 82.2 93.8 90.2 91.4 89.4 92.4 85.6 83.2 89.6 82.6 78];
Black= [93.6 93.6 87.4 78.4 85.6 87.4 84.8 84 88.4 85.6 73.4 88.4 84.8 82.2];
Bump= [90.2 92.8 93.4 80.6 91.4 90.6 89.6 88 87 91.8 80.4 86.4 85.8 84.2];
Hamm= [96.4 93.8 83.6 82.2 90 88.4 86 85.8 87 84.4 76 89.8 85 82.4];
Kaiser= [96.2 94.8 86 79.8 91.6 88 87.6 83.6 87.2 82.4 72.2 91.4 85.2 80.6];
Morse= [94.8 95.2 92.8 83.6 89 92.8 90.4 89 90.2 90 82.2 90.2 84.6 81.2];

AmorP = mean(Amor);
BlackP = mean(Black);
BumpP = mean(Bump);
HammP = mean(Hamm);
KaiserP = mean(Kaiser);
MorseP = mean(Morse);
vet = [AmorP;BlackP;BumpP;HammP;KaiserP;MorseP];
[a,ven]=max(vet);
[a,mven]=min(vet);

Mat =[Amor; Black; Bump; Hamm; Kaiser; Morse];
Mat2=mean(Mat,1);
[a,barra]=max(Mat2);
[a,mbarra]=min(Mat2);

%% Graficas 
figure()
bar(Amor,'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5)
grid on;
set(gca,'Fontname','Times New Roman','FontSize',12)
title('Wavelet Amor','FontSize',17);
xticklabels({'Duvan','Oscar','ResNet152','InceptionV3','InceptionResNetV2','DenseNet201','Xception','ResNet101','MobileNet',...
                        'MobileNetV2','NasNetMobile','EfficientNetB0','EfficientNetB3','EfficientNetB7'})
ylabel('Porcentaje de Error %','FontSize',15)
xlabel('MODELOS','FontSize',15)
h=gca;    
h.XTickLabelRotation = 45;


figure()
bar(Black,'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5)
grid on;
set(gca,'Fontname','Times New Roman','FontSize',12)
title('Ventana Blackman','FontSize',17);
xticklabels({'Duvan','Oscar','ResNet152','InceptionV3','InceptionResNetV2','DenseNet201','Xception','ResNet101','MobileNet',...
                        'MobileNetV2','NasNetMobile','EfficientNetB0','EfficientNetB3','EfficientNetB7'})
ylabel('Porcentaje de Error %','FontSize',15)
xlabel('MODELOS','FontSize',15)
h=gca;    
h.XTickLabelRotation = 45;
ylim([0,100])

figure()
bar(Bump,'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5)
grid on;
set(gca,'Fontname','Times New Roman','FontSize',12)
title('Wavelet Bump','FontSize',17);
xticklabels({'Duvan','Oscar','ResNet152','InceptionV3','InceptionResNetV2','DenseNet201','Xception','ResNet101','MobileNet',...
                        'MobileNetV2','NasNetMobile','EfficientNetB0','EfficientNetB3','EfficientNetB7'})
ylabel('Porcentaje de Error %','FontSize',15)
xlabel('MODELOS','FontSize',15)
h=gca;    
h.XTickLabelRotation = 45;

figure()
bar(Hamm,'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5)
grid on;
set(gca,'Fontname','Times New Roman','FontSize',12)
title('Ventana Hamming','FontSize',17);
xticklabels({'Duvan','Oscar','ResNet152','InceptionV3','InceptionResNetV2','DenseNet201','Xception','ResNet101','MobileNet',...
                        'MobileNetV2','NasNetMobile','EfficientNetB0','EfficientNetB3','EfficientNetB7'})
ylabel('Porcentaje de Error %','FontSize',15)
xlabel('MODELOS','FontSize',15)
h=gca;    
h.XTickLabelRotation = 45;
ylim([0,100])

figure()
bar(Kaiser,'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5)
grid on;
set(gca,'Fontname','Times New Roman','FontSize',12)
title('Ventana Kaiser','FontSize',17);
xticklabels({'Duvan','Oscar','ResNet152','InceptionV3','InceptionResNetV2','DenseNet201','Xception','ResNet101','MobileNet',...
                        'MobileNetV2','NasNetMobile','EfficientNetB0','EfficientNetB3','EfficientNetB7'})
ylabel('Porcentaje de Error %','FontSize',15)
xlabel('MODELOS','FontSize',15)
h=gca;    
h.XTickLabelRotation = 45;

figure()
bar(Morse,'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5)
grid on;
set(gca,'Fontname','Times New Roman','FontSize',12)
title('Wavelet Morse','FontSize',17);
xticklabels({'Duvan','Oscar','ResNet152','InceptionV3','InceptionResNetV2','DenseNet201','Xception','ResNet101','MobileNet',...
                        'MobileNetV2','NasNetMobile','EfficientNetB0','EfficientNetB3','EfficientNetB7'})
ylabel('Porcentaje de Error %','FontSize',15)
xlabel('MODELOS','FontSize',15)
h=gca;    
h.XTickLabelRotation = 45;

%% Promedios Ventanas
figure()
b=bar([AmorP, BlackP, BumpP, HammP, KaiserP, MorseP],'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5)
b.FaceColor = 'flat';
b.CData(ven,:)=[0.2 0.5 0.7];
b.CData(mven,:)=[0.8 0 0.2];
grid on;
set(gca,'Fontname','Times New Roman','FontSize',12)
title('Promedio de precision Ventanas','FontSize',17);
xticklabels({'Amor','Black','Bump','Hamm','Kaiser','Morse'})
ylabel('Accuracy %','FontSize',15)
xlabel('VENTANA','FontSize',15)
h=gca;    
h.XTickLabelRotation = 45;
ylim([0,100])

%% Promedios Ventanas
figure()
b=bar(Mat2,'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5)
b.FaceColor = 'flat';
b.CData(barra,:)=[0.2 0.5 0.7];
b.CData(mbarra,:)=[0.8 0 0.2];
grid on;
set(gca,'Fontname','Times New Roman','FontSize',12)
title('Promedio de precision por Modelo','FontSize',17);
xticklabels({'Duvan','Oscar','ResNet152','InceptionV3','InceptionResNetV2','DenseNet201','Xception','ResNet101','MobileNet',...
                        'MobileNetV2','NasNetMobile','EfficientNetB0','EfficientNetB3','EfficientNetB7'})
ylabel('Accuracy %','FontSize',15)
xlabel('MODELO','FontSize',15)
h=gca;    
h.XTickLabelRotation = 45;
