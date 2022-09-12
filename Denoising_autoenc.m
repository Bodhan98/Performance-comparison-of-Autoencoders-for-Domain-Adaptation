clear
close all
clc

rng('default')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
training=1;
train1=training;

load Kolkata_classify.mat;
dataset1=dataset;
dataset1=normalize(dataset1,1,'norm');
labels1=labels;
load Ahmedabad_classify.mat;
dataset=normalize(dataset,1,'norm');

%Add white noise to training data
noise=wgn(size(dataset,1),size(dataset,2),1);
noise=0.1*normalize(noise,1,'norm');
dataset_noisy=dataset+noise;

%Main Program

if training==1
    
hiddenSize = 100;
denoising_autoenc = trainAutoencoder(dataset_noisy,hiddenSize,'MaxEpochs',1000,'EncoderTransferFunction','satlin','DecoderTransferFunction','purelin','L2WeightRegularization',1e-5,'SparsityRegularization',4, 'SparsityProportion',0.10,'UseGPU',true);
save denoising_autoenc;

else
    
    load denoising_autoenc;
    
end

dataset_reconstructed=predict(denoising_autoenc,dataset_noisy);
mseError=mse(dataset-dataset_reconstructed);
disp(mseError);

%Domain Adaptation
dataset1_reconstructed=predict(denoising_autoenc,dataset1);
mseError1=mse(dataset1-dataset1_reconstructed);
disp(mseError1);

%Plots
figure(1)
subplot(2,2,1),scatter(dataset(:,1),dataset(:,2),'Marker','^');
xlabel('dim1');
ylabel('dim2');
title('Ahmedabad(original)');
grid on;
subplot(2,2,2),scatter(dataset_reconstructed(:,1),dataset_reconstructed(:,2),'Marker','^');
xlabel('dim1');
ylabel('dim2');
title('Ahmedabad(reconstructed)');
grid on;
subplot(2,2,3),scatter(dataset1(:,1),dataset1(:,2),'Marker','<');
xlabel('dim1');
ylabel('dim2');
title('Kolkata(original)');
grid on;
subplot(2,2,4),scatter(dataset1_reconstructed(:,1),dataset1_reconstructed(:,2),'Marker','<');
xlabel('dim1');
ylabel('dim2');
title('Kolkata(reconstructed)');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Classification using sparse autoencoder
feat1=encode(denoising_autoenc,dataset_noisy);

if train1==1
    softnet=trainSoftmaxLayer(feat1,labels,'MaxEpochs',1000);
    save softnet1;
else
    load softnet1;
end

if training==1
    
    denoising_stacknet=stack(denoising_autoenc,softnet);
    denoising_stacknet=train(denoising_stacknet,dataset,labels);
    save denoising_stacknet;
    
else
    
    load denoising_stacknet;
    
end

y=denoising_stacknet(dataset1);

view(denoising_autoenc);
view(softnet);
view(denoising_stacknet);

figure(2)
plotconfusion(labels1,y);
figure(3)
plotroc(labels1,y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%