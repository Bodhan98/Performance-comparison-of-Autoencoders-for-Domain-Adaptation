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

%Main Program


if training==1
    
hiddenSize = 100;
sparse_autoenc = trainAutoencoder(dataset,hiddenSize,'MaxEpochs',1000,'EncoderTransferFunction','satlin','DecoderTransferFunction','purelin','L2WeightRegularization',1e-5,'SparsityRegularization',4, 'SparsityProportion',0.10,'UseGPU',true);
save sparse_autoenc;

else
    
    load sparse_autoenc;
    
end

dataset_reconstructed=predict(sparse_autoenc,dataset);
mseError=mse(dataset-dataset_reconstructed);
disp(mseError);

%Domain Adaptation
dataset1_reconstructed=predict(sparse_autoenc,dataset1);
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
feat1=encode(sparse_autoenc,dataset);

if train1==1
    softnet=trainSoftmaxLayer(feat1,labels,'MaxEpochs',1000);
    save softnet;
else
    load softnet;
end


if training==1
    
    sparse_stacknet=stack(sparse_autoenc,softnet);
    sparse_stacknet=train(sparse_stacknet,dataset,labels);
    save sparse_stacknet;
    
else
    
    load sparse_stacknet;
    
end
    
y=sparse_stacknet(dataset1);

view(sparse_autoenc);
view(softnet);
view(sparse_stacknet);

figure(2)
plotconfusion(labels1,y);
figure(3)
plotroc(labels1,y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%