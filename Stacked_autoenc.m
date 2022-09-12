clear
close all
clc

rng('default');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
training=1;

load Kolkata_classify.mat;
dataset1=dataset;
dataset1=normalize(dataset1,1,'norm');
labels1=labels;
load Ahmedabad_classify.mat;
dataset=normalize(dataset,1,'norm');

%Main Program
hiddenSize1=100;
hiddenSize2=60;
hiddenSize3=20;

if training==1
    
    autoenc1=trainAutoencoder(dataset,hiddenSize1,'MaxEpochs',1000,'EncoderTransferFunction','satlin','DecoderTransferFunction','purelin','L2WeightRegularization',1e-5,'SparsityRegularization',4, 'SparsityProportion',0.10,'UseGPU',true);
    feat1=encode(autoenc1,dataset);
    autoenc2=trainAutoencoder(feat1,hiddenSize2,'MaxEpochs',1000,'EncoderTransferFunction','satlin','DecoderTransferFunction','purelin','L2WeightRegularization',1e-5,'SparsityRegularization',4, 'SparsityProportion',0.10,'UseGPU',true);
    feat2=encode(autoenc2,feat1);
    autoenc3=trainAutoencoder(feat2,hiddenSize3,'MaxEpochs',1000,'EncoderTransferFunction','satlin','DecoderTransferFunction','purelin','L2WeightRegularization',1e-5,'SparsityRegularization',4, 'SparsityProportion',0.10,'UseGPU',true);
    feat3=encode(autoenc3,feat2);
    softnet2=trainSoftmaxLayer(feat3,labels,'MaxEpochs',1000);
    
    save autoenc1;
    save autoenc2;
    save autoenc3;
    save softnet2;
    
else
    
    load autoenc1;
    load autoenc2;
    load autoenc3;
    load softnet2;
    
end

%Perform fine tuning
if training==1
    
stackednet=stack(autoenc1,autoenc2,autoenc3,softnet2);    
stackednet=train(stackednet,dataset,labels);
save stackednet;

else
    load stackednet;
    
end

view(stackednet);
    
%Domain Adaptation

y=stackednet(dataset1);

%Plots
figure(1)
plotconfusion(labels1,y);
figure(2)
plotroc(labels1,y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%