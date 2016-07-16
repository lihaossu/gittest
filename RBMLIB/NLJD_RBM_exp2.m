%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This experiment choose the difference between Second and Third harmonics
% as features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%================Step 0: Generating data ================================
clc; clear all; close all;
num_tr = 500; % number of training sets
num_te = 200; % number of testing sets
Training_Standard_Deviation = 50;
% Metal
Metal_x1 = 600; Metal_y1 = 200; Metal_x2 = 900; Metal_y2 = 400;
% Test Metal
Metal_te_x = 750; Metal_te_y = 300;
label0 = 1*ones(length(Metal_x1),1);
% Electronic
Electronic_x1 = 300; Electronic_y1 = 700; Electronic_x2 = 800;Electronic_y2 =1400;
Electronic_te_x = 550; Electronic_te_y = 1050;
label1 = -1*ones(length(Electronic_x1),1);
num_dim = 50;
% num_tr = 500 , so the total training data number is 2000
for i = (1 : num_tr),
    Metal1_x1(i,:) = Metal_x1 + Training_Standard_Deviation*randn(1,num_dim);
    Metal2_x2(i,:) = Metal_x2 + Training_Standard_Deviation*randn(1,num_dim);
    Metal1_y1(i,:) = Metal_y1 + Training_Standard_Deviation*randn(1,num_dim);
    Metal2_y2(i,:) = Metal_y2 + Training_Standard_Deviation*randn(1,num_dim);
    data1_metal(i,:) = Metal1_x1(i,:) - Metal1_y1(i,:);
    data2_metal(i,:) = Metal2_x2(i,:) - Metal2_y2(i,:);
    Electronic1_x1(i,:) = Electronic_x1 + Training_Standard_Deviation*randn(1,num_dim);
    Electronic1_y1(i,:) = Electronic_y1 + Training_Standard_Deviation*randn(1,num_dim);
    Electronic2_x2(i,:) = Electronic_x2 + Training_Standard_Deviation*randn(1,num_dim);
    Electronic2_y2(i,:) = Electronic_y2 + Training_Standard_Deviation*randn(1,num_dim);
    data1_Elec(i,:) = Electronic1_x1(i,:) - Electronic1_y1(i,:);
    data2_Elec(i,:) = Electronic2_x2(i,:) - Electronic2_y2(i,:);
end
label0 = 1 * ones(length(data1_metal),1);
label1 = 0 * ones(length(data1_Elec),1);
% data 1,2,3,4
data1 = data1_metal;
data2 = data2_metal;
data3 = data1_Elec;
data4 = data2_Elec;
data = [data1;data2;data3;data4];
%scale the data to [0,1] range 
max1 = max(data(:,1:50));
min1 = min(data(:,1:50));
for i = (1:num_tr*4),
    data_scaled(i,:) = (data(i,:) - min1)./(max1 - min1);
end
labels = [label0;label0;label1;label1];
data_pre = [data_scaled , labels];
data_tr_shuffle = [data_pre,labels];
[n_tr_row,n_tr_col] = size(data_tr_shuffle);
shuffle_tr_seq = randperm(n_tr_row);
for i = (1:n_tr_row),
    data_tr_shuffledall(i,:) = data_tr_shuffle(shuffle_tr_seq(i),:);
end
data_tr_shuffled = data_tr_shuffledall(:,1:50);
labels_tr_shuffled = data_tr_shuffledall(:,51);
%% Testdata num_te = 200, so the total test data number is 400
for i = (1:num_te),
    Metal1_te_x(i,:) = Metal_te_x + Training_Standard_Deviation*randn(1,num_dim);
    Metal1_te_y(i,:) = Metal_te_y + Training_Standard_Deviation*randn(1,num_dim);
    data1_te_metal(i,:) = Metal1_te_x(i,:) - Metal1_te_y(i,:);
    Electronic1_te_x(i,:) = Electronic_te_x + Training_Standard_Deviation*randn(1,num_dim);
    Electronic1_te_y(i,:) = Electronic_te_y + Training_Standard_Deviation*randn(1,num_dim);
    data1_te_Elec(i,:) = Electronic1_te_x(i,:) - Electronic1_te_y(i,:);
end
label_te_0 = 1 * ones(length(data1_te_metal),1);
label_te_1 = 0 * ones(length(data1_te_Elec),1);
% data_te 1,2
data_te1 = data1_te_metal;
data_te2 = data1_te_Elec;
data_te = [data_te1;data_te2];
%scale the test data to [0,1] range
max1_te = max(data_te(:,1:50));
min1_te = min(data_te(:,1:50));
for i = (1:num_te*2),
    data_te_scaled(i,:) = (data_te(i,:) - min1_te)./(max1_te - min1_te);
end
labels_te = [label_te_0;label_te_1];
data_te_pre = [data_te_scaled,labels_te];
data_te_shuffle = [data_te_pre,labels_te];
[n_te_row,n_te_col] = size(data_te_shuffle);
shuffle_te_seq = randperm(n_te_row);
for i = (1:n_te_row),
    data_te_shuffledall(i,:) = data_te_shuffle(shuffle_te_seq(i),:);
end
data_te_shuffled = data_te_shuffledall(:,1:50);
labels_te_shuffled = data_te_shuffledall(:,51);
%% Train RBM for classification
%train rbm with 100 hidden units
m=rbmFit(data_tr_shuffled,100,labels_tr_shuffled,'verbose',true);
yhat=rbmPredict(m,data_te_shuffled);




%print error
fprintf('Classification error using RBM with 100 hiddens is %f\n', ...
    sum(yhat~=labels_te_shuffled)/length(yhat));



