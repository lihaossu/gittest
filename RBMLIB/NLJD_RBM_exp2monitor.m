%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This datagenerator plan to generate one object dataset including 
% training data , taining labels and test data , test labels.
% failed !!! the problem should be the feature. Actually this project
% features are not the second and third harmonics but should be the differ
% between the third and second harmonics.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




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
data_tr_shuffle = data_pre;
[n_tr_row,n_tr_col] = size(data_tr_shuffle);
shuffle_tr_seq = randperm(n_tr_row);
for i = (1:n_tr_row),
    data_tr_shuffledall(i,:) = data_tr_shuffle(shuffle_tr_seq(i),:);
end
data_tr_shuffled = data_tr_shuffledall(:,1:50);
labels_tr_shuffled = data_tr_shuffledall(:,51);
%% Testdata num_te = 200, so the total test data number is 400
Training_Standard_Deviation_te = 40;
for i = (1:num_te),
    Metal1_te_x(i,:) = Metal_te_x + Training_Standard_Deviation_te*randn(1,num_dim);
    Metal1_te_y(i,:) = Metal_te_y + Training_Standard_Deviation_te*randn(1,num_dim);
    data1_te_metal(i,:) = Metal1_te_x(i,:) - Metal1_te_y(i,:);
    Electronic1_te_x(i,:) = Electronic_te_x + Training_Standard_Deviation_te*randn(1,num_dim);
    Electronic1_te_y(i,:) = Electronic_te_y + Training_Standard_Deviation_te*randn(1,num_dim);
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
data_te_shuffle = data_te_pre;
[n_te_row,n_te_col] = size(data_te_shuffle);
shuffle_te_seq = randperm(n_te_row);
for i = (1:n_te_row),
    data_te_shuffledall(i,:) = data_te_shuffle(shuffle_te_seq(i),:);
end
data_te_shuffled = data_te_shuffledall(:,1:50);
labels_te_shuffled = data_te_shuffledall(:,51);
%% Train RBM for classification
X = data_tr_shuffled;
y = labels_tr_shuffled;
testdata = data_te_shuffled;
testlabels = labels_te_shuffled;
numhid = 100;
nclasses = nunique(y);
method = 'CD';
eta = 0.1; % learning rate
momentum = 0.5;%?????????????????
maxepoch = 50;
avglast = 5;%??????????????????
penalty = 2e-4;
batchsize = 50;
anneal = false; 
avgstart = maxepoch - avglast;%why need this? following 
oldpenalty = penalty;
[N,d]= size(X);


fprintf('Processing data ... \n');

%Create targets matrix: 1-of-k encodings for each discrete label
u = unique(y);
targets = zeros(N, nclasses);
for i=1:length(u)
    targets(y==u(i),i)=1;% y == u(i)  amazing , very useful! 
end

%Creat batches
numbatches = ceil(N/batchsize);% ceil(7/3) == 3
groups = repmat(1:numbatches, 1, batchsize);  % group of data
groups = groups(1:N);
groups = groups(randperm(N));% shuffle the groups
for i = 1:numbatches
    batchdata{i} = X(groups==i,:);
    batchtargets{i} = targets(groups==i, :);
end

%fit RBM
numcases = N;
numdims = d;
numclasses = length(u);
W = 0.1*randn(numdims,numhid);%  weight between X and  hidden layer
c = zeros(1,numdims);% bias of visible unit of X
b = zeros(1,numhid); % bias of hidden unit 
Wc = 0.1*randn(numclasses,numhid);% weight between  Y and hidden layer
cc = zeros(1,numclasses);% bias of visible unit of Y(label)
ph = zeros(numcases,numhid);%previous hidden 
nh = zeros(numcases,numhid);%next hidden
phstates = zeros(numcases,numhid);
nhstates = zeros(numcases,numhid);
negdata = zeros(numcases,numdims);% ????
negdatastates = zeros(numcases,numdims);
Winc = zeros(numdims,numhid);
binc = zeros(1,numhid);
cinc = zeros(1,numdims);
Wcinc = zeros(numclasses,numhid);
ccinc = zeros(1,numclasses);
Wavg = W;% why average????
bavg = b;
cavg = c;
Wcavg = Wc;
bavg = b;
cavg = c;
Wcavg = Wc;
ccavg = cc;
t = 1;
errors = zeros(1,maxepoch);


for epoch = 1 : maxepoch
    
    errsum = 0;
    if (anneal)% the purpose? here is a option to choose a stable rate or not
        penalty = oldpenalty - 0.9*epoch/maxepoch*oldpenalty;
    end
    
    for batch = 1:numbatches
        [numcases numdims] = size(batchdata{batch});
        data = batchdata{batch};
        classes = batchtargets{batch};
        
        %go up
        ph = logistic(data*W + classes*Wc + repmat(b, numcases,1));
        phstates = ph > rand(numcases, numhid);%????????
        if (isequal(method,'SML'))
            if (epoch == 1 && batch == 1)
                nhstates = phstates;
            end
        elseif (isequal(method,'CD'))
            nhstates = phstates;
        end
        
        %go down
        negdata = logistic(nhstates*W' + repmat(c,numcases,1));% logistic function
        negdatastates = negdata > rand(numcases,numdims);
        negclasses = softmaxPmtk(nhstates*Wc' + repmat(cc,numcases,1));% softmax!!!!! why use softmaxPmtk??
        negclassesstates = softmax_sample (negclasses);% however here is softmax_sample???debug it
        
        %go up one more time
        nh = logistic(negdatastates*W + negclassesstates*Wc + ...  
            repmat(b,numcases,1));
        nhstates = nh > rand(numcases,numhid);
        
        %update weights and biases
        dW = (data'*ph - negdatastates'*nh);
        dc = sum(data) - sum(negdatastates);
        db = sum(ph) - sum(nh);
        dWc = (classes'*ph - negclassesstates'*nh);
        dcc = sum(classes) - sum(negclassesstates); 
        Winc = momentum*Winc + eta*(dW/numcases - penalty*W);
        binc = momentum*binc + eta*(db/numcases);
        cinc = momentum*cinc + eta*(dc/numcases);
        Wcinc = momentum*Wcinc + eta*(dWc/numcases - penalty*Wc);
        ccinc = momentum*ccinc + eta*(dcc/numcases);
        W  = W + Winc;
        b  = b + binc;
        c  = c + cinc;
        Wc = Wc + Wcinc;
        cc = cc + ccinc;
        
        ccinc = momentum*ccinc + eta*(dcc/numcases);
        W = W + Winc;
        b = b + binc;
        c = c + cinc;
        Wc = Wc + Wcinc;
        cc = cc + ccinc;
        
        if (epoch > avgstart)% Attemtion ?? maybe For the last update of ..
            % the hidden units, it is silly to use stochastic binary ...
            % states because nothing depends on which state is chosen...
            %
            %apply averaging
            Wavg = Wavg - (1/t)*(Wavg - W);
            cavg = cavg - (1/t)*(cavg - c);
            bavg = bavg - (1/t)*(bavg - b);
            ccavg = ccavg - (1/t)*(ccavg - cc);
            t = t+1;
        else
            Wavg = W;
            bavg = b;
            cavg = c;
            Wcavg = Wc;
            ccavg = cc;
        end
        
        %accumulate reconstruction error
        err = sum(sum((data - negdata).^2));
        errsum = err + errsum;
    end
    
    errors(epoch)=errsum;
    
    fprintf('Ended epoch %i/%i, Reconstruction error is %f\n', ...
            epoch, maxepoch, errsum);
end

model.W  = Wavg;
model.b  = bavg;
model.c  = cavg;
model.Wc = Wcavg;
model.cc = ccavg;
model.labels = u;
            
 % Test        
 %% Prediction
 % Use RBM to predict discrete label for testdata
 
 %INPUTS:
 %m         ...is the model from rbmFit() consisting of W,b,c,Wc,cc
 %testdata  ...binary, or in [0,1] interpreted as probabilities
 
 %OUTPUTS:
 %prediction ...the discrete labels for every class
 
 numclasses = size(model.Wc, 1);
 numcases = size(testdata,1);
 F = zeros(numcases, numclasses);
 
 %set every class bit in turn and find -free energy of the configuration
 for i = 1:numclasses
     T = zeros(numcases,numclasses);
     T(:,i)=1;
     F(:,i) = repmat(model.cc(i),numcases,1).*T(:,i)+ ...
         sum(log(exp(testdata*model.W+ ...
         T*model.Wc + repmat(model.b,numcases,1))+1),2);
 end
 
 % take the max
 % compare the each label result ,the largest one means the probability is largest.
 [q,predid] = max(F, [],2);
 prediction = zeros(size(predid));
 for i = 1 : length(prediction) % convert back to users labels
     prediction(i) = model.labels(predid(i));
 end
 % error rate
 yhat = prediction;
 error = sum(yhat~=testlabels)/length(yhat);
 fprintf('Classification error using RBM with hiddens is %f\n', error);
 

