load mnist_classify;
%% Train RBM for classification
X = data;
y = labels;
numhid = 100;
nclasses = nunique(y);
method = 'CD';
eta = 0.1; % learning rate
momentum = 0.5;%?????????????????
maxepoch = 100;
avglast = 5;%??????????????????
penalty = 2e-4;
batchsize = 100;
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
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    

















