function [X,Fx,StopMessage] = pso(ObjectiveFunction,Nvars,LB,UB)
%% Default Parameters
C1 = 1.2;
C2 = 0.012;
MaxIterations = 10000;
N = 30;
W = 0.0004;
StopMessage = 'Stopping PSO : Reached Maximum Number of Iterations';
%% Check Input            
if nargin < 4
    error('PSO : Not Enough Input Arguments');
end
if ~isequal(Nvars,size(LB,2),size(UB,2))
    error('PSO : Invalid Arguments : isequal(Nvars,size(LB,2),size(UB,2)) should be true ');
end 
%% Initial Position and Velocity

CurrentPosition = zeros(N,Nvars); % Initial Position
for i = 1:Nvars
    CurrentPosition(:,i) = random('unif',LB(i),UB(i),N,1);
end
Velocity = W.*rand(N,Nvars) ; % Initial Velocity
%% Evalute Initial Position
CurrentFitness = zeros(N,1); % Fitness Value
for i = 1:N
    CurrentFitness(i) = ObjectiveFunction(CurrentPosition(i,:));
end
%% Update Local Best
LocalBestPosition = CurrentPosition; % Local Best
LocalBestFitness = CurrentFitness;
%% Update Global Best
[GlobalBestFitness,index] = min(LocalBestFitness);
GlobalBestPosition = repmat(LocalBestPosition(index,:),N,1); % Global Best
%% Update Velocity and Position
R1 = rand(N,Nvars); % random Number 1
R2 = rand(N,Nvars); % random Number 2
Velocity = W.*Velocity + C1.*(R1.*(LocalBestPosition-CurrentPosition))...
    + C2.*(R2.*(GlobalBestPosition-CurrentPosition));
CurrentPosition = CurrentPosition + Velocity ;
%% Bound Data LB UB
for i = 1:Nvars
    indexes = CurrentPosition(:,i) < LB(i).*ones(N,1);
    CurrentPosition(indexes,i) = LB(i);
    indexes = CurrentPosition(:,i) > UB(i).*ones(N,1);
    CurrentPosition(indexes,i) = UB(i);
end
%% Iterate to Achive Best
Iter = 0;
while (Iter < MaxIterations)
    Iter = Iter+1;
    %%Evalute Current Position
    for i = 1:N
        CurrentFitness(i) = ObjectiveFunction(CurrentPosition(i,:));
    end
    %% Update Local Best
    indexes = find(CurrentFitness < LocalBestFitness);
    LocalBestFitness(indexes) = CurrentFitness(indexes);
    LocalBestPosition(indexes,:) = CurrentPosition(indexes,:);
    %% Update Global Best
    [GlobalBestFitnessNew,index] = min(LocalBestFitness);
    if GlobalBestFitnessNew < GlobalBestFitness
        GlobalBestFitness = GlobalBestFitnessNew;
        GlobalBestPosition = repmat(LocalBestPosition(index,:),N,1);
    end
    %% Update Velocity and Position
    R1 = randn(N,Nvars); % random Number 1
    R2 = randn(N,Nvars); % random Number 2
    Velocity = W.*Velocity + C1.*(R1.*(LocalBestPosition-CurrentPosition))...
        + C2.*(R2.*(GlobalBestPosition-CurrentPosition));
    CurrentPosition = CurrentPosition + Velocity ;
    %% Bound Data LB UB
    for i = 1:Nvars
        indexes = CurrentPosition(:,i) < LB(i).*ones(N,1);
        CurrentPosition(indexes,i) = LB(i);
        indexes = CurrentPosition(:,i) > UB(i).*ones(N,1);
        CurrentPosition(indexes,i) = UB(i);
    end
    %% Print Result
    M = mean(LocalBestFitness);
    if (Iter==1)||(rem(Iter,100)==0)
        fprintf('Iteration-%5.0d | Best F(x) = %d | Mean F(x) = %d\n',Iter,GlobalBestFitness,M);
    end
end
%% Output
X = GlobalBestPosition(1,:);
Fx = GlobalBestFitness;
end