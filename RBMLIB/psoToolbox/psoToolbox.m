function [] = psoToolbox
%% -Help psoToolbox V1.0
%   psoToolbox provides an interective GUI based Toolbox to solve
%   optimization problems using particle swarm optimization.
%   Creat a fitness function in M-file.
%   Inputs:
%     Function        : Function handle of fitness function.
%     Nvars           : Number of variable to be optimized.
%     LB              : Lower Bound of Nvars (1 X Nvars)
%     UB              : Upper Bound of Nvars (1 X Nvars)
%   Parameters:
%     C1              : Cognative Attraction
%     C2              : Social Attraction
%     W               : Inertial
%     Population Size : Number of Swarms
%     Max Iterations  : Maximum number of epochs.
%   Click on " RUN PSO " button to start PSO search. You will get the out put
%   at Edit box below the axes.
%   
%   To solve problem on command prompt use "pso.m".
%% Figure and Back ground
set(0,'units','pixel');
screensize = get(0,'screensize');
height = 600;
width = 800;
basecolor = [rand 1 rand]./2;
imdata = creatback(height,width,basecolor);
fig_position = [(screensize(3)/2)-(width/2) (screensize(4)/2)-(height/2) width height];
handle.fig = figure('Name','PSOToolbox',...
    'units','pixel',...
    'position',fig_position,...
    'numbertitle','off',...
    'menubar','none',...
    'resize','off');
handle.back = axes('parent',handle.fig,...
    'units','pixel',...
    'position',[0 0 width+2 height+2],...
    'xtick',[],...
    'ytick',[]);
imshow(imdata);
handle.titletext = text(20,20,'Particle Swarm Optimization Toolbox V1.0',...
    'fontsize',20,...
    'fontname','cambria',...
    'color','w');
%% Panels 1
handle.box1 = line([30 20 20 300 300 90],[80 80 300 300 80 80],...
    'linewidth',2,...
    'color','w');
handle.box1name = text(30,75,'Inputs',...
    'fontsize',16,...
    'fontname','cambria',...
    'color','w');
handle.text11 = text(35,110,'Function : ',...
    'fontsize',14,...
    'fontname','cambria',...
    'color','w');
handle.edit11 = uicontrol('style','edit',...
    'parent',handle.fig,...
    'position',[132 478 150 25],...
    'backgroundcolor',basecolor*(height-475)/height,...
    'fontsize',14,...
    'fontname','cambria',...
    'foregroundcolor','w',...
    'horizontalalignment','left');
handle.text12 = text(35,160,'Nvars       : ',...
    'fontsize',14,...
    'fontname','cambria',...
    'color','w');
handle.edit12 = uicontrol('style','edit',...
    'parent',handle.fig,...
    'position',[132 428 150 25],...
    'backgroundcolor',basecolor*(height-428)/height,...
    'fontsize',14,...
    'fontname','cambria',...
    'foregroundcolor','w',...
    'horizontalalignment','left');
handle.text13 = text(35,210,'Lower Bound : ',...
    'fontsize',14,...
    'fontname','cambria',...
    'color','w');
handle.edit13 = uicontrol('style','edit',...
    'parent',handle.fig,...
    'position',[162 378 120 25],...
    'backgroundcolor',basecolor*(height-378)/height,...
    'fontsize',14,...
    'fontname','cambria',...
    'foregroundcolor','w',...
    'horizontalalignment','left');
handle.text14 = text(35,260,'Upper Bound : ',...
    'fontsize',14,...
    'fontname','cambria',...
    'color','w');
handle.edit14 = uicontrol('style','edit',...
    'parent',handle.fig,...
    'position',[162 328 120 25],...
    'backgroundcolor',basecolor*(height-328)/height,...
    'fontsize',14,...
    'fontname','cambria',...
    'foregroundcolor','w',...
    'horizontalalignment','left');
%% Panel 2
handle.box2 = line([30 20 20 300 300 135],[335 335 555 555 335 335],...
    'linewidth',2,...
    'color','w');
handle.box2name = text(30,330,'Parameters',...
    'fontsize',16,...
    'fontname','cambria',...
    'color','w');
handle.text21 = text(35,365,'C1 : ',...
    'fontsize',14,...
    'fontname','cambria',...
    'color','w');
handle.edit21 = uicontrol('style','edit',...
    'parent',handle.fig,...
    'position',[72 223 70 25],...
    'backgroundcolor',basecolor*(height-223)/height,...
    'fontsize',14,...
    'fontname','cambria',...
    'foregroundcolor','w',...
    'horizontalalignment','left');
handle.text22 = text(150,365,'C2 : ',...
    'fontsize',14,...
    'fontname','cambria',...
    'color','w');
handle.edit22 = uicontrol('style','edit',...
    'parent',handle.fig,...
    'position',[186 223 90 25],...
    'backgroundcolor',basecolor*(height-223)/height,...
    'fontsize',14,...
    'fontname','cambria',...
    'foregroundcolor','w',...
    'horizontalalignment','left');
handle.text23 = text(35,415,'W  : ',...
    'fontsize',14,...
    'fontname','cambria',...
    'color','w');
handle.edit23 = uicontrol('style','edit',...
    'parent',handle.fig,...
    'position',[72 173 90 25],...
    'backgroundcolor',basecolor*(height-173)/height,...
    'fontsize',14,...
    'fontname','cambria',...
    'foregroundcolor','w',...
    'horizontalalignment','left');
handle.text24 = text(35,465,'Population Size : ',...
    'fontsize',14,...
    'fontname','cambria',...
    'color','w');
handle.edit24 = uicontrol('style','edit',...
    'parent',handle.fig,...
    'position',[186 123 90 25],...
    'backgroundcolor',basecolor*(height-123)/height,...
    'fontsize',14,...
    'fontname','cambria',...
    'foregroundcolor','w',...
    'horizontalalignment','left');
handle.text25 = text(35,515,'Max Iterations   : ',...
    'fontsize',14,...
    'fontname','cambria',...
    'color','w');
handle.edit25 = uicontrol('style','edit',...
    'parent',handle.fig,...
    'position',[186 73 90 25],...
    'backgroundcolor',basecolor*(height-73)/height,...
    'fontsize',14,...
    'fontname','cambria',...
    'foregroundcolor','w',...
    'horizontalalignment','left');
%% Convergence Axes 
handle.axes1 = axes('parent',handle.fig,...
    'units','pixel',...
    'position',[380 300 380 220],...
    'gridlinestyle','--',...
    'fontsize',11,...
    'fontname','cambria',...
    'color',basecolor*(height-380)/height,...
    'xcolor','w',...
    'ycolor','w');
xlabel('Iteration');
ylabel('F(x)');
h=title('Convergence of F(x)');
set(h,'color','w');
grid on
handle.edit3 = uicontrol('style','edit',...
    'units','pixel',...
    'position',[380 40 380 200],...
    'horizontalalignment','left',...
    'fontsize',14,...
    'fontname','cambria',...
    'foregroundcolor','w',...
    'backgroundcolor',basecolor*(height-200)/height,...
    'max',10,...
    'string','PSO :',...
    'Enable','on');
%% Button
F = getframe(handle.back,[70 10 160 30]);
handle.push = uicontrol('style','pushbutton',...
    'position',[70 10 160 30],...
    'cdata',F.cdata,...
    'string','RUN PSO',...
    'fontsize',14,...
    'fontname','cambria',...
    'foregroundcolor','w',...
    'callback',@push1_callback);

%% Background Image
    function cdata = creatback(height,width,basecolor)
        cdata = zeros(height,width,3);
        [height , ~, page] = size(cdata);
        for i = 1:height
            color = basecolor.*(i/height);
            for j = 1:page
                cdata(i,:,j) = color(j);
            end
        end
    end
%% Default
set(handle.edit21,'string','1.2');%C1
set(handle.edit22,'string','0.012');%C2
set(handle.edit23,'string','0.0004');%W
set(handle.edit24,'string','50');%N
set(handle.edit25,'string','10000');%Maxiter
set(handle.edit11,'string','@findsqrt');%function
set(handle.edit12,'string','3');
set(handle.edit13,'string','[0 0 0]');
set(handle.edit14,'string','[10 10 10]');

    function [] = push1_callback(varargin)
        axes(handle.axes1);cla
        set(handle.edit3,'string','PSO');
        
        C1=str2double(get(handle.edit21,'string'));%C1
        C2=str2double(get(handle.edit22,'string'));%C2
        W=str2double(get(handle.edit23,'string'));%W
        N=str2double(get(handle.edit24,'string'));%N
        MaxIterations=str2double(get(handle.edit25,'string'));%Maxiter
        ObjectiveFunction = get(handle.edit11,'string');%objectiev Function
        Nvars = str2double(get(handle.edit12,'string'));% nvars
        LB = str2num(get(handle.edit13,'string'));% LB
        UB = str2num(get(handle.edit14,'string'));%UB
        if isempty(ObjectiveFunction)
            errordlg('PS0 : Define Objective Function');
        else
            ObjectiveFunction=str2func(ObjectiveFunction);
        end
        if ~isequal(Nvars,size(LB,2),size(UB,2))
            errordlg('PSO : Invalid Arguments : isequal(Nvars,size(LB,2),size(UB,2)) should be true ');
        end 
        str{1} = get(handle.edit3,'string');
        str{2} = sprintf('\nRunning Optimization....');
        set(handle.edit3,'string',str);
        % PSO
        CurrentPosition = zeros(N,Nvars); % Initial Position
        for i = 1:Nvars
            CurrentPosition(:,i) = random('unif',LB(i),UB(i),N,1);
        end
        Velocity = W.*rand(N,Nvars) ; % Initial Velocity
        % Evalute Initial Position
        CurrentFitness = zeros(N,1); % Fitness Value
        for i = 1:N
            CurrentFitness(i) = ObjectiveFunction(CurrentPosition(i,:));
        end
        % Update Local Best
        LocalBestPosition = CurrentPosition; % Local Best
        LocalBestFitness = CurrentFitness;
        % Update Global Best
        [GlobalBestFitness,index] = min(LocalBestFitness);
        GlobalBestPosition = repmat(LocalBestPosition(index,:),N,1); % Global Best
        % Update Velocity and Position
        R1 = rand(N,Nvars); % random Number 1
        R2 = rand(N,Nvars); % random Number 2
        Velocity = W.*Velocity + C1.*(R1.*(LocalBestPosition-CurrentPosition))...
            + C2.*(R2.*(GlobalBestPosition-CurrentPosition));
        CurrentPosition = CurrentPosition + Velocity ;
        % Bound Data LB UB
        for i = 1:Nvars
            indexes = CurrentPosition(:,i) < LB(i).*ones(N,1);
            CurrentPosition(indexes,i) = LB(i);
            indexes = CurrentPosition(:,i) > UB(i).*ones(N,1);
            CurrentPosition(indexes,i) = UB(i);
        end
        % Iterate to Achive Best
        Iter = 0;
        while (Iter < MaxIterations)
            Iter = Iter+1;
            %Evalute Current Position
            for i = 1:N
                CurrentFitness(i) = ObjectiveFunction(CurrentPosition(i,:));
            end
            % Update Local Best
            indexes = find(CurrentFitness < LocalBestFitness);
            LocalBestFitness(indexes) = CurrentFitness(indexes);
            LocalBestPosition(indexes,:) = CurrentPosition(indexes,:);
            % Update Global Best
            [GlobalBestFitnessNew,index] = min(LocalBestFitness);
            if GlobalBestFitnessNew < GlobalBestFitness
                GlobalBestFitness = GlobalBestFitnessNew;
                GlobalBestPosition = repmat(LocalBestPosition(index,:),N,1);
            end
            % Update Velocity and Position
            R1 = randn(N,Nvars); % random Number 1
            R2 = randn(N,Nvars); % random Number 2
            Velocity = W.*Velocity + C1.*(R1.*(LocalBestPosition-CurrentPosition))...
                + C2.*(R2.*(GlobalBestPosition-CurrentPosition));
            CurrentPosition = CurrentPosition + Velocity ;
            % Bound Data LB UB
            for i = 1:Nvars
                indexes = CurrentPosition(:,i) < LB(i).*ones(N,1);
                CurrentPosition(indexes,i) = LB(i);
                indexes = CurrentPosition(:,i) > UB(i).*ones(N,1);
                CurrentPosition(indexes,i) = UB(i);
            end
            if (Iter==1)||(rem(Iter,100)==0)
                plot(Iter,GlobalBestFitness,'r.','Linewidth',3);hold on
                set(handle.axes1,'xcolor','w','ycolor','w',...
                    'fontsize',11,...
                    'fontname','cambria',...
                    'color',basecolor*(height-380)/height);
                xlabel('Iteration');
                ylabel('F(x)');
                h = title(['Best Value : ' num2str(GlobalBestFitness)]);
                set(h,'color','w');
                grid on
                drawnow
            end
        end
        str{3}= sprintf('\nOptimization Complete.');
        set(handle.edit3,'string',str);
        str{4} = sprintf('\nF(x) = %d',GlobalBestFitness);
        str{5} = sprintf('\nx = %d',GlobalBestPosition(1,:));
        set(handle.edit3,'string',str);
        clear('str');
    end
end