clear
profile on
%Parameters--------------------------------------------------------
NetworkNodes=20;
discarded_steps=500;
OutputNodes=40;
%Training Reservoir----------------------------------------------------------------------------------------
%training data------------------------------------------------------
training_data=load([pwd '\Generating_Teacher_Data\sample_STM_delay[1_40]_1.mat']);
I=training_data.u;
%Input Weights&&Internal Connections--------------------------------------------------------
W_inputweights=-0.5+(0.5-(-0.5)).*rand(NetworkNodes,size(I,1));
W=randsrc(NetworkNodes,NetworkNodes,[0 0.47 -0.47;0.8 0.1 0.1]);
% W=randsrc(NetworkNodes,NetworkNodes,[0 0.1 -0.1;0.9875 0.00625 0.00625]);

s=['The spectral radius of Internal Connections Matrix is ' num2str(max(eig(W)))];%test code
disp(s)%test code
%Reservoir------------------------------------------------------``--
x_network = Reservior(I,NetworkNodes,W_inputweights,W);
%Training Output Weights---------------------------------------------------
W_output=cell(1,OutputNodes);
TY=cell(1,OutputNodes);
y_trained=cell(1,OutputNodes);
nrmse=zeros(1,OutputNodes);
STM_Coef=zeros(1,OutputNodes);
for i=1:OutputNodes
    TY{i}=training_data.y{i};
    W_output{i}=TrainFunc(x_network(:,discarded_steps:end),TY{i}(discarded_steps:end),NetworkNodes);
    y_trained{i}=Output_Func(W_output{i}*x_network);
    nrmse(i)=NRMSE(TY{i},y_trained{i});
    STM_Coef(i)=Determination_Coefficient(I,TY{i},y_trained{i});
end
%Plots--------------------------------------s-----------------------------------------------------------
S=zeros(1,4);
S(1)=1;
S(2)=find(STM_Coef>=0.5,1,'last');
S(3)=find(STM_Coef>=0.25,1,'last');
S(4)=find(STM_Coef>=0.1,1,'last');
figure('Name',['Testing the trained delay network for delays ' num2str(S)])
for i=1:length(S)
    tmp=length(S)*100+10+i;
    subplot(tmp)
    plot(TY{S(i)},'r');
    hold on
    plot(y_trained{S(i)},'b');
    legend({'$y$','$\hat{y}$'},'Interpreter','Latex')
    title(['Delay=' num2str(training_data.tau(S(i))) ...
        ', NRMSE=' num2str(nrmse(S(i))) ...
        ', The ' num2str(S(i)) '-delay STM capacity=' num2str(STM_Coef(S(i)))])
end
save([pwd '\Results\MaskFunc_and_OutputWeights_test_100nodes'],'W_inputweights','W','W_output');
save([pwd '\Results\Training_Result_of_test_100nodes.mat']);
figure('Name','The forgetting curve of the trained network')
plot(STM_Coef)

ss=['The STM capacity of the network is ' num2str(sum(STM_Coef))];%test code
disp(ss)%test code
%--------------------------------------------------------------------------------------------
%Test Reservoir--------------------------------------------------------------------------------------------
%test prediction---------------------------------------------------------------
% %Load MaskFunc and OutputWeights-------------------------------------------
% %Reservoir--------------------------------------------------------
% %Load output weights-------------------------------------------------------
profile viewer
