clear
profile on
%Parameters--------------------------------------------------------
NetworkNodes=20;
% OutputNodes=10;
%Training Reservoir----------------------------------------------------------------------------------------
%training data------------------------------------------------------
training_data=load([pwd '\生成教师数据\sample_STM_delay5.mat']);
I=training_data.u(1:100);
TY=training_data.y(1:100);
%Input Weights&&Internal Connections--------------------------------------------------------
W_inputweights=-1+(1-(-1)).*rand(NetworkNodes,size(I,1));
W=randsrc(NetworkNodes,NetworkNodes,[0 0.1 -0.1;0.9875 0.00625 0.00625]);
%Reservoir------------------------------------------------------``--
x_network = Reservior(I,NetworkNodes,W_inputweights,W);
%Training Output Weights---------------------------------------------------
w_output=TrainFunc(x_network,TY,NetworkNodes);
y_trained=w_output*x_network;
disp(NRMSE(TY,y_trained));
subplot(311)
plot(TY);
subplot(312)
plot(y_trained)
subplot(313)
plot(abs(y_trained-TY));
% save([pwd '\计算结果\MaskFunc_and_OutputWeights_test_100nodes'],'M','w_output');
% save([pwd '\计算结果\Training_Result_of_test_100nodes.mat']);