clear
%Parameter setting-------------------------------------------------
Number_of_neurons=200;
Training_steps=10000;
Predicting_steps=1000;
T=Training_steps+Predicting_steps;
discarded_steps=1000;
%training data------------------------------------------------------
training_data=load([pwd '\Generating_training_data\Sample_of_Lorenz_system.mat']);
Input_streaming=training_data.u(:,1:T);
%Generating state noise------------------------------------------------------
epsilon=0.0001;
noise=normrnd(0,epsilon,[Number_of_neurons,T]);
%Input Weights&&Internal Connections--------------------------------
Interconnetions=randsrc(Number_of_neurons,Number_of_neurons,[0 0.1 -0.1;0.9875 0.00625 0.00625]);
a=0.1;
Input_mask=-a+(a-(-a)).*rand(Number_of_neurons,size(Input_streaming,1));
%update_reservior_states------------------------------------------------------
x=zeros(Number_of_neurons,T);
x(:,1:Training_steps) = update_reservior_states(Training_steps,Input_streaming,Number_of_neurons,Input_mask,Interconnetions,rand(Number_of_neurons,1),noise);
%on line training---------------------------------------------------
W_output=OnLine_training(x(:,discarded_steps+1:T),training_data.y(:,discarded_steps+1:T),0.0001);
%Training Output Weights---------------------------------------------------
% W_output=Batch_training(x(:,discarded_steps+1:T),training_data.y(:,discarded_steps+1:T),0.01);
%Prediction--------------------------------------------------------------------------------------------
W_output=W_output';
for t=Training_steps+1:T
    Ip=Output_Func(W_output*x(:,t-1));
    x(:,t) = update_reservior_states(1,Ip,Number_of_neurons,Input_mask,Interconnetions,x(:,t-1),noise(:,t));
end
y_trained=Output_Func(W_output*x);

%Plot results--------------------------------------------------------------------------------------------
tmpT=discarded_steps;
tstep=training_data.theta;
figure('name','1')
tl=linspace(0+(tmpT-1)*tstep,tstep*T,size(y_trained,2)-tmpT+1);
for i=1:size(training_data.y,1)
    s=[num2str(size(training_data.y,1)) '1' num2str(i)];
    subplot(s)
    hold on
    plot(tl(1:Training_steps-tmpT+1),y_trained(i,tmpT:Training_steps),'m')
    plot(tl(Training_steps-tmpT+2:end),y_trained(i,Training_steps+1:end),'r')
    plot(tl,training_data.y(i,tmpT:T),'b')
    plot(linspace((Training_steps)*tstep,(Training_steps)*tstep,100),linspace(min(y_trained(i,tmpT:end)),max(y_trained(i,tmpT:end)),100),'k')
    xlabel('$t$','Interpreter','latex');
    ylabel('Output','Interpreter','latex');
    legend({'Reservoir output','Predictive output','Actual'},'Interpreter','latex')
%     text(Tl*tstep,0,'\rightarrow Prediction')
end
figure(2)
xt=return_map(training_data.y(3,discarded_steps+1:T));
xp=return_map(y_trained(3,discarded_steps+1:end));
plot(xt(1,:),xt(2,:),'.');
hold on
plot(xp(1,:),xp(2,:),'.');
legend({'Actual','Reservoir output'},'Interpreter','latex')
