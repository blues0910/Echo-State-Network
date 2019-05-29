function x = Reservior(I,NetworkNodes,W_inputweights,W)
%RESERVIOR �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%Parameters--------------------------------------------------------
T=size(I,2);
x=zeros(NetworkNodes,T);%N*T
x0=zeros(NetworkNodes,1);%N*1
%Mask--------------------------------------------------------
J=W_inputweights*I;%N*K*K*T=N*T
%Reservoir--------------------------------------------------------
for t=1:T
    if t==1
        x(:,t)=Internal_Sigmoid_Func(J(:,t)+W*x0);
    else
        x(:,t)=Internal_Sigmoid_Func(J(:,t)+W*x(:,t-1));
    end    
%     disp(t) %test command
end
end

