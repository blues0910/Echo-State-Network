function x = update_reservior_states(T,I,NetworkNodes,W_inputweights,W,x0,zt)
%RESERVIOR 此处显示有关此函数的摘要
%   此处显示详细说明
%Parameters--------------------------------------------------------
x=zeros(NetworkNodes,T);%N*T
% x0=zeros(NetworkNodes,1);%N*1
%Mask--------------------------------------------------------
J=W_inputweights*I;%N*K*K*T=N*T
%Reservoir--------------------------------------------------------
for t=1:T
    if t==1
        x(:,t)=Internal_Sigmoid_Func(J(:,t)+W*x0+zt(:,t));
    else
        x(:,t)=Internal_Sigmoid_Func(J(:,t)+W*x(:,t-1)+zt(:,t));
    end    
end
end

