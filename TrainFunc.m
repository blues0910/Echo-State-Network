function w_output = TrainFunc(x_network,y_teach,NetworkNodes)
%LNEARREGRESS �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%Moore-Penrose pseudo-inverse------------------------------------------------------------------
x_MPP=pinv(x_network);
w_output=atanh(y_teach)*x_MPP;
%Tikhonov regularisation or ridge regression------------------------------------------------------------------
% w_output=y_teach*(pinv(x_network'*x_network+0.1*diag(var(x_network)))*x_network');
end