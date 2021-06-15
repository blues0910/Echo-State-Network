function [xp] = return_map(x)
%RETURN_MAP 此处显示有关此函数的摘要
%   此处显示详细说明
j=1;
for i=2:length(x)-1
    if x(i)>=x(i-1) && x(i)>=x(i+1)
        dx(j)=x(i);
        j=j+1;
    end
end
xp(1,:)=dx(1:end-1);
xp(2,:)=dx(2:end);
end

