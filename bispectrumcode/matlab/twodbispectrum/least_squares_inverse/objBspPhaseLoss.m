
function [f,df] = objBspPhaseLoss(fphi,bphi,A,w)

% For checking gradient
% function [f,df] = objBspPhaseLoss(fphi)
% global bphi
% global A
% global w

f  = sum(((mod(bphi-A*fphi+pi,2*pi)-pi)./w).^2)/2;
df = -A'*(((mod(bphi-A*fphi+pi,2*pi)-pi)./w)./w);