% Reconstruct an image from a set of bispectra...
% Given...
%   X       [pixels x pixels x samples] set of images
% Return...
%   imhat   [N x N] reconstruction (up to a global phase shift)
%   Fhat    [N x N] reconstruction of the fft2 (filled in by conjugate symmetry)
%   id      (non-redundant set of) indices of the fourier transforms in this set of equations
%   bid     (non-redundant set of) indices of the bispectra in this set of equations [k1 k2 k1+k2]
%   Am      a system of linear equations Am*log(abs(F(id)))=log(abs(B(bid)))
%   Ap      a system of linear equations Ap*angle(F(id))=angle(B(bid))
%   b       bispectra
%
% NOTE: I think Sadler & Giannakis has a couple errors...
% to get the right rank I modified...
%    (25) x_0   = ln(abs(F(0,0)))... ln(abs(F(0,N/2))
%    (25) x_N/2 = ln(abs(F(N/2,0)))... ln(abs(F(N/2,N/2))
%    (26) k22 = 0,...,N-1-k12

function [imhat, Fhat, id, bid, Am, Ap, b] = bsp_leastsquares_recon(X)

[N,N,nsamp]=size(X);

fprintf('Setting up systems of equations...')
tic
% Non-redundant set of Fourier coefficients to use...
id = bsp_getFid(N);
% Non-redundant set of Bispectrum coefficients (and linear systems Am and Ap for the Fourier magnitude and phase)...
[bid,Am,Ap] = bsp_getBid(id,N);
% Compute bispectra...
f = reshape(fft2(X),N*N,[]);
b = f(bid(:,1),:).*f(bid(:,2),:).*conj(f(bid(:,3),:));
toc

% % Check that equations are unique
% hmax = max(bid(:));
% length(unique(bid(:,1)*hmax+bid(:,2)))

% % Check that equations are correct
% b = f(bid(:,1)).*f(bid(:,2)).*conj(f(bid(:,3)));
% bhat = Am*log(abs(f(id)'));

% Least-squares for the amplitudes...
fprintf('Least-squares for amplitudes...')
tic
midx = [2:size(Am,2)];  % exclude F(0,0)
xhat = Am(:,midx)\log(mean(abs(b),2));
mhat = zeros(N,N)*NaN;
mhat(id(midx)) = exp(xhat); % estimated magnitude
mhat(1) = 0;
toc

% Least squares only defines phase up to a multiple of 2*pi...
% I'm using phase unwrapping optimization instead...
fprintf('Phase unwrapping optimization...')
tic
pidx_rem = sub2ind([N N],[1; 1; N/2+1],[1; N/2+1; 1]); % exclude F(0,0), F(0,N/2), F(N/2,0)
pidx=id*0+1;
for j=1:length(pidx_rem)
    pidx(id==pidx_rem(j))=0;
end
pidx = logical(pidx);
phat0 = rand(sum(pidx),1)*6*pi-3*pi;
ab = angle(sum(exp(1i*angle(b)),2));    % circular mean
r = abs(sum(exp(1i*angle(b)),2))/nsamp;
% w = sqrt(-2*log(r));                    % circular std
w = sqrt(2*(1-r));                        % angular deviation
if nsamp==1, w=w*0+1; end

options = optimset('GradObj','on');
phat = fminunc(@objBspPhaseLoss,phat0,options,ab,Ap(:,pidx),w);
% phat = Ap\angle(b);
ahat = zeros(N,N)*NaN;
ahat(id(pidx))=phat;
ahat(pidx_rem)=0;
toc

Fhat = mhat.*(cos(ahat)+1i*sin(ahat));
Fhat = fconjsym(Fhat); % Conjugate symmetry to fill in the rest of the fft2...

% Estimated image...
imhat = real(ifft2(Fhat));