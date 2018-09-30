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

function [imhat, Fhat, id, bid, Am, Ap, b, fpopt] = bsp_leastsquares_recon_fromFullBSP(B)

[nx,ny,tmp,tmp]=size(B);
nsamp=1;

fprintf('Setting up systems of equations...')
tic
% Non-redundant set of Fourier coefficients to use...
id = bsp_getFid([nx ny]);
% Non-redundant set of Bispectrum coefficients (and linear systems Am and Ap for the Fourier magnitude and phase)...
[bid,Am,Ap] = bsp_getBid(id,[nx ny]);
% Compute bispectra...
% f = reshape(fft2(X),nx*ny,[]);
% b = f(bid(:,1),:).*f(bid(:,2),:).*conj(f(bid(:,3),:));
[k1x,k1y] = ind2sub([nx ny],bid(:,1));
[k2x,k2y] = ind2sub([nx ny],bid(:,2));
bbid = sub2ind([nx ny nx ny],k1x,k1y,k2x,k2y);
b = B(bbid);
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

% % least-squares on the mean bispectrum...
if nsamp>2
    bmu = sum(b,2)/nsamp;
else
    bmu=b;
end
xhat = Am(:,midx)\log(abs(bmu));

% % weighted least-squares...
% W = sparse(diag(1./std(log(abs(b)),1,2).^2));
% xhat = (Am(:,midx)'*W*Am(:,midx))\Am(:,midx)'*W*mean(log(abs(b)),2);

mhat = zeros(nx,ny)*NaN;
mhat(id(midx)) = exp(xhat); % estimated magnitude
mhat(1) = 0;
toc

% Least squares only defines phase up to a multiple of 2*pi...
% I'm using phase unwrapping optimization instead...
fprintf('Phase unwrapping optimization...')
tic
pidx_rem = sub2ind([nx ny],[1; 1; nx/2+1],[1; ny/2+1; 1]); % exclude F(0,0), F(0,N/2), F(N/2,0)
pidx=id*0+1;
for j=1:length(pidx_rem)
    pidx(id==pidx_rem(j))=0;
end
pidx = logical(pidx);
ab = angle(sum(b,2));    % circular mean
r = abs(sum(exp(1i*angle(b)),2))/nsamp;
% w = sqrt(-2*log(r));                    % circular std
w = sqrt(2*(1-r));                        % angular deviation

% keyboard
% w = sqrt(var(imag(bsxfun(@times,b,-sum(b,2)/nsamp)),1,2)./abs(sum(b,2)/nsamp)/nsamp);

if nsamp==1, w=w*0+1; end

% phat0 = rand(sum(pidx),1)*6*pi-3*pi;
% phat0 = Ap(:,pidx)\ab+(ceil(rand(sum(pidx),1)*3)-2)*pi;
phat0 = Ap(:,pidx)\ab;
% phat0 = angle(f(id(pidx),1));

% % Check gradient...
% keyboard
% global bphi
% global A
% global w
% bphi = ab;
% A = Ap(:,pidx);
% [grad,err,finaldelta] = gradest(@objBspPhaseLoss,phat0);
% [f,df] = objBspPhaseLoss(phat0);

% options = optimset('GradObj','on');
% phat = fminunc(@objBspPhaseLoss,phat0,options,ab,Ap(:,pidx),w);

% keyboard

% Random restarts...
fpopt=[];
rrnum = 100;
rriter = 50;
phat=zeros(length(phat0),rriter);
fprintf('Random restarts...')
for i=1:rrnum
    fprintf('%03i/%03i...\n',i,rrnum)
%     phat0 = Ap(:,pidx)\ab+(ceil(rand(sum(pidx),1)*3)-2)*pi;
    phat0 = rand(sum(pidx),1)*6*pi-3*pi;
    [phat(:,i),fpopt(:,i)] = minimize_verbose1(phat0, 'objBspPhaseLoss', rriter, ab, Ap(:,pidx), w);
end
[tmp,i]=min(fpopt(end,:));
phat=phat(:,i);
% make sure it converges...
phat = minimize_verbose1(phat, 'objBspPhaseLoss', 1000, ab, Ap(:,pidx), w);

% phat = fminunc(@objBspPhaseLoss,phat0,options,angle(b(:)),repmat(Ap(:,pidx),nsamp,1),b(:)*0+1);
% phat = Ap\angle(b);
ahat = zeros(nx,ny)*NaN;
ahat(id(pidx))=phat;
ahat(pidx_rem)=0;
toc

Fhat = mhat.*(cos(ahat)+1i*sin(ahat));
Fhat = fconjsym(Fhat); % Conjugate symmetry to fill in the rest of the fft2...

% Estimated image...
imhat = real(ifft2(Fhat));