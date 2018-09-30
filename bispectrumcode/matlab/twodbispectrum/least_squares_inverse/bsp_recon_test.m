
nx = 28;
ny = 32;
nsamp = 1;

% % Circle image
% [x,y]=meshgrid(linspace(-1,1,N),linspace(-1,1,N));
% X = zeros(N,N);
% X((x.^2+y.^2)<0.25) =1;
% imagesc(X);
%
% % Calvin
% X = imread('calvin_12.png');
% X = rgb2gray(X(1:90,1:90,:));
% X = double(imresize(X,[N N]));

X = imread('spaceinv.png');
X = double((X(:,15:30)));
X = imresize(X,[nx ny]);
% X = padarray(X,4);
% X = padarray(X',4)';
[nx ny] = size(X);
X = whiten_contnorm(-X,0,0);
X=-X;

X0 = (X-mean(X(:)))/std(X(:));
X = repmat(X0,[1 1 nsamp]);
Xn = repmat(X0,[1 1 nsamp]);

if nsamp==1
    X = X+randn(nx,ny)/100; % for nsamp==1, make sure we have some power at all frequencies
else
    for i=1:nsamp
        X(:,:,i) = circshift(X(:,:,i),ceil(rand(1,2).*[nx ny])) + randn(nx,ny);
%         X(:,:,i) = circshift(X(:,:,i),ceil(rand(1,2)*11)-6) + randn(N,N);
        Xn(:,:,i) = Xn(:,:,i) + randn(nx,ny);
        tmp = X(:,:,i);
        X(:,:,i) = X(:,:,i)-mean(tmp(:));
    end
end

[imhat, Fhat, id, bid, Am, Ap, b, fpopt] = bsp_leastsquares_recon_v2(X);

%%
clf
subplot(2,4,1)
imagesc(X0)
axis image off;
title('True')
subplot(2,4,5)
imagesc(fftshift(abs(fft2(X0))))
axis image off;
title('True FFT')

subplot(2,4,2)
imagesc(X(:,:,1))
axis image off;
title('Sample')
subplot(2,4,6)
imagesc(fftshift(abs(fft2(X(:,:,1)))))
axis image off;
title('Sample FFT')

subplot(2,4,3)
imagesc(imhat)
axis image off;
title('Reconstruction')
subplot(2,4,7)
imagesc(fftshift(abs(Fhat)))
axis image off;
title('FFT Reconstruction')

subplot(2,4,4)
imagesc(mean(Xn,3))
axis image off;
title('No-Shift Avg')
subplot(2,4,8)
imagesc(fftshift(abs(fft2(mean(Xn,3)))))
axis image off;
title('FFT No-Shift Avg')