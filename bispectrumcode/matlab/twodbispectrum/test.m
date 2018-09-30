
%% Testing 2D Bispectrum

X=imread('street1.jpg');
X=double(rgb2gray(X));
X = X-mean(X(:));
X = X/std(X(:));

% randomized phase image...
F=fft2(X);
r=sqrt(real(F).^2+imag(F).^2);
p=rand(size(r))*2*pi;
Xr = real(ifft2(r.*cos(p)+1i.*sin(p)));

%%
clf
nF=6;  % number of frequencies to compute
cP=4;   % bispectra plotting range

B = TwoDBispectrum(X,nF,nF);
subplot(2,3,1)
imagesc(X)
axis image; axis off
subplot(2,3,2)
% F=fft2(X);
imagesc(-log(fftshift(sqrt(F.*conj(F)))))
axis image; axis off
title('2D-Fourier Amplitude')
subplot(2,3,3)
plotKZB(B,cP)
title('Bispectrum Amplitude')

B = TwoDBispectrum(Xr,nF,nF);
subplot(2,3,4)
imagesc(Xr)
axis image; axis off
subplot(2,3,5)
F=fft2(Xr);
imagesc(-log(fftshift(sqrt(F.*conj(F)))))
axis image; axis off
title('2D-Fourier Amplitude')
subplot(2,3,6)
plotKZB(B,cP)
title('Bispectrum Amplitude')

colormap gray
