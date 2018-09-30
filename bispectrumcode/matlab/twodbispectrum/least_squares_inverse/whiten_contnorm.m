
% Whiten and contrast normalize an image...
function imn = whiten_contnorm(im,doCN,doPlot)

if nargin<2, doCN=true; end
if nargin<3, doPlot=false; end

% Convert to frequency domain...
imf=fftshift(fft2(im));
impf=abs(imf).^2;

N = min(size(im));
fxvec=-size(im,1)/2:size(im,1)/2-1;
fyvec=-size(im,2)/2:size(im,2)/2-1;

% Whitening Filter
[fx fy] = meshgrid(fyvec,fxvec);
[theta rho]=cart2pol(fx,fy);
filtf = rho.*exp(-0.5*(rho/(0.7*N/2)).^2);
% filtf = rho.*exp(-0.5*(rho/(10*N/2)).^2);
% filtf(rho==1)=0;
% keyboard

imwf = filtf.*imf;
imwpf = abs(imwf).^2;
imw = ifft2(fftshift(imwf));

if doCN
    % Contrast normalize
    D=16;
    [x y] = meshgrid(-D/2:D/2-1);
    G = exp(-0.5*((x.^2+y.^2)/(D/2)^2));
    G = G/sum(G(:));
    imv = conv2(imw.^2,G,'same');

    imn = imw./sqrt(imv);
    imnf=fftshift(fft2(imn));
    imnpf=abs(imnf).^2;
    imn = real(imn);
else
    imnpf=imwpf;
    imn = real(imw);
end

if doPlot
    subplot(3,2,1)
    imagesc(im), axis image off
    title('Original')
    colormap gray
    subplot(3,2,2)
    imagesc(fxvec,fyvec,log10(impf)), axis xy
    subplot(3,2,3)
    imagesc(real(imw)), axis image off
    title('Whitened')
    subplot(3,2,4)
    imagesc(fxvec,fyvec,log10(imwpf)), axis xy
    if doCN
        subplot(3,2,5)
        imagesc(real(imn)), axis image off
        title('Whitened+ContrastNorm')
        subplot(3,2,6)
        imagesc(fxvec,fyvec,log10(imnpf)), axis xy
    end
end