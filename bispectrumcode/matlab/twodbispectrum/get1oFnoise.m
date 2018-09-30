
% Whiten and contrast normalize an image...
function imw = get1oFnoise(n,b)

if nargin<2, b=1; end
if numel(n)==1
    im = randn(n,n);
elseif numel(n)==2
    im = randn(n(1),n(2));
else
    im=n;
end

imf=fftshift(fft2(im));
impf=abs(imf).^2;

N = min(size(im));
fxvec=-size(im,1)/2:size(im,1)/2-1;
fyvec=-size(im,2)/2:size(im,2)/2-1;

% 1/F^b
[fx fy] = meshgrid(fyvec,fxvec);
[theta rho]=cart2pol(fx,fy);
rho(rho==0)=1;
filtf = 1./rho.^b;

imwf = filtf.*imf;
imwpf = abs(imwf).^2;
imw = ifft2(fftshift(imwf));
