
function im = centerImage(im)

[nx ny]=size(im);
xs=angle(exp(1i*linspace(-pi,pi,nx))*sum(im.^2)');
ys=angle(exp(1i*linspace(-pi,pi,ny))*sum(im.^2,2));

im = circshift(im,[round(ys*nx) round(xs*ny)]);

[xs ys]