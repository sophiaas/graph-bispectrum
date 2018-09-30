% 2d examples of bispectrum

% tiny example
f = rand(2,3)
B = TwoDBispectrum(f,2,3);
freconstruct = real(invTwoDBispectrum(B,2,3))

% small image example
colormap(gray);
X = spiral(8);
image(X);
Bsp = TwoDBispectrum(X,8,8);
X2 = real(invTwoDBispectrum(Bsp,8,8));
image(X2);
colormap(gray);

% small line image example
I = imread('line.bmp');
image(I);
Bsp = TwoDBispectrum(double(I),32,32);
I2 = real(invTwoDBispectrum(Bsp,32,32));
image(I2);

