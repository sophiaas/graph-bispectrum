% invTwoDBispectrum.m - compute inverse of bispectrum of f: Z/mZ x Z/nZ -> C
%
% function f = invTwoDBispectrum(B,m,n)
%
% m,n:        positive integers
% B:        complex vector of length mn+2
% 
% Computes a m x n matrix f whose truncated bispectrum is B
% The inversion is only up to a 2D Translation
%
% Christopher Hillar 2008

function f = invTwoDBispectrum(B,m,n)

% first find Fourier transform of f
F = zeros(m,n);

% Create 1D vectors to recover f^(i,0)  i = 1..m-1
%                              f^(0,j)  j = 1..n-1     
B1dcol = B(1:m+1);
F1dcol = invOneDBispectrumFFTversion(B1dcol,m);
B1drow = [B(1),B(m+2:m+n+1)];
F1drow = invOneDBispectrumFFTversion(B1drow,n);

F(1,1:n) = F1drow;
F(1:m,1) = F1dcol;

% fill in the rest of the matrix for f

itercount = 1;
for j=1:n-1
for i=1:m-1
    F(i+1,j+1) = B(m+n+1+itercount)/conj(F(1,j+1)*F(i+1,1));
    itercount = itercount + 1;
end
end

% invert Fourier transform F to get back f
f = (ifft2(F));