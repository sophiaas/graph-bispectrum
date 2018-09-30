% TwoDBispectrum.m - compute a truncated 2D bispectrum of 
%                    f: {0..m-1} x {0..n-1} = Z/mZ x Z/nZ -> C
%
% function B = TwoDBispectrum(f,m,n)
%
% m,n:      positive integers
% f:        complex matrix of size m x n
% 
% Computes an mn+2 vector B = [B(0,0,0,0),B(i,0,1,0),B(0,j,0,1),B(0,j,i,0)]
%                                       i = 0..m-1  j = 0..n-1  j,i = 1..n-1,m-1
% in which B(k1,k2,k3,k4) = f^(k1,k2)* f^(k3,k4)* f^(k1+k3,k2+k4)
%
% Here f^ is the 2D Fourier Transform of f and * means complex conjugation
% (starting at 0 not 1 like matlab)
%
% Christopher Hillar 2008

function B = TwoDBispectrum(f,m,n)

B = zeros(1,m*n+2);
F = fft2(f);
B(1) = conj(F(1,1)*F(1,1))*F(1,1);
for i=1:m-1
    B(i+1) = conj(F(i,1)*F(2,1))*F(i+1,1);
end
    B(m+1) = conj(F(m,1)*F(2,1))*F(1,1);

for j=1:n-1
    B(m+1+j) = conj(F(1,j)*F(1,2))*F(1,j+1);
end
    B(m+1+n) = conj(F(1,n)*F(1,2))*F(1,1);

itercount = 1;
for j=1:n-1
for i=1:m-1
    B(m+1+n+itercount) = conj(F(1,j+1)*F(i+1,1))*F(i+1,j+1);
    itercount = itercount + 1;
end
end
    
    

    
    
    