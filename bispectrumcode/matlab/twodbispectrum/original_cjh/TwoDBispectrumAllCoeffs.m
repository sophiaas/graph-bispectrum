% TwoDBispectrumAllCoeffs.m - compute the full 2D bispectrum of 
%                    f: {0..m-1} x {0..n-1} = Z/mZ x Z/nZ -> C
%
% function B = TwoDBispectrumAllCoeffs(f,m,n)
%
% m,n:      positive integers
% f:        complex matrix of size m x n
% 
% Computes an m x n x m x n matrix B = [B(i,j,k,l)]
%                        i = 0..m-1  j = 0..n-1  k = 1..m-1  l = 1..n-1
% in which B(k1,k2,k3,k4) = f^(k1,k2)* f^(k3,k4)* f^(k1+k3,k2+k4)
%
% Here f^ is the 2D Fourier Transform of f and * means complex conjugation
% (starting at 0 not 1 like matlab)
%
% One can now flatten and unflatten this matrix into a vector as:
%	reshape(B,m^2*n^2,1)
%
% Christopher Hillar 2008
%
% Example:   
%	  f = rand(2,3);
%     B = TwoDBispectrumAllCoeffs(f,2,3);
%	  reshape(B,36,1)
%     TwoDBispectrum(f,2,3)'
%

function B = TwoDBispectrumAllCoeffs(f,m,n)

B = zeros(m,n,m,n);
F = fft2(f);
for i=1:m
	for j=1:n
		for k=1:m
			for l=1:n
				sumik = mod(i + k - 2,m)+1;
				sumjl = mod(j + l - 2,n)+1;
				B(i,j,k,l) = conj(F(i,j)*F(k,l))*F(sumik,sumjl);
			end
		end
	end
end
    
    

    
    
    