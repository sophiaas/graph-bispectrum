% TwoDBispectrum.m
% Inputs = f, m, n
% Outputs = B_red, B_full
% 
% Uses Chris Hillar's python code idea for returnign reduced features
% Get easy to visualize (highly redundant) bispectra, following
% Fig 2 from Krieger, Zetzsche, and Barth, 1997

function [B] = TwoDBispectrum(f,m,n)

if nargin <3 & nargin >=1
    sz = size(f);
    m = sz(1); n = sz(2);
end

B = zeros(2*m+1,2*n+1,2*m+1,2*n+1);
F = fft2(f);
[fn,fm]=size(F);
for i=-m:m
	for j=-n:n
		for k=-m:m
			for l=-n:n
                sumik = mod(i+k,fn)+1;
                sumjl = mod(j+l,fm)+1;
  				B(i+m+1,j+n+1,k+m+1,l+n+1) = F(mod(i,fn)+1,mod(j,fm)+1)*F(mod(k,fn)+1,mod(l,fm)+1)*conj(F(sumik,sumjl));
			end
		end
	end
end

%  Computes an mn+2 vector B = 
% [B(0,0,0,0),B(i,0,1,0),B(0,j,0,1),B(0,j,i,0)]
%               i = 0..m-1  j = 0..n-1  j,i = 1..n-1,m-1
%    in which B(k1,k2,k3,k4) = f^(k1,k2)* f^(k3,k4)* f^(k1+k3,k2+k4)
%****
%    Here f^ is the 2D Fourier Transform of f and * means complex conjugation
%    (starting at 0 not 1 like matlab)
%i=0,j=0
% % B_red_1 = B(:,1,2,1);
% % B_red_2 = B(1,:,1,2);
% % B_red_3 = B(2,:,:,2);
% % B_red = [B(1,1,1,1), B_red_1(:)', B_red_2(:)', B_red_3(:)'];
end
    
