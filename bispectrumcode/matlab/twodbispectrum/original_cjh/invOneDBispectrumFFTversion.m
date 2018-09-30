% invOneDBispectrumFFTversion.m - compute FFT inverse of bispectrum of f: Z/nZ -> C
%
% function f = invOneDBispectrumFFTversion(B,n)
%
% n:        positive integer
% B:        complex vector of length n+1
% 
% Computes a 1x(n) vector f^ whose fourier inverse, f, has truncated bispectrum B
%
% Christopher Hillar 2008

function F = invOneDBispectrumFFTversion(B,n)

% recover Fourier transform of f
F = zeros(1,n);
F(1) = norm(B(1))^(1/3)*exp(-i*angle(B(1)));

% first need to recover F(2) - up to translation
Prod = 1;
for j=3:n+1
    if mod(j,2) == 1
        Prod = Prod * B(j);
    else
        Prod = Prod / conj(B(j));
    end
    if mod(n,2) == 1
        Prod = Prod / F(1);
    else
        Prod = Prod * conj(F(1));
    end    
end

F2norm = (B(2)/conj(F(1)))^(1/2);
F2arg = -(1/n)*angle(Prod);
F(2) = F2norm*exp(i*F2arg); %for different shifts: *exp(2*pi*i*k/n);

% recover the rest of the Fourier transform of f
for j=3:n
    F(j) = B(j)/conj(F(2)*F(j-1));
end
