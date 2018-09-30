

% Test...
f = phantom(128);
f = f-mean(f(:));
f = f/std(f(:));
f = imresize(f,[18 22]);
[B] = TwoDBispectrumAllCoeffs(f,size(f,1),size(f,2));
[imhat, Fhat, id, bid, Am, Ap, b, fpopt] = bsp_leastsquares_recon_fromFullBSP(B);

figure(1)
clf
subplot(1,2,1)
imagesc(imhat)
axis image off;
title('Reconstruction')
subplot(1,2,2)
imagesc(fftshift(abs(Fhat)))
axis image off;
title('FFT')
colormap gray
drawnow

%%

dat_dir = 'C:\Users\ian\Documents\My Dropbox\bispectrum_cat\code\py\mat_BIS_STA_files\';
for n=6:22
    try
        load([dat_dir sprintf('BIS_STA_N%i.mat',n)])
        [imhat, Fhat, id, bid, Am, Ap, b, fpopt] = bsp_leastsquares_recon_fromFullBSP(arr_0);

        timhat = timeshift_phi(imhat);
        timhat = timeshift_phi(timhat')';
        imhat=timhat;
        imhat = imhat/max(abs(imhat(:)));
        a = uint8(255*((imhat+1)/2)+1);
        imwrite(a,[dat_dir sprintf('lsqinvBIS_STA_N%i.png',n)])
        a = imresize(imhat,size(imhat)*10);

        a = uint8(255*((a+1)/2)+1);
        imwrite(a,[dat_dir sprintf('lsqinvBIS_STA_N%i_resize.png',n)])


        figure(1)
        clf
        subplot(1,2,1)
        imagesc(imhat)
        axis image off;
        title('Reconstruction')
        subplot(1,2,2)
        imagesc(fftshift(abs(Fhat)))
        axis image off;
        title('FFT')
        colormap gray
        drawnow

        figure(2)
        imagesc(a)
        axis image off;
        colorbar
        drawnow
    end
end
