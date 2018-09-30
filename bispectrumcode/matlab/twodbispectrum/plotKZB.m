
function bv=plotKZB(B,cP)

nF=(size(B,1)-1)/2;
[i i m n]=size(B);
bv=[];
for i=(nF-cP+1):(nF+cP+1)
    br=[];
    for j=(nF-cP+1):(nF+cP+1)
        tmp = squeeze(B(i,j,:,:));
        tmp = sqrt(tmp.*conj(tmp));
        br = [br tmp];
    end
    bv = [bv;br];
end
imagesc(log(bv))
% colormap(1-gray(256))s
axis image; axis off