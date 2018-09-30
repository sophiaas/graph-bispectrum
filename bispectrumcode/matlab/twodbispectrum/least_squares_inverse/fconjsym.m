
function f = fconjsym(f)

[nx ny]=size(f);
[r,c]=find(~isfinite(f));
for i=1:length(r)
    f(r(i),c(i)) = conj(f(mod(nx-r(i)+1,nx)+1,mod(ny-c(i)+1,ny)+1));
end