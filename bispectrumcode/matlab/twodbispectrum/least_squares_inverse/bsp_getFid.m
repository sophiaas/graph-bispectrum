
function id = bsp_getFid(N)

if length(N)==1
    N = [N N];
end

id=[];
[y,x] = meshgrid(0,0:(N(1)/2));
id = [id; [x(:) y(:)]];
for i=1:(N(2)/2-1)
    [y,x] = meshgrid(i,0:(N(1)-1));
    id = [id; [x(:) y(:)]];
end
[y,x] = meshgrid(N(2)/2,0:(N(1)/2));
id = [id; [x(:) y(:)]];
id = sub2ind([N(1) N(2)],id(:,1)+1,id(:,2)+1);
id = id';