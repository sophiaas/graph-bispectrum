function new_phi = center_im(phi)

[nx ny]=size(phi);

center = 'mass';

mid = ceil(ny/2);

new_phi = zeros(size(phi));

den = sum(sum(abs(phi)));
[tmp,ind] = max(  sum(abs(phi)) / den );
ind
    delta = mid - ind;
    delta
    new_phi = circshift(phi',delta)';