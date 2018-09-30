function [bid,Am,Ap] = bsp_getBid(id,N)

k11 = 0:(N(2)-1);
k12 = 0:(N(1)-1);

[y,x] = meshgrid(k11,k12);
K1v = [x(:) y(:)];
K1id = sub2ind([N(1) N(2)],K1v(:,1)+1,K1v(:,2)+1);

Am  = sparse(max(N)^3,N(1)*N(2)/2+2);
Ap  = sparse(max(N)^3,N(1)*N(2)/2+2);
bid = zeros(max(N)^3,3);
c=1;
for i=1:size(K1v,1)
    if any(id==K1id(i)) && (K1id(i)>1)
        [y,x] = meshgrid((K1v(i,1)):(N(2)-1-K1v(i,1)),0:(N(1)-1-K1v(i,2)));
        K2id = sub2ind([N(1) N(2)],x(:)+1,y(:)+1);
        k1pk2 = bsxfun(@plus,K1v(i,:),[x(:) y(:)]);
        
        % Ignore k2=0...
        valid_id = (K2id>1);
        k1pk2 = k1pk2(valid_id,:);
        K2id  = K2id(valid_id);
        
        % Ignore k1+k2 outside of range...
        valid_id = (k1pk2(:,1)<N(1))&(k1pk2(:,2)<N(2));
        k1pk2 = k1pk2(valid_id,:);
        K2id  = K2id(valid_id);
        K1K2id = sub2ind([N(1) N(2)],k1pk2(:,1)+1,k1pk2(:,2)+1);
       
        if ~isempty(K2id)
            tmp1 = sparse(bsxfun(@minus,K2id,id)==0);
            tmp2 = sparse(bsxfun(@minus,K1K2id,id)==0);
            valid_id = (sum(tmp1,2).*sum(tmp2,2))>0;
            
            cid = c:(c+sum(valid_id)-1);
            Am(cid,id==K1id(i))=1;
            Am(cid,:)=Am(cid,:)+tmp1(valid_id,:);
            Am(cid,:)=Am(cid,:)+tmp2(valid_id,:);
            
            Ap(cid,id==K1id(i))=1;
            Ap(cid,:)=Ap(cid,:)+tmp1(valid_id,:);
            Ap(cid,:)=Ap(cid,:)-tmp2(valid_id,:);
            
            bid(cid,1) = K1id(i);
            bid(cid,2) = K2id(valid_id);
            bid(cid,3) = K1K2id(valid_id);

            c = c+length(cid);
%         else
%             keyboard
        end
    end
end
Am  = Am(1:(c-1),:);
Ap  = Ap(1:(c-1),:);
bid = bid(1:(c-1),:);