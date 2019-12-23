function[W_new]=Sparse_new(W,Results_segment,num_segment)
W_new=W(Results_segment.index{num_segment},:);
[R,C]=size(Results_segment.index{num_segment});
[I,J,S]=find(W_new);
[N,M]=size(J);
for i=1:N    
   if( size(find(J(i)==Results_segment.index{num_segment}),2))
          J(i)=find(J(i)==Results_segment.index{num_segment});   
   else
   
     S(i)=0;     
   end
end
W_new=sparse(I,J,S);
W_new=W_new(:,1:C);
if(size(W_new,1)~=size(W_new,2))
    Add=zeros(1,C);
    W_new=[W_new;Add];
    
end



end

