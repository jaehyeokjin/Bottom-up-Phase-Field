load 1_1.dat
pmf=X1_1;
ids=zeros(size(pmf,1),1);
for k=1:1:size(pmf,1)
ids(k)=k;
e(k)=-trapz(pmf(end:-1:k,1),pmf(end:-1:k,2));
end
data(:,1)=ids;
data(:,2)=pmf(:,1);
data(:,3)=e;
data(:,4)=pmf(:,2);
dlmwrite('MeOH.table',data,'delimiter','\t','precision','%.8f');
