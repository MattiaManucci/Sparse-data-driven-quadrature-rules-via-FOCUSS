function [ w ] = Heuristic_approach( A,b,tol,m )
%% Initialization
% m number of max non zero element I want 
M = size(A,2);
z = []; %integration points
y = 1:1:M;
nnz = 0;
k = 1; %number of iterations
res = b ;
Nres=norm(res);
options = optimset('TolX',1e-28);
ww=zeros(1,M); J=zeros(size(A,1),M);
for i = 1:M
    ww(i)=1/norm(A(:,i));
    J(:,i)=A(:,i)*ww(i);
end
k=0; % Counter for the iterations
while Nres>tol && nnz<=m
    v=zeros(numel(y),1);
    for j = 1:numel(y)
        v(j) = J(:,y(j))'*res/Nres;
    end
    [~,i]=max(v);
    %z(k)=y(i);
    z=[z,y(i)];
    y(i)=[];
    z=sort(z);
    Jz=J(:,z);
    x=(Jz'*Jz)\(Jz'*b);
    nneg=find(x<0);
    if numel(nneg)>=1
        x = lsqnonneg(Jz,b,options);
        z0=find(x==0);
        z(z0)=[];
        x(z0)=[];
        y=[y,z0'];
        y=sort(y);
        Jz=J(:,z);
    end
     
    res=b-Jz*x;
    Nres=norm(res);
    k=k+1;
    nnz=numel(z);
    tolll=Nres;
    fprintf('Residual after %d iterations : %g , number of non-zero entries: %g\n',k,tolll,nnz);
end

w=zeros(M,1);
w(z)=x.*(ww(z)');

end

