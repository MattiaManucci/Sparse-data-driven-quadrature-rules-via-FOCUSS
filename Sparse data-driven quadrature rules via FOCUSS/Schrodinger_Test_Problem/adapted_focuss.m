function [X,x] = adapted_focuss(A,b,x0,gamma,p,maxit,tol)
%% Input arguments
% ||Ax-b||<=gamma
% 0.5<p<1
% Maxit: number of maximum iterations
% tol: tolerance
% x0: starting solutions (normally we take the full order quadrature)
%% Output arguments
% X: contains the quadrature found at each algorithm iteration
% x: final sparse quadrature
n=size(A,2);
X=[];
%% Solving with FOCUSS
delta=gamma; delta_or=delta; gamma=logspace(-50,-2,200); q=0; NZ_old=n;
res=norm(A*x0-b);
% Ceck that delta is not smaller than the numerical error associated to
% the residual produced by the full quadrature
if res>delta
    fprintf('Warning: required epsilon_1 %g is smaller than the residual associated to the full quadrature %g \n',delta,res);
    delta=res;
end
x = x0; x_full=x0;
X=[X;x0'];
zer=[]; n_zer=1:1:numel(x0);
for k = 1:maxit
    xold = x;
    w2 = (x.^(p));
    w2=real(w2); % Needs beacuse small negative entries could not be truncated
    A_N=A; A_N(:,zer)=[]; % We do not consider columns associated with zero entries of the quadrature
    W=spdiags(w2,0,numel(w2),numel(w2)); AW=A_N*W;
    [UUw,SSw,VVw]=svd(AW,'econ'); SSw=diag(SSw);
    coef=UUw'*b;
    res=@(lambda) sqrt(((lambda./(SSw.^2+lambda)).^2)'*(coef).^2);
    vec=res(gamma); [~,ind]=max(vec(vec<delta));
    x=zeros(size(w2));
    if numel(ind)<1
        [~,ind]=min(vec);
    end
    for i=1:min(size(A_N))
        x=x+(SSw(i)/(SSw(i)^2+gamma(ind(1))))*VVw(:,i)*(coef(i));
    end
    x=W*x;
    % Truncations of entries which are considered too small
    x(abs(x)<delta*1e-3)=0;
    % Ceck of negative entries
    ind_n=find(x<0);
    % Applaying relaxetion to denay negative entries
    if numel(ind_n)>0
        s=1;
        while numel(ind_n)>0
            alpha=-(abs(xold((ind_n(s))))/(x((ind_n(s)))-abs(xold(ind_n(s)))));
            x=alpha*x+(1-alpha)*xold;
            ind_n=find(x<-eps);
        end
    end
    x_full(n_zer)=x;
    resnorm = norm((A_N*x-b));
    zer=find(abs(x_full)<eps);
    n_zer=find(abs(x_full)>=eps);
    NZ=numel(x_full)-numel(zer); %Number of non-zero entries
    if NZ==NZ_old
        q=q+1;
    else
        q=0;
    end
    NZ_old=NZ;
    X=[X;x_full'];
    err = norm((x-xold))/norm(x); % Error between two iterate solutions
    fprintf('Error after %d iterations : %g , residual norm : %g\n',k,err,resnorm);
    if ((err<tol)&&(NZ<=size(A,1)&&(resnorm<delta_or)))||(q>100)
        x=x_full;
        X=X';
        ind=find(x);
        % Truncating small entries
        for j=1:length(ind)
            if abs(x(ind(j)))<delta*1e-3
                x(ind(j))=0;
            end
        end
        fprintf('Converged at iteration : %d \n',k);
        break
    end
    x=x_full;
    x(zer)=[];
end
%% Ceck that the maximum number of iterations is not reached
if k == maxit
    x=x_full;
    ind=find(x);
    Max=max(abs(x));
    % Truncating small entries
    for j=1:length(ind)
        if abs(x(ind(j)))<tol*Max
            x(ind(j))=0;
        end
    end
    fprintf('Algorithm did not converge after %d iterations, err : %g\n',k,err);
end
end