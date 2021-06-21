clc
clearvars
close all
%% Schrodinger fundamental solution, see [1]
%% References
% [1] M. Manucci, J. V. Aguado, D. Borzacchiello
% Sparse data-driven quadrature rules via $\ell^p$-quasi-norm minimization.
% arXiv preprint arXiv:2012.05264
% [2] J.A. Hernandez, M.A. Caicedo, A.Ferrer
% Dimensional  hyper-reduction  of  nonlinear  finite  element models via
% empirical cubature.
% Computer Methods in Applied Mechanics and Engineering, 2017.
%% Script to compare FOCUSS (fo)-Linear Programming (lp)- Empirical Cubature (ec) 
alpha_min=0; alpha_max=2; t0=0.2; tmax=4; eps_max=4; N=1200; 
delta=1e-6; %Accuracy required
J=40; %Dimension of the train Sample 
N_test=200; %Dimension of the test sample
ff=ones(N,1);
a=linspace(alpha_min,alpha_max,J); b=linspace(t0,tmax,J);
alpha=kron(a,ones(size(b))); t=kron(ones(size(a)),b);
par=[alpha; t]; OC=numel(alpha);
QP=linspace(0,eps_max,N); A=zeros(OC,N); aa=0.5; %Varianza Gaussiana
f_t=@ (s) exp((-aa*s^2));
g=@ (alpha,t, s) (cos(((alpha*s))/(2*t))*cos(((s^2))/(4*t))-sin(((s^2))/(4*t))*sin(((alpha*s))/(2*t)))*f_t(s); 
%% Evaluating the integrals on the train sample
c=(eps_max/N);
for i=1:OC
    A(i,1)= c*(1/(2))*g(par(1,i),par(2,i),QP(1)); A(i,end)= c*(1/(2))*g(par(1,i),par(2,i),QP(end));
    for k=2:(N-1)
        A(i,k)=c*g(par(1,i),par(2,i),QP(k));
    end
end
% Addaning contrain on the domain size
A=[A;ones(1,N)];
b=sum(A,2);
%% Sparse rule by linear programming algorithm
N_A=[A;-A]; N_b=[delta+b; delta-b]; lb=zeros(N,1); ub=[]; Aeq=[]; beq=[];
options = optimoptions('linprog','Algorithm','dual-simplex','OptimalityTolerance',10e-9,'ConstraintTolerance',1e-8);
tic
x = linprog(ff,N_A,N_b,Aeq,beq,lb,ub,options);
time_LP=toc;
ind_LP=find(x>0); w_LP=x(ind_LP);
%% Sparse rule by FOCUSS algorithm and methodolgy developed in [1]
f1=0.1; f2=1; % The A and B in [1]
% Reducing size of train sample by truncated SVD
tic
[UU,S,VV]=svd(A');
time_SVD=toc;
d= (cumsum(vpa(diag(S).^2))); d=((norm(ones(N,1)))+sum((ones(N,1))))*sqrt(vpa(d(end))*ones(numel(d),1)-vpa(d)); %d=d*871;
ind_f = find(double(d)<f1*delta);
U=UU(:,1:ind_f(1)); SS=S(1:ind_f(1),1:ind_f(1)); V=VV(:,1:ind_f(1));
A_F=SS*U'; b_F=V'*b; 
S_f=max((sum(abs(A_F*A'),1)));
p=0.95; maxit=4000; tol=1e-10;
delta_2=f2*delta/S_f; %epsilon_1 in [1]
x0=ff;
tic
[X,x1] = adapted_focuss(A_F,b_F,x0,delta_2,p,maxit,tol);
time_FO=time_SVD+toc;
ind_FO=find(x1>0); w_FO=x1(ind_FO);
%% Sparse rule by Non-Negative Least Square Method algorithm, see [2]
f1=0.1; ind_f = find(double(d)<f1*delta);
U=UU(:,1:ind_f(1)); SS=S(1:ind_f(1),1:ind_f(1)); V=VV(:,1:ind_f(1));
options = optimset('TolX',1e-14);
tic
xls= Heuristic_approach(A,b,delta,100);
time_NL=toc;
ind_NL=find(xls>0); w_NL=xls(ind_NL);
%% Test sample
a_test=(alpha_max-alpha_min)*linspace(0,1,N_test)+alpha_min; b_test=(tmax-t0)*linspace(0,1,N_test)+t0;
alpha_test=kron(a_test,ones(size(b_test))); t_test=kron(ones(size(a_test)),b_test);
par_test=[alpha_test; t_test]; OC_test=numel(alpha_test); err_LP=zeros(1,OC_test); err_FO=zeros(1,OC_test); err_NL=zeros(1,OC_test);
for i=1:OC_test
    
    I_true=(eps_max/(2*N))*g(par_test(1,i),par_test(2,i),QP(1))+(eps_max/(2*N))*g(par_test(1,i),par_test(2,i),QP(end));
    %Computing integrals with full order quadrature
    for k=2:(N-1)
        I_true=I_true+(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(k));
    end
    I_LP=0;
    %Computing integrals with sparse linear programming quadrature
    if ind_LP(1)==1
        
        I_LP=w_LP(1)*(eps_max/(2*N))*g(par_test(1,i),par_test(2,i),QP(1));
        for k=2:(numel(ind_LP)-1)
            I_LP=I_LP+w_LP(k)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_LP(k)));
            
        end
        
        if ind_FO(end)==N
            I_LP=I_LP+w_LP(end)*(eps_max/(2*N))*g(par_test(1,i),par_test(2,i),QP(ind_LP(end)));
        else
            I_LP=I_LP+w_LP(end)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_LP(end)));
        end
    else
        for k=1:(numel(ind_LP)-1)
            I_LP=I_LP+w_LP(k)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_LP(k)));
            
        end
        if ind_LP(end)==N
            I_LP=I_LP+w_LP(end)*(eps_max/(2*N))*g(par_test(1,i),par_test(2,i),QP(ind_LP(end)));
        else
            I_LP=I_LP+w_LP(end)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_LP(end)));
        end
    end
    %Computing integrals with sparse FOCUSS quadrature
    I_FO=0;
    if ind_FO(1)==1
        I_FO=w_FO(1)*(eps_max/(2*N))*g(par_test(1,i),par_test(2,i),QP(1));
        for k=2:(numel(ind_FO)-1)
            I_FO=I_FO+w_FO(k)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_FO(k)));
            
        end
        
        if ind_FO(end)==N
            I_FO=I_FO+w_FO(end)*(eps_max/(2*N))*g(par_test(1,i),par_test(2,i),QP(ind_FO(end)));
        else
            I_FO=I_FO+w_FO(end)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_FO(end)));
        end
    else
        for k=1:(numel(ind_FO)-1)
            I_FO=I_FO+w_FO(k)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_FO(k)));
            
        end
        if ind_FO(end)==N
            I_FO=I_FO+w_FO(end)*(eps_max/(2*N))*g(par_test(1,i),par_test(2,i),QP(ind_FO(end)));
        else
            I_FO=I_FO+w_FO(end)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_FO(end)));
        end
    end
    %Computing integrals with sparse EMPIRICAL CUBATURE quadrature
    I_NL=0;
    if ind_NL(1)==1
        I_NL=w_NL(1)*(eps_max/(2*N))*g(par_test(1,i),par_test(2,i),QP(1));
        for k=2:(numel(ind_NL)-1)
            I_NL=I_NL+w_NL(k)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_NL(k)));
            
        end
        
        if ind_NL(end)==N
            I_NL=I_NL+w_NL(end)*(eps_max/(2*N))*g(par_test(1,i),par_test(2,i),QP(ind_NL(end)));
        else
            I_NL=I_NL+w_NL(end)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_NL(end)));
        end
    else
        for k=1:(numel(ind_NL)-1)
            I_NL=I_NL+w_NL(k)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_NL(k)));
            
        end
        if ind_NL(end)==N
            I_NL=I_NL+w_NL(end)*(eps_max/(2*N))*g(par_test(1,i),par_test(2,i),QP(ind_NL(end)));
        else
            I_NL=I_NL+w_NL(end)*(eps_max/N)*g(par_test(1,i),par_test(2,i),QP(ind_NL(end)));
        end
    end
    err_LP(i)=abs(I_true-I_LP); err_FO(i)=abs(I_true-I_FO); err_NL(i)=abs(I_true-I_NL);
end
FOCUS_RULE=zeros(N,1); FOCUS_RULE(ind_FO)=w_FO;
MAX_ERR_LP=max(err_LP); MAX_ERR_FO=max(err_FO);   MAX_ERR_NL=max(err_NL);
AV_ERR_LP=mean(err_LP); AV_ERR_FO=mean(err_FO);   AV_ERR_NL=mean(err_NL);
MIN_ERR_LP=min(err_LP); MIN_ERR_FO=min(err_FO);   MIN_ERR_NL=min(err_NL);

