close all
clearvars
clc
%% Parametrized Non-Linear Diffusion Problem: script to validate the sparse quadrature rules, see [1]
%% References
% [1] M. Manucci, J. V. Aguado, D. Borzacchiello
% Sparse data-driven quadrature rules via $\ell^p$-quasi-norm minimization.
% arXiv preprint arXiv:2012.05264
% [2] J.A. Hernandez, M.A. Caicedo, A.Ferrer
% Dimensional  hyper-reduction  of  nonlinear  finite  ele-ment models via
% empirical cubature.
% Computer Methods in Applied Mechanics and Engineering, 2017.
dataname = 'Data/Output/Database/Database_1';
load(dataname);
C1 = [0,10];
C2 = [1,10];
C3 = [0,10];
%% Inizializzations
delta_r=1e-10; delta=1e-6;
%% POD-BASE
[U,S,~]=svd(TT,'econ');
d = 1-cumsum(diag(S).^2)/sum(diag(S).^2);
ind = find(d<delta_r);
ind(1)=7;
PHI_T = U(:,1:ind(1));
Nr = size(KK,2); Nb = size(PHI_T,2);
RES= zeros(Nr*Nb,size(KK,1));
YY = zeros(Nr*Nb^2,1);
%% BULDING DATA FOR HYPER-REDUCTION
%1) Corresponding Reduced Solution
In_r=(PHI_T'*In')'; Dx_r=(PHI_T'*Dx')'; Dy_r=(PHI_T'*Dy')'; f_r=PHI_T'*f;
Nq=size(Q,1); TT=zeros(Nb,Nr); KK=zeros(Nq,Nr);
for j = 1:Nr
    c1 = mu1(j)*(C1(2)-C1(1))+C1(1);
    c2 = mu2(j)*(C2(2)-C2(1))+C2(1);
    c3 = mu3(j)*(C3(2)-C3(1))+C3(1);
    
    kouter = @(u)(1+c1*u);
    kinclusion = @(u)(c2+c2*u.^2);
    T = zeros(Nb,1);
    % Loop for the non-linearity
    for i = 1:100
        Told = T;
        Tint = In_r*T;
        k = zeros(Nq,1);
        k(Tag==2)=kouter(Tint(Tag==2));
        k(Tag==3)=kinclusion(Tint(Tag==3));
        K = Dx_r'*Q*spdiags(k,0,size(Q,1),size(Q,2))*Dx_r+...
            Dy_r'*Q*spdiags(k,0,size(Q,1),size(Q,2))*Dy_r;
        T =K\f_r;
        errest = norm(T-Told)/norm(T);
        fprintf('Simulation : %d/%d Error estimation at iteration %d = %g\n',j,Nr,i,errest);
        if errest < 1e-8
            break
        end
    end
    TT(:,j)=T; KK(:,j)=k;
    
    for n=1:Nb
        RES(Nb*(j-1)+n,:)=((Dx_r*T)'.*(Dx_r(:,n))'+(Dy_r*T)'.*(Dy_r(:,n))').*k'-10*(In_r(:,n))';
    end
    
end

RES=[RES;ones(1,Nq)]; b=zeros(Nb*Nr+1,1); b(end)=1;
%% Sparse rule by linear programming algorithm
ff=ones(Nq,1); delta_LP=delta;
N_A=[RES;-RES]; N_b=[delta+b; delta-b]; lb=zeros(Nq,1); ub=[]; Aeq=[]; beq=[];
tic
options = optimoptions('linprog','Algorithm','dual-simplex','OptimalityTolerance',1e-5,'ConstraintTolerance',1e-5);
x = linprog(ff,N_A,N_b,Aeq,beq,lb,ub,options);
time_LP=toc;
ind_LP=find(x>0); w_LP=x(ind_LP);
%% Truncated SVD on constrain matrix
f1=0.01; f2=1; % The A and B in [1]
tic
[U,S,V]=rsvd(RES',1000);
time_FO_SVD=toc;
d= (cumsum(vpa(diag(S).^2))); d=((norm(diag(Q)))+sum(diag(Q)))*sqrt(vpa(d(end))*ones(numel(d),1)-vpa(d));
ind_f = find(double(d)<f1*delta);
UU=U(:,1:ind_f(1)); SS=S(1:ind_f(1),1:ind_f(1)); VV=V(:,1:ind_f(1));
A_F=SS*UU'; b_F=(VV'*b);
%% Sparse rule by FOCUSS algorithm and methodolgy developed in [1]
p=0.95; maxit=2000; tol=1e-10; flag=1; x0=diag(Q); S_f=max((sum(abs(A_F*RES'),1))); delta_2=f2*delta/S_f;
tic
[XX,x1] = adapted_focuss(A_F,b_F,x0,delta_2,p,maxit,tol);
time_FO=time_FO_SVD+toc;
ind_FO=find(x1>0); w_FO=x1(ind_FO);
%% Sparse rule by Non-Negative Least Square Method algorithm, see [2]
tic
xls= Heuristic_approach(RES,b,delta,150);
time_NL=toc;
ind_NL=find(xls>0); w_NL=xls(ind_NL);
%% Validation
% Focuss quadrature
Dx_hr=Dx_r(ind_FO,:); Dy_hr=Dy_r(ind_FO,:); In_hr=In_r(ind_FO,:); Qr = spdiags(w_FO,0,numel(ind_FO),numel(ind_FO)); Tag_hr=Tag(ind_FO);
f_hr = 10*In_hr'*Qr*ones(size(Qr,2),1); Ntrain=300; err_r_FO=zeros(Ntrain,1); Err_FO=zeros(Ntrain*Nb,1); err_full_FO=zeros(Ntrain,1);
% LP quadrature
Dx_hr_LP=Dx_r(ind_LP,:); Dy_hr_LP=Dy_r(ind_LP,:); In_hr_LP=In_r(ind_LP,:); Qr_LP = spdiags(w_LP,0,numel(ind_LP),numel(ind_LP)); Tag_hr_LP=Tag(ind_LP);
f_hr_LP = 10*In_hr_LP'*Qr_LP*ones(size(Qr_LP,2),1); err_r_LP=zeros(Ntrain,1); Err_LP=zeros(Ntrain*Nb,1); err_full_LP=zeros(Ntrain,1);
% EC quadrature
Dx_hr_EC=Dx_r(ind_NL,:); Dy_hr_EC=Dy_r(ind_NL,:); In_hr_EC=In_r(ind_NL,:); Qr_EC = spdiags(w_NL,0,numel(ind_NL),numel(ind_NL)); Tag_hr_EC=Tag(ind_NL);
f_hr_EC = 10*In_hr_EC'*Qr_EC*ones(size(Qr_EC,2),1); err_r_EC=zeros(Ntrain,1); Err_EC=zeros(Ntrain*Nb,1); err_full_EC=zeros(Ntrain,1);
% Test Sample
mu1 = rand(Ntrain,1); mu2 = rand(Ntrain,1); mu3 = rand(Ntrain,1);
for j=1:Ntrain
    
    
    c1 = mu1(j)*(C1(2)-C1(1))+C1(1);
    c2 = mu2(j)*(C2(2)-C2(1))+C2(1);
    c3 = mu3(j)*(C3(2)-C3(1))+C3(1);
    
    kouter = @(u)(1+c1*u);
    kinclusion = @(u)(c2+c3*u.^2);
    mask = true(size(In,2),1); mask(Boundary)=false;
    T_full=zeros(size(In,2),1); T = zeros(Nb,1); z = zeros(Nb,1); z_LP=zeros(Nb,1); z_EC=zeros(Nb,1);
    %% Loop on non-linearity of the full Problem
    for i = 1:100
        Told = T_full;
        Tint = In*T_full;
        k = zeros(Nq,1);
        k(Tag==2)=kouter(Tint(Tag==2));
        k(Tag==3)=kinclusion(Tint(Tag==3));
        K = Dx'*Q*spdiags(k,0,size(Q,1),size(Q,2))*Dx+...
            Dy'*Q*spdiags(k,0,size(Q,1),size(Q,2))*Dy;
        T_full(mask) =K(mask,mask)\f(mask);
        errest = norm(T_full-Told)/norm(T_full);
        fprintf('Simulation : %d/%d Error estimation at iteration %d = %g\n',j,Ntrain,i,errest);
        if errest < 1e-8
            break
        end
    end
    %% Loop on non-Linearity of Standard Reduced-Basis
    for i = 1:100
        Told = T;
        Tint = In_r*T;
        k = zeros(Nq,1);
        k(Tag==2)=kouter(Tint(Tag==2));
        k(Tag==3)=kinclusion(Tint(Tag==3));
        K = Dx_r'*Q*spdiags(k,0,size(Q,1),size(Q,2))*Dx_r+...
            Dy_r'*Q*spdiags(k,0,size(Q,1),size(Q,2))*Dy_r;
        T =K\f_r;
        errest = norm(T-Told)/norm(T);
        fprintf('Simulation : %d/%d. Error estimation at iteration %d = %g\n',j,Ntrain,i,errest);
        if errest < 1e-8
            break
        end
    end
    %% Loop on non-Linearity of Hypereduction-FOCUSS (fo) MOR
    for i = 1:100
        zold = z;
        zint = In_hr*z;
        k_FO = zeros(Nq,1);
        k_FO(Tag_hr==2)=kouter(zint(Tag_hr==2));
        k_FO(Tag_hr==3)=kinclusion(zint(Tag_hr==3));
        K = Dx_hr'*Qr*spdiags(k_FO,0,size(Qr,1),size(Qr,2))*Dx_hr+...
            Dy_hr'*Qr*spdiags(k_FO,0,size(Qr,1),size(Qr,2))*Dy_hr;
        z =K\f_hr;
        errest = norm(z-zold)/norm(z);
        fprintf('ROM:Error estimation at iteration %d = %g\n',i,errest);
        if errest < 1e-8
            break
        end
    end
    %% Loop on non-Linearity of Hypereduction-Linear-Programming (lp) MOR
    for i = 1:100
        zold_LP = z_LP;
        zint_LP = In_hr_LP*z_LP;
        k_hr_LP = zeros(Nq,1);
        k_hr_LP(Tag_hr_LP==2)=kouter(zint_LP(Tag_hr_LP==2));
        k_hr_LP(Tag_hr_LP==3)=kinclusion(zint_LP(Tag_hr_LP==3));
        K = Dx_hr_LP'*Qr_LP*spdiags(k_hr_LP,0,size(Qr_LP,1),size(Qr_LP,2))*Dx_hr_LP+...
            Dy_hr_LP'*Qr_LP*spdiags(k_hr_LP,0,size(Qr_LP,1),size(Qr_LP,2))*Dy_hr_LP;
        z_LP =K\f_hr_LP;
        errest = norm(z_LP-zold_LP)/norm(z_LP);
        fprintf('ROM:Error estimation at iteration %d = %g\n',i,errest);
        if errest < 1e-8
            break
        end
    end
    %% Loop on non-Linearity of Hypereduction-Empirical-Cubature (ec) MOR
    for i = 1:100
        zold_EC = z_EC;
        zint_EC = In_hr_EC*z_EC;
        k_hr_EC = zeros(Nq,1);
        k_hr_EC(Tag_hr_EC==2)=kouter(zint_EC(Tag_hr_EC==2));
        k_hr_EC(Tag_hr_EC==3)=kinclusion(zint_EC(Tag_hr_EC==3));
        K = Dx_hr_EC'*Qr_EC*spdiags(k_hr_EC,0,size(Qr_EC,1),size(Qr_EC,2))*Dx_hr_EC+...
            Dy_hr_EC'*Qr_EC*spdiags(k_hr_EC,0,size(Qr_EC,1),size(Qr_EC,2))*Dy_hr_EC;
        z_EC =K\f_hr_EC;
        errest = norm(z_EC-zold_EC)/norm(z_EC);
        fprintf('ROM:Error estimation at iteration %d = %g\n',i,errest);
        if errest < 1e-8
            break
        end
    end
    %% Error Ralated to full and hr solution
    err_full_FO(j)=sqrt((PHI_T*z-T_full)'*(Dx'*Q*Dx+Dy'*Q*Dy)*(PHI_T*z-T_full))/(sqrt(T_full'*(Dx'*Q*Dx+Dy'*Q*Dy)*T_full));
    err_full_LP(j)=sqrt((PHI_T*z_LP-T_full)'*(Dx'*Q*Dx+Dy'*Q*Dy)*(PHI_T*z_LP-T_full))/(sqrt((T_full)'*(Dx'*Q*Dx+Dy'*Q*Dy)*(T_full)));
    err_full_EC(j)=sqrt((PHI_T*z_EC-T_full)'*(Dx'*Q*Dx+Dy'*Q*Dy)*(PHI_T*z_EC-T_full))/(sqrt((T_full)'*(Dx'*Q*Dx+Dy'*Q*Dy)*(T_full)));
    %% Ralated to integration of the non-linearity
    err_r_FO(j)=sqrt((PHI_T*(z-T))'*(Dx'*Q*Dx+Dy'*Q*Dy)*(PHI_T*(z-T)))/(sqrt((PHI_T*(T))'*(Dx'*Q*Dx+Dy'*Q*Dy)*(PHI_T*(T))));
    err_r_LP(j)=sqrt((PHI_T*(z_LP-T))'*(Dx'*Q*Dx+Dy'*Q*Dy)*(PHI_T*(z_LP-T)))/(sqrt((PHI_T*(T))'*(Dx'*Q*Dx+Dy'*Q*Dy)*(PHI_T*(T))));
    err_r_EC(j)=sqrt((PHI_T*(z_EC-T))'*(Dx'*Q*Dx+Dy'*Q*Dy)*(PHI_T*(z_EC-T)))/(sqrt((PHI_T*(T))'*(Dx'*Q*Dx+Dy'*Q*Dy)*(PHI_T*(T))));
    
    k_hr=k(ind_FO);
    k_hr_LP=k(ind_LP);
    k_hr_EC=k(ind_NL);
    %% Error ralated to Residual formulation
    for n=1:Nb
        Err_FO(Nb*(j-1)+n)=abs((((Dx_r*T)'.*(Dx_r(:,n))'+(Dy_r*T)'.*(Dy_r(:,n))').*k'-10*(In_r(:,n))')*diag(Q)-...
            (((Dx_hr*T)'.*(Dx_hr(:,n))'+(Dy_hr*T)'.*(Dy_hr(:,n))').*k_hr'-10*(In_hr(:,n))')*w_FO);
        Err_LP(Nb*(j-1)+n)=abs((((Dx_r*T)'.*(Dx_r(:,n))'+(Dy_r*T)'.*(Dy_r(:,n))').*k'-10*(In_r(:,n))')*diag(Q)-...
            (((Dx_hr_LP*T)'.*(Dx_hr_LP(:,n))'+(Dy_hr_LP*T)'.*(Dy_hr_LP(:,n))').*k_hr_LP'-10*(In_hr_LP(:,n))')*w_LP);
        Err_EC(Nb*(j-1)+n)=abs((((Dx_r*T)'.*(Dx_r(:,n))'+(Dy_r*T)'.*(Dy_r(:,n))').*k'-10*(In_r(:,n))')*diag(Q)-...
            (((Dx_hr_EC*T)'.*(Dx_hr_EC(:,n))'+(Dy_hr_EC*T)'.*(Dy_hr_EC(:,n))').*k_hr_EC'-10*(In_hr_EC(:,n))')*w_NL);
        
    end
end
maxErr_full_FO=max(err_full_FO); maxErr_full_LP=max(err_full_LP); maxErr_full_EC=max(err_full_EC);
maxErr_FO=max(Err_FO);           maxErr_LP=max(Err_LP);           maxErr_EC=max(Err_EC);
maxerr_r_FO=max(err_r_FO);       maxerr_r_LP=max(err_r_LP);       maxerr_r_EC=max(err_r_EC);



