close all
clearvars
clc
%% Parametrized Non-Linear Diffusion Problem: script to create the data-Set for MOR, see [1]
%% References
% [1] M. Manucci, J. V. Aguado, D. Borzacchiello
% Sparse data-driven quadrature rules via $\ell^p$-quasi-norm minimization.
% arXiv preprint arXiv:2012.05264
%The mesh used
datamesh = 'Data/Mesh/2D_Inclusion';
%FEM Matrices
FEM_assamble='Data/FEM/FEM_Matrices';
%Databese created
dataname = 'Data/Output/Database/Database_1';
C1 = [0,10];
C2 = [1,10];
C3 = [0,10];
% Load Mesh and FEM Matrices
load(datamesh)
load(FEM_assamble)
f = 10*In'*Q*ones(size(Q,2),1);
Boundary = unique(TOPO{1}(PHYSICAL_TAG{1}==1));
Nq = size(Q,2)/size(TOPO{2},1);
N = size(X,1);
Tag = repmat(PHYSICAL_TAG{2}(:)',Nq,1); Tag=Tag(:);
%% Loop To evaluate the Solution
N1=7; N2=7; N3=7; Ntest=N1*N2*N3;
a=linspace(0,1,N1); b=linspace(0,1,N2); c=linspace(0,1,N3);
mu1=kron(ones(1,N1*N3),b); mu2=kron(ones(1,N3),kron(a,ones(1,N2))); mu3=kron(c,ones(1,N1*N2));
TT = zeros(N,Ntest);
KK = zeros(size(Q,1),Ntest);
% Loop for the train sample
for j = 1:Ntest
    c1 = mu1(j)*(C1(2)-C1(1))+C1(1);
    c2 = mu2(j)*(C2(2)-C2(1))+C2(1);
    c3 = mu3(j)*(C3(2)-C3(1))+C3(1);
    
    kouter = @(u)(1+c1*u);
    kinclusion = @(u)(c2+c3*u.^2);
    mask = true(N,1); mask(Boundary)=false;
    T = zeros(N,1);
    % Loop on the non-linearity
    for i = 1:100
        Told = T;
        Tint = In*T;
        k = zeros(Nq,1);
        k(Tag==2)=kouter(Tint(Tag==2));
        k(Tag==3)=kinclusion(Tint(Tag==3));
        K = Dx'*Q*spdiags(k,0,size(Q,1),size(Q,2))*Dx+...
            Dy'*Q*spdiags(k,0,size(Q,1),size(Q,2))*Dy;
        T(mask) =K(mask,mask)\f(mask);
        errest = norm(T-Told)/norm(T);
        fprintf('Simulation : %d/%d Error estimation at iteration %d = %g\n',j,Ntest,i,errest);
        if errest < 1e-8
            break
        end
    end
    TT(:,j)=T; KK(:,j)=k;
end
%% Saving important variables
save(dataname,'X','TT','KK','Dx','Dy','In','Q','TOPO','Tag','f','mu1','mu2','mu3','Boundary');
