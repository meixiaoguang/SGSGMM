
% Summary of this function goes here
%   Detailed explanation goes here
close all;
clc;
init_choose='hypervca'; %'k-means'


load('toy_image_end_var_3.mat');
M=4;
is_real_dataset=0;
Sw = 7;
[rows,cols,B] = size(I);
A_results=zeros(rows*cols,M);
[Y_org,A_gt,rows,cols] = reshape_hsi(I,A_gt);
reduced_dim=10;
[Y, mapping] = pca(Y_org, reduced_dim);
[~,M]=size(A_gt);
I_org=I;
[rows,cols,B] = size(I);
D = 0.01^2 * eye(B);
beta1 =0.2;
beta2=0;
beta3=0;
beta4=0.1;
options.show_fig = 0;
options.names = names;
options.D = D;
options.project_mode = 'image';
options.convergence_thresh = 0.0001;

tic;
P = round(rows*cols/Sw^2);
Ws = 0.3;
seg = slic_HSI(I, P, Ws);
labels=reshape(seg.labels,rows,cols);
Results_segment = seg_im_class(I, labels);
Num=size(Results_segment.Y,2);

options.reduced_dim=10;
options.convergence_thresh=0.0001;
options.beta2_decay=0.05;
options.beta1=beta1;
options.beta4=beta4;
toc;
tic;
switch init_choose
    case 'hypervca'
        [A_init,~] = hyperVca(seg.X_c,M);
        S_init = fcls(A_init,Y_org');
        sigma0 = 0.08;
        for j = 1:M
            mu_jk_ori{j}(1,:) = A_init(:,j)';
            sigma_jk_ori{j}(:,:,1) = sigma0^2 * eye(B);
            w_jk{j}(1,1) = 1;
        end
        K=ones(1,M);
        endmember_scatter_plot_end_var(Y_org,w_jk,mu_jk_ori,sigma_jk_ori,names,options);
        %%%%%%PCA%%%%%%%
        sigma = mapping.M'*D*mapping.M;
        R=[];
        for j = 1:M
            mu_jk{j} = gmm_project(mu_jk_ori{j}, mapping);
            R=[R;mu_jk{j}];
            sigma_jk{j} = mapping.M'*sigma_jk_ori{j}*mapping.M;
        end
        
        A_results = project_to_simplex(S_init');
        A=A_results;
        [seg.X_c,~] = pca(seg.X_c', reduced_dim);
        Wpmatrix= calc_A_from_mus(seg.X_c, mu_jk);
        Wpmatrix = 1./(M.^2*Wpmatrix + 1);
        
    case 'k-means'
        I=reshape(Y,rows,cols,reduced_dim);
        [mu_jk,sigma_jk,w_jk,K,A] = gmm_init(I,M,options);
        A_results=A;
        [mu_jk_ori,sigma_jk_ori] = restore_from_projection(mu_jk,sigma_jk,[],mapping.mean,mapping.M);
        endmember_scatter_plot_end_var(Y_org,w_jk,mu_jk_ori,sigma_jk_ori,names,options);
        [seg.X_c,~] = pca(seg.X_c', reduced_dim);
        Wpmatrix= calc_A_from_mus(seg.X_c, mu_jk);
        Wpmatrix = 1./(M.^2*Wpmatrix + 1);
end



[K,w_jk,mu_jk,sigma_jk,A1] = estimate_num_comp(Y, A_results, [rows,cols], 0, 4);
options.w_jk = w_jk;
options.mu_jk = mu_jk;
options.sigma_jk = sigma_jk;
Wpmatrix= calc_A_from_mus(seg.X_c, mu_jk);
Wpmatrix = 1./(M.^2*Wpmatrix + 1);
options.K = K;
options.show_approx=1;
for i= 1:seg.P
    
    I_temp = Results_segment.Y{1,i};
    num_segment=i;
    [rows,cols,B] = size(I);
    D = 0.001^2 * eye(200);
    options.beta1 =beta1;
    options.beta2=beta2;
    options.beta3=beta3;
    options.beta4=beta4;
    options.show_fig= 1;
    options.names = names;
    options.D = D;
    options.project_mode = 'image';
    options.convergence_thresh = 0.0001;
    Cj = seg.Cj(Results_segment.index{num_segment})';
    wp = Wpmatrix(i,:)';
    options.Cj=Cj;
    options.wp=wp;
    options.project_mapping=mapping;
    A_temp=A_results(Results_segment.index{num_segment},:);
    options.A=A_temp;
    [A_temp,R_temp,w_jk_temp,mu_jk_temp,sigma_jk_temp,extra_temp] = gmm_huexmei(I_temp,options,endmembers,I_org,Results_segment, num_segment,wp,Cj);
    A(Results_segment.index{i},:)=A_temp;
end
toc;
A_error=calc_abundance_error(A_gt,A,is_real_dataset);
disp(['A_error= ', num2str(A_error)]);
