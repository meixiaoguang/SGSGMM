function [ output_args ] = test_gmm_ex( input_args )
%TEST_GMM_HU_EX Summary of this function goes here
%   Detailed explanation goes here
close all;
clc;
dataset = ('Synthenic_new.mat');
[endmembers,I,Y,R_gt,A_gt,names,wl] = prepare_supervised_unmixing(dataset);
[rows,cols,B] = size(I);
I_org=I;
D = 0.01^2 * eye(B);
options.beta1 =1;
options.beta2 =1;
options.beta3=1;
options.show_fig = 1;
options.names = names;
options.D = D;
options.project_mode = 'image';
options.convergence_thresh = 0.0001;

disp('原始结果');
options.beta1 =0.1;
options.beta2 =0.1;
options.beta3=0.1;
[A,R,w_jk,mu_jk,sigma_jk,extra] = gmm_hu_ex(I, endmembers, options);
E = gmm_hu_endmember(I,A,D,w_jk,mu_jk,sigma_jk);
[rows,cols,B] = size(I);
A_gt=reshape(A_gt,[rows*cols,size(w_jk,2)]);
mdiff(A,A_gt);
show_abundances(A,rows,cols);
replay_scatter_abund(extra.frames_scatter, extra.frames_abund);


%% 
disp('超像素分割');
[M1, N1, C1] = size(I);
p1 = 1; % number of principle components
[X_pca] = pca(reshape(Y, M1*N1, C1), p1);
img = im2uint8(mat2gray(reshape(X_pca', M1, N1, p1)));

segnum = 4; 
labels = mex_ers(double(img), segnum);
%grey_img = im2uint8(mat2gray(Y(:,:,1)));



figure;
 imshow(img,[]);
[height width] = size(img);
 [bmap] = seg2bmap(labels,width,height);
 bmapOnImg = img;
 idx = find(bmap>0);
 timg = img;
 timg(idx) = 255;
bmapOnImg(:,:,2) = timg;
bmapOnImg(:,:,1) = img;
bmapOnImg(:,:,3) = img;
 
figure;
imshow(bmapOnImg,[]);
Results_segment = seg_im_class(I, labels);
Num=size(Results_segment.Y,2);
A_segment=cell(1,4);
 E_segment=cell(1,4);
%  load('result_gmm_segment_muufl_gulfport.mat');
 %% 
for i=1:Num %这里重点要改
    options.beta1=0.1;    
    options.beta2=0.1;  
    options.beta3=0.1; 
    I_temp = Results_segment.Y{1,i};
    num_segment=i;
    [A_temp,R_temp,w_jk_temp,mu_jk_temp,sigma_jk_temp,extra_temp] = gmm_hu_ex(I_temp, endmembers, options);
 %  A(Results_segment.index{i},:) = A_temp;
    [abs_err(i),rel_err(i)]=  mdiff(A_temp,A_gt(Results_segment.index{i},:));
%    E = gmm_hu_endmember(I_temp,A_temp,D,w_jk_temp,mu_jk_temp,sigma_jk_temp);
% end
  %save('result_gmm_segment_muufl_gulfport.mat','A1');
%  [rows,cols,N]=size(I);
% A_gt=reshape(A_gt,[rows*cols,4]);
% mdiff(A1,A_gt);
% show_abundances(A1,rows,cols);
end
% abs_err_count{k}=abs_err;
% rel_err_count{k}=rel_err;
abs_err_mean=sum(abs_err)/Num;
rel_err_mean=sum(rel_err)/Num;
 save('result_gmm_M=4计数.mat','abs_err_mean','rel_err_mean','options','abs_err','rel_err');
 %% 

% for i=i:Num %这里重点要改
% i=1;
%  abs_err_min=1;
% rel_err_min=1;    
% % disp('原始结果');
% for beta1 =0:0.5:5
%     for beta2=0:0.1:0.5
%         for beta3=0:0.1:0.5
%     options.beta1=beta1;    
%     options.beta2=beta2;  
%     options.beta3=beta3;  
%   I_temp = Results_segment.Y{1,i};
%   [A_temp,R_temp,w_jk_temp,mu_jk_temp,sigma_jk_temp,extra_temp] = gmm_hu_ex(I_temp,endmembers,options);
% %  A(Results_segment.index{i},:) = A_temp;
%   [abs_err,rel_err]=mdiff(A_temp,A_gt(Results_segment.index{i},:));
%   if abs_err_min>abs_err && rel_err_min>rel_err
%     abs_err_min=abs_err;
%     rel_err_min=rel_err;
%     save('result_gmm_M=2new.mat','beta1','beta2','beta3','abs_err_min','rel_err_min');
%     close all;
%   end
%         end
%     end
% end
% end

   %E = gmm_hu_endmember(I1_temp,A_temp,D,w_jk_temp,mu_jk_temp,sigma_jk_temp);
% end
%  save('result_gmm_segment_B.mat','A1');
%  [rows,cols,N]=size(I);
% A_gt=reshape(A_gt,[rows*cols,4]);
% mdiff(A1,A_gt);
% show_abundances(A1,rows,cols);



%% 
disp('分割结果');
Result_segment=cell(1,4);
Result_segment{1,1}=I(1:30,1:30,:);
Result_segment{1,2}=I(31:60,1:30,:);
Result_segment{1,3}=I(1:30,31:60,:);
Result_segment{1,4}=I(31:60,31:60,:);
num=size(Result_segment,2);
A_segment=cell(1,4);
E_segment=cell(1,4);
for i=1:num
    options.beta1 =1;
    options.beta2 =0.1;
    options.beta3=0;
     I_temp=Result_segment{1,i};
     M=4;
     [A_temp,R_temp,w_jk_temp,mu_jk_temp,sigma_jk_temp,extra_temp] = gmm_hu_ex(I_temp,endmembers,options);
     mdiff(A_temp,A_gt(1:30,1:30,:));
     A_segment{1,i} = reshape(A_temp,60,50,4);
end
save('result_gmm_jqw1.mat','A_segment');
load('result_gmm_jqw1.mat');
load('toy_image_end_var_3_jqw1.mat');
A1=cat(1,A_segment{1},A_segment{2});
A2=cat(1,A_segment{3},A_segment{4});
A3=cat(2,A1,A2);
mdiff(A3,A_gt);

%% 
%% 


        
% disp('原始结果');
% options.beta1 =1.0;
% options.beta2 =0.15;
% options.beta3=0;
% [A,R,w_jk,mu_jk,sigma_jk,extra] = gmm_hu_ex(I, endmembers, options);
% E = gmm_hu_endmember(I,A,D,w_jk,mu_jk,sigma_jk);
% save('result_gmm.mat','A','R','E','w_jk','mu_jk','sigma_jk');
% [rows,cols,B] = size(I);
% A_gt=reshape(A_gt,[rows*cols,4]);
% mdiff(A,A_gt);
% show_abundances(A,rows,cols);
% replay_scatter_abund(extra.frames_scatter, extra.frames_abund);


end

