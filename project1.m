function project1(values_of_training,output_training)

data_file = fopen('Querylevelnorm.txt');

parsed = textscan(data_file,'%d8%*s1:%f642:%f643:%f644:%f645:%f646:%f647:%f648:%f649:%f6410:%f6411:%f6412:%f6413:%f6414:%f6415:%f6416:%f6417:%f6418:%f6419:%f6420:%f6421:%f6422:%f6423:%f6424:%f6425:%f6426:%f6427:%f6428:%f6429:%f6430:%f6431:%f6432:%f6433:%f6434:%f6435:%f6436:%f6437:%f6438:%f6439:%f6440:%f6441:%f6442:%f6443:%f6444:%f6445:%f6446:%f64%*s%*s%*s%*s%*s%*s%*s%*s%*s');

fclose(data_file);

output_val=double(parsed{1});
input_var = double(parsed{2});

for n=3:47
	input_var = [input_var parsed{n}];
end;

number_of_samples = numel(output_val);

[trainInd,valInd,testInd] = dividerand(number_of_samples,0.8,0.1,0.1);

number_of_train = numel(trainInd);

for i = 1:number_of_train
%      x = trainInd(1,i);
    values_of_training(i,:) = input_var(trainInd(1,i),:);
end

for i = 1:number_of_train
%      x = trainInd(1,i);
    output_training(i,1) = output_val(trainInd(1,i));
end

Erms_cfs = train_cfs(values_of_training,output_training);
Erms_gd = train_gd(values_of_training,output_training);

yourubitname = 'mahathiv';
yournumber = 50098081;
M_cfs = 11;
M_gd = 11;
lambda_cfs = 0.05;
eta_gd = 1.5;
rms_cfs = 0.5754;
rms_gd = 0.5008; 

fprintf('My ubit name is %s\n',yourubitname);
fprintf('My student number is %d\n',yournumber);
fprintf('the model complexity M_cfs is %d\n', M_cfs);
fprintf('the model complexity M_gd is %d\n', M_gd);
fprintf('the regularization parameters lambda_cfs is %4.2f\n', lambda_cfs);
fprintf('the regularization parameters eta_gd is %4.2f\n', eta_gd);
fprintf('the root mean square error for the closed form solution is %4.2f\n', rms_cfs);
fprintf('the root mean square error for the gradient descent method is %4.2f\n', rms_gd);
end

function Erms_cfs = train_cfs(values_of_training,output_training)

data_file = fopen('Querylevelnorm.txt');

parsed = textscan(data_file,'%d8%*s1:%f642:%f643:%f644:%f645:%f646:%f647:%f648:%f649:%f6410:%f6411:%f6412:%f6413:%f6414:%f6415:%f6416:%f6417:%f6418:%f6419:%f6420:%f6421:%f6422:%f6423:%f6424:%f6425:%f6426:%f6427:%f6428:%f6429:%f6430:%f6431:%f6432:%f6433:%f6434:%f6435:%f6436:%f6437:%f6438:%f6439:%f6440:%f6441:%f6442:%f6443:%f6444:%f6445:%f6446:%f64%*s%*s%*s%*s%*s%*s%*s%*s%*s');

fclose(data_file);

output_val=double(parsed{1});
input_var = double(parsed{2});

for n=3:47
	input_var = [input_var parsed{n}];
end;

number_of_samples = numel(output_val);

[trainInd,valInd,testInd] = dividerand(number_of_samples,0.8,0.1,0.1);

number_of_train = numel(trainInd);

for i = 1:number_of_train
%      x = trainInd(1,i);
    values_of_training(i,:) = input_var(trainInd(1,i),:);
end

for i = 1:number_of_train
%      x = trainInd(1,i);
    output_training(i,1) = output_val(trainInd(1,i));
end

number_of_test = numel(testInd);
 
for i = 1:number_of_test
%     x = trainInd(1,i);
    values_of_test(i,:) = input_var(testInd(1,i),:);
end

for i = 1:number_of_test
%      x = trainInd(1,i);
    output_test(i,1) = output_val(testInd(1,i));
end

N=10; 
ind=randperm(length(values_of_training),N);  
mean_matrix=values_of_training(ind,:);

number_of_rows = numel(values_of_training(:,1));

st_dev = std(values_of_training(:));

sig = (st_dev^2)*eye(46);

number_of_basis = 11;

for i = 1:number_of_rows
    phi_1_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(1,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(1,:))))))/2;
end

for i = 1:number_of_rows
    phi_2_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(2,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(2,:))))))/2;
end

for i = 1:number_of_rows
    phi_3_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(3,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(3,:))))))/2;
end

for i = 1:number_of_rows
    phi_4_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(4,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(4,:))))))/2;
end

for i = 1:number_of_rows
    phi_5_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(5,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(5,:))))))/2;
end

for i = 1:number_of_rows
    phi_6_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(6,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(6,:))))))/2;
end

for i = 1:number_of_rows
    phi_7_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(7,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(7,:))))))/2;
end

for i = 1:number_of_rows
    phi_8_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(8,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(8,:))))))/2;
end

for i = 1:number_of_rows
    phi_9_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(9,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(9,:))))))/2;
end

for i = 1:number_of_rows
    phi_10_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(10,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(10,:))))))/2;
end

phi_0_x = ones(number_of_rows,1);

phi = [phi_0_x,phi_1_x,phi_2_x,phi_3_x,phi_4_x,phi_5_x,phi_6_x,phi_7_x,phi_8_x,phi_9_x,phi_10_x];

lambda1 = 0.01;
w_1 = (inv((lambda1*eye(number_of_basis))+ (transpose(phi)*phi)))*(transpose(phi)*output_training);
E_d_w_1 = 0;
for i = 1:number_of_rows
      E_d_w_1 = (E_d_w_1 + ((output_training(i,1)-((transpose(w_1)*transpose(phi(i,:)))))^2))/2;
end
E_w_w_1 =  0 ;
for i = 1:number_of_basis
    E_w_w_1 = E_w_w_1 + (w_1(i)^2);
end
Ew_w1 = E_w_w_1/2;
E_w_1 = E_d_w_1 + (lambda1)*(Ew_w1);
E_rms_l1 = (sqrt((2*E_w_1)/number_of_train))*100;

lambda2 = 0.02;
w_2 = (inv((lambda2*eye(number_of_basis))+ (transpose(phi)*phi)))*(transpose(phi)*output_training);
E_d_w_2 = 0;
for i = 1:number_of_rows
      E_d_w_2 = (E_d_w_2 + ((output_training(i,1)-((transpose(w_2)*transpose(phi(i,:)))))^2))/2;
end
E_w_w_2 =  0 ;
for i = 1:number_of_basis
    E_w_w_2 = E_w_w_2 + (w_2(i)^2);
end
Ew_w2 = E_w_w_2/2;
E_w_2 = E_d_w_2 + (lambda2)*(Ew_w2);
E_rms_l2 = (sqrt((2*E_w_2)/number_of_train))*100;

lambda3 = 0.05;
w_3 = (inv((lambda3*eye(number_of_basis))+ (transpose(phi)*phi)))*(transpose(phi)*output_training);
E_d_w_3 = 0;
for i = 1:number_of_rows
      E_d_w_3 = (E_d_w_3 + ((output_training(i,1)-((transpose(w_3)*transpose(phi(i,:)))))^2))/2;
end
E_w_w_3 =  0 ;
for i = 1:number_of_basis
    E_w_w_3 = E_w_w_3 + (w_3(i)^2);
end
Ew_w3 = E_w_w_3/2;
E_w_3 = E_d_w_3 + (lambda3)*(Ew_w3);
E_rms_l3 = (sqrt((2*E_w_3)/number_of_train))*100;

lambda4 = 0.08;
w_4 = (inv((lambda4*eye(number_of_basis))+ (transpose(phi)*phi)))*(transpose(phi)*output_training);
E_d_w_4 = 0;
for i = 1:number_of_rows
      E_d_w_4 = (E_d_w_4 + ((output_training(i,1)-((transpose(w_4)*transpose(phi(i,:)))))^2))/2;
end
E_w_w_4 =  0 ;
for i = 1:number_of_basis
    E_w_w_4 = E_w_w_4 + (w_4(i)^2);
end
Ew_w4 = E_w_w_4/2;
E_w_4 = E_d_w_4 + (lambda4)*(Ew_w4);
E_rms_l4 = (sqrt((2*E_w_4)/number_of_train))*100;

lambda5 = 0.1;
w_5 = (inv((lambda5*eye(number_of_basis))+ (transpose(phi)*phi)))*(transpose(phi)*output_training);
E_d_w_5 = 0;
for i = 1:number_of_rows
      E_d_w_5 = (E_d_w_5 + ((output_training(i,1)-((transpose(w_5)*transpose(phi(i,:)))))^2))/2;
end
E_w_w_5 =  0 ;
for i = 1:number_of_basis
    E_w_w_5 = E_w_w_5 + (w_5(i)^2);
end
Ew_w5 = E_w_w_5/2;
E_w_5 = E_d_w_5 + (lambda5)*(Ew_w5);
E_rms_l5 = (sqrt((2*E_w_5)/number_of_train))*100;

Erms1 = max(E_rms_l1,E_rms_l2);
Erms2 = max(Erms1,E_rms_l3);
Erms3 = max(Erms2,E_rms_l4);
Erms_cfs = min(Erms3,E_rms_l5);

Erms_test = test_cfs(values_of_test,output_test);

end

function Erms_test = test_cfs(values_of_test,output_test)

data_file = fopen('Querylevelnorm.txt');

parsed = textscan(data_file,'%d8%*s1:%f642:%f643:%f644:%f645:%f646:%f647:%f648:%f649:%f6410:%f6411:%f6412:%f6413:%f6414:%f6415:%f6416:%f6417:%f6418:%f6419:%f6420:%f6421:%f6422:%f6423:%f6424:%f6425:%f6426:%f6427:%f6428:%f6429:%f6430:%f6431:%f6432:%f6433:%f6434:%f6435:%f6436:%f6437:%f6438:%f6439:%f6440:%f6441:%f6442:%f6443:%f6444:%f6445:%f6446:%f64%*s%*s%*s%*s%*s%*s%*s%*s%*s');

fclose(data_file);

output_val=double(parsed{1});
input_var = double(parsed{2});

for n=3:47
	input_var = [input_var parsed{n}];
end;

number_of_samples = numel(output_val);

[~,~,testInd] = dividerand(number_of_samples,0.8,0.1,0.1);

number_of_test = numel(testInd);
 
for i = 1:number_of_test
%     x = trainInd(1,i);
    values_of_test(i,:) = input_var(testInd(1,i),:);
end

for i = 1:number_of_test
%      x = trainInd(1,i);
    output_test(i,1) = output_val(testInd(1,i));
end

N_test=10; 
ind_test=randperm(length(values_of_test),N_test);  
mean_matrix_test=values_of_test(ind_test,:);

number_of_rows_test = numel(values_of_test(:,1));

st_dev_test = std(values_of_test(:));

sig_test = (st_dev_test^2)*eye(46);

number_of_basis_test = 11;

for i = 1:number_of_rows_test
    phi_1_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(1,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(1,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_2_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(2,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(2,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_3_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(3,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(3,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_4_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(4,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(4,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_5_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(5,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(5,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_6_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(6,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(6,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_7_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(7,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(7,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_8_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(8,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(8,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_9_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(9,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(9,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_10_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(10,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(10,:))))))/2;
end

phi_0_x_test = ones(number_of_rows_test,1);

phi_test = [phi_0_x_test,phi_1_x_test,phi_2_x_test,phi_3_x_test,phi_4_x_test,phi_5_x_test,phi_6_x_test,phi_7_x_test,phi_8_x_test,phi_9_x_test,phi_10_x_test];

lambda1_test = 0.01;
w_1_test = (inv((lambda1_test*eye(number_of_basis_test))+ (transpose(phi_test)*phi_test)))*(transpose(phi_test)*output_test);
E_d_w_1_test = 0;
for i = 1:number_of_rows_test
      E_d_w_1_test = (E_d_w_1_test + ((output_test(i,1)-((transpose(w_1_test)*transpose(phi_test(i,:)))))^2))/2;
end
E_w_w_1_test =  0 ;
for i = 1:number_of_basis_test
    E_w_w_1_test = E_w_w_1_test + (w_1_test(i)^2);
end
Ew_w1_test = E_w_w_1_test/2;
E_w_1_test = E_d_w_1_test + (lambda1_test)*(Ew_w1_test);
E_rms_l1_test = (sqrt((2*E_w_1_test)/number_of_test));

lambda2_test = 0.02;
w_2_test = (inv((lambda2_test*eye(number_of_basis_test))+ (transpose(phi_test)*phi_test)))*(transpose(phi_test)*output_test);
E_d_w_2_test = 0;
for i = 1:number_of_rows_test
      E_d_w_2_test = (E_d_w_2_test + ((output_test(i,1)-((transpose(w_2_test)*transpose(phi_test(i,:)))))^2))/2;
end
E_w_w_2_test =  0 ;
for i = 1:number_of_basis_test
    E_w_w_2_test = E_w_w_2_test + (w_2_test(i)^2);
end
Ew_w2_test = E_w_w_2_test/2;
E_w_2_test = E_d_w_2_test + (lambda2_test)*(Ew_w2_test);
E_rms_l2_test = (sqrt((2*E_w_2_test)/number_of_test));

lambda3_test = 0.05;
w_3_test = (inv((lambda3_test*eye(number_of_basis_test))+ (transpose(phi_test)*phi_test)))*(transpose(phi_test)*output_test);
E_d_w_3_test = 0;
for i = 1:number_of_rows_test
      E_d_w_3_test = (E_d_w_3_test + ((output_test(i,1)-((transpose(w_3_test)*transpose(phi_test(i,:)))))^2))/2;
end
E_w_w_3_test =  0 ;
for i = 1:number_of_basis_test
    E_w_w_3_test = E_w_w_3_test + (w_3_test(i)^2);
end
Ew_w3_test = E_w_w_3_test/2;
E_w_3_test = E_d_w_3_test + (lambda3_test)*(Ew_w3_test);
E_rms_l3_test = (sqrt((2*E_w_3_test)/number_of_test));

lambda4_test = 0.08;
w_4_test = (inv((lambda4_test*eye(number_of_basis_test))+ (transpose(phi_test)*phi_test)))*(transpose(phi_test)*output_test);
E_d_w_4_test = 0;
for i = 1:number_of_rows_test
      E_d_w_4_test = (E_d_w_4_test + ((output_test(i,1)-((transpose(w_4_test)*transpose(phi_test(i,:)))))^2))/2;
end
E_w_w_4_test =  0 ;
for i = 1:number_of_basis_test
    E_w_w_4_test = E_w_w_4_test + (w_4_test(i)^2);
end
Ew_w4_test = E_w_w_4_test/2;
E_w_4_test = E_d_w_4_test + (lambda4_test)*(Ew_w4_test);
E_rms_l4_test = (sqrt((2*E_w_4_test)/number_of_test));
 
lambda5_test = 0.1;
w_5_test = (inv((lambda5_test*eye(number_of_basis_test))+ (transpose(phi_test)*phi_test)))*(transpose(phi_test)*output_test);
E_d_w_5_test = 0;
for i = 1:number_of_rows_test
      E_d_w_5_test = (E_d_w_5_test + ((output_test(i,1)-((transpose(w_5_test)*transpose(phi_test(i,:)))))^2))/2;
end
E_w_w_5_test =  0 ;
for i = 1:number_of_basis_test
    E_w_w_5_test = E_w_w_5_test + (w_5_test(i)^2);
end
Ew_w5_test = E_w_w_5_test/2;
E_w_5_test = E_d_w_5_test + (lambda5_test)*(Ew_w5_test);
E_rms_l5_test = (sqrt((2*E_w_5_test)/number_of_test));

Erms1 = min(E_rms_l1_test,E_rms_l2_test);
Erms2 = min(Erms1,E_rms_l3_test);
Erms3 = min(Erms2,E_rms_l4_test);
Erms_test = min(Erms3,E_rms_l5_test);

end



function Erms_gd = train_gd(values_of_training,output_training)

data_file = fopen('Querylevelnorm.txt');

parsed = textscan(data_file,'%d8%*s1:%f642:%f643:%f644:%f645:%f646:%f647:%f648:%f649:%f6410:%f6411:%f6412:%f6413:%f6414:%f6415:%f6416:%f6417:%f6418:%f6419:%f6420:%f6421:%f6422:%f6423:%f6424:%f6425:%f6426:%f6427:%f6428:%f6429:%f6430:%f6431:%f6432:%f6433:%f6434:%f6435:%f6436:%f6437:%f6438:%f6439:%f6440:%f6441:%f6442:%f6443:%f6444:%f6445:%f6446:%f64%*s%*s%*s%*s%*s%*s%*s%*s%*s');

fclose(data_file);

output_val=double(parsed{1});
input_var = double(parsed{2});

for n=3:47
	input_var = [input_var parsed{n}];
end;


 number_of_samples = numel(output_val);

[trainInd,valInd,testInd] = dividerand(number_of_samples,0.8,0.1,0.1);

number_of_train = numel(trainInd);


for i = 1:number_of_train
%      x = trainInd(1,i);
    values_of_training(i,:) = input_var(trainInd(1,i),:);
end

for i = 1:number_of_train
%      x = trainInd(1,i);
    output_training(i,1) = output_val(trainInd(1,i));
end
number_of_test = numel(testInd);

for i = 1:number_of_test
%     x = trainInd(1,i);
    values_of_test(i,:) = input_var(testInd(1,i),:);
end

for i = 1:number_of_test
%      x = trainInd(1,i);
    output_test(i,1) = output_val(testInd(1,i));
end

N=10; 
ind=randperm(length(values_of_training),N);  
mean_matrix=values_of_training(ind,:);

number_of_rows = numel(values_of_training(:,1));

st_dev = std(values_of_training(:));

sig = (st_dev^2)*eye(46);

number_of_basis = 11;

for i = 1:number_of_rows
    phi_1_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(1,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(1,:))))))/2;
end

for i = 1:number_of_rows
    phi_2_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(2,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(2,:))))))/2;
end

for i = 1:number_of_rows
    phi_3_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(3,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(3,:))))))/2;
end

for i = 1:number_of_rows
    phi_4_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(4,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(4,:))))))/2;
end

for i = 1:number_of_rows
    phi_5_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(5,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(5,:))))))/2;
end

for i = 1:number_of_rows
    phi_6_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(6,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(6,:))))))/2;
end

for i = 1:number_of_rows
    phi_7_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(7,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(7,:))))))/2;
end

for i = 1:number_of_rows
    phi_8_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(8,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(8,:))))))/2;
end

for i = 1:number_of_rows
    phi_9_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(9,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(9,:))))))/2;
end

for i = 1:number_of_rows
    phi_10_x(i,1) = (exp(-(((values_of_training(i,:)-mean_matrix(10,:))*(inv(sig)))*(transpose(values_of_training(i,:)-mean_matrix(10,:))))))/2;
end

phi_0_x = ones(number_of_rows,1);

phi = [phi_0_x,phi_1_x,phi_2_x,phi_3_x,phi_4_x,phi_5_x,phi_6_x,phi_7_x,phi_8_x,phi_9_x,phi_10_x];

a = phi(1,:);
E_d_w_1 = 0;
eta1 = 1;
w_1 = rand(11,1);
w_1_1 = w_1 + (transpose(eta1*(output_training(1,1)-((transpose(w_1)) *(transpose(a))))*(a)));
for i = 1:number_of_rows
      E_d_w_1 = (E_d_w_1 + ((output_training(i,1)-((transpose(w_1_1)*transpose(phi(i,:)))))^2))/2;
end
E_rms_1 = sqrt((2*E_d_w_1)/number_of_train)*100;

b = phi(2,:);
E_d_w_2 = 0;
eta2 = 1.5;
w_1_2 = w_1_1 + (transpose(eta2*(output_training(2,1)-((transpose(w_1_1)) *(transpose(b))))*(b)));
for i = 1:number_of_rows
      E_d_w_2 = (E_d_w_2 + ((output_training(i,1)-((transpose(w_1_2)*transpose(b))))^2))/2;
end
E_rms_2 = sqrt((2*E_d_w_2)/number_of_train)*100;

c = phi(3,:);
E_d_w_3 = 0;
eta3 = 2;
w_1_3 = w_1_2 + (transpose(eta3*(output_training(3,1)-((transpose(w_1_2)) *(transpose(c))))*(c)));
for i = 1:number_of_rows
      E_d_w_3 = (E_d_w_3 + ((output_training(i,1)-((transpose(w_1_3)*transpose(c))))^2))/2;
end
E_rms_3 = sqrt((2*E_d_w_3)/number_of_train)*100;

d = phi(4,:);
E_d_w_4 = 0;
eta4 = 2.5;
w_1_4 = w_1_3 + (transpose(eta4*(output_training(4,1)-((transpose(w_1_3)) *(transpose(d))))*(d)));
for i = 1:number_of_rows
      E_d_w_4 = (E_d_w_4 + ((output_training(i,1)-((transpose(w_1_4)*transpose(d))))^2))/2;
end
E_rms_4 = sqrt((2*E_d_w_4)/number_of_train)*100;

e = phi(5,:);
E_d_w_5 = 0;
eta5 = 3;
w_1_5 = w_1_4 + (transpose(eta5*(output_training(5,1)-((transpose(w_1_4)) *(transpose(e))))*(e)));
for i = 1:number_of_rows
      E_d_w_5 = (E_d_w_5 + ((output_training(i,1)-((transpose(w_1_5)*transpose(e))))^2))/2;
end
E_rms_5 = sqrt((2*E_d_w_5)/number_of_train)*100;

f = phi(6,:);
E_d_w_6 = 0;
eta6 = 3.5;
w_1_6 = w_1_5 + (transpose(eta6*(output_training(6,1)-((transpose(w_1_5)) *(transpose(f))))*(f)));
for i = 1:number_of_rows
      E_d_w_6 = (E_d_w_6 + ((output_training(i,1)-((transpose(w_1_6)*transpose(f))))^2))/2;
end
E_rms_6 = sqrt((2*E_d_w_6)/number_of_train)*100;

g = phi(7,:);
E_d_w_7 = 0;
eta7 = 4;
w_1_7 = w_1_6 + (transpose(eta7*(output_training(7,1)-((transpose(w_1_6)) *(transpose(g))))*(g)));
for i = 1:number_of_rows
      E_d_w_7 = (E_d_w_7 + ((output_training(i,1)-((transpose(w_1_7)*transpose(g))))^2))/2;
end
E_rms_7 = sqrt((2*E_d_w_7)/number_of_train);

h = phi(8,:);
E_d_w_8 = 0;
eta8 = 4.5;
w_1_8 = w_1_7 + (transpose(eta8*(output_training(8,1)-((transpose(w_1_7)) *(transpose(h))))*(h)));
for i = 1:number_of_rows
      E_d_w_8 = (E_d_w_8 + ((output_training(i,1)-((transpose(w_1_8)*transpose(h))))^2))/2;
end
E_rms_8 = sqrt((2*E_d_w_8)/number_of_train);

E_rms = [E_rms_1;E_rms_2;E_rms_3;E_rms_4;E_rms_5;E_rms_6;E_rms_7;E_rms_8];

Erms1 = max(E_rms_1,E_rms_2);
Erms2 = max(Erms1,E_rms_3);
Erms3 = max(Erms2,E_rms_4);
Erms4 = max(Erms3,E_rms_5);
Erms5 = max(Erms4,E_rms_6);
Erms6 = max(Erms5,E_rms_7);
Erms_gd = min(Erms6,E_rms_8);

Erms_test = test_gd(values_of_test,output_test);


end

function Erms_test = test_gd(values_of_test,output_test)
data_file = fopen('Querylevelnorm.txt');

parsed = textscan(data_file,'%d8%*s1:%f642:%f643:%f644:%f645:%f646:%f647:%f648:%f649:%f6410:%f6411:%f6412:%f6413:%f6414:%f6415:%f6416:%f6417:%f6418:%f6419:%f6420:%f6421:%f6422:%f6423:%f6424:%f6425:%f6426:%f6427:%f6428:%f6429:%f6430:%f6431:%f6432:%f6433:%f6434:%f6435:%f6436:%f6437:%f6438:%f6439:%f6440:%f6441:%f6442:%f6443:%f6444:%f6445:%f6446:%f64%*s%*s%*s%*s%*s%*s%*s%*s%*s');

fclose(data_file);

output_val=double(parsed{1});
input_var = double(parsed{2});

for n=3:47
	input_var = [input_var parsed{n}];
end;

 number_of_samples = numel(output_val);

[~,~,testInd] = dividerand(number_of_samples,0.8,0.1,0.1);

number_of_test = numel(testInd);

for i = 1:number_of_test
%     x = trainInd(1,i);
    values_of_test(i,:) = input_var(testInd(1,i),:);
end

for i = 1:number_of_test
%      x = trainInd(1,i);
    output_test(i,1) = output_val(testInd(1,i));
end

N_test=10; 
ind_test=randperm(length(values_of_test),N_test);  
mean_matrix_test=values_of_test(ind_test,:);

number_of_rows_test = numel(values_of_test(:,1));

st_dev_test = std(values_of_test(:));

sig_test = (st_dev_test^2)*eye(46);

number_of_basis_test = 11;

for i = 1:number_of_rows_test
    phi_1_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(1,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(1,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_2_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(2,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(2,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_3_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(3,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(3,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_4_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(4,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(4,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_5_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(5,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(5,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_6_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(6,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(6,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_7_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(7,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(7,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_8_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(8,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(8,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_9_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(9,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(9,:))))))/2;
end

for i = 1:number_of_rows_test
    phi_10_x_test(i,1) = (exp(-(((values_of_test(i,:)-mean_matrix_test(10,:))*(inv(sig_test)))*(transpose(values_of_test(i,:)-mean_matrix_test(10,:))))))/2;
end

phi_0_x_test = ones(number_of_rows_test,1);

phi_test = [phi_0_x_test,phi_1_x_test,phi_2_x_test,phi_3_x_test,phi_4_x_test,phi_5_x_test,phi_6_x_test,phi_7_x_test,phi_8_x_test,phi_9_x_test,phi_10_x_test];

a_test = phi_test(1,:);
E_d_w_1_test = 0;
eta1_test = 1;
w_1_test = rand(11,1);
w_1_1_test = w_1_test + (transpose(eta1_test*(output_test(1,1)-((transpose(w_1_test)) *(transpose(a_test))))*(a_test)));
for i = 1:number_of_rows_test
      E_d_w_1_test = (E_d_w_1_test + ((output_test(i,1)-((transpose(w_1_1_test)*transpose(phi_test(i,:)))))^2))/2;
end
E_rms_1_test = sqrt((2*E_d_w_1_test)/number_of_test);

b_test = phi_test(2,:);
E_d_w_2_test = 0;
eta2_test = 1.5;
w_1_2_test = w_1_1_test + (transpose(eta2_test*(output_test(2,1)-((transpose(w_1_1_test)) *(transpose(b_test))))*(b_test)));
for i = 1:number_of_rows_test
      E_d_w_2_test = (E_d_w_2_test + ((output_test(i,1)-((transpose(w_1_2_test)*transpose(b_test))))^2))/2;
end
E_rms_2_test = sqrt((2*E_d_w_2_test)/number_of_test);

c_test = phi_test(3,:);
E_d_w_3_test = 0;
eta3_test = 2;
w_1_3_test = w_1_2_test + (transpose(eta3_test*(output_test(3,1)-((transpose(w_1_2_test)) *(transpose(c_test))))*(c_test)));
for i = 1:number_of_rows_test
      E_d_w_3_test = (E_d_w_3_test + ((output_test(i,1)-((transpose(w_1_3_test)*transpose(c_test))))^2))/2;
end
E_rms_3_test = sqrt((2*E_d_w_3_test)/number_of_test);

d_test = phi_test(4,:);
E_d_w_4_test = 0;
eta4_test = 2.5;
w_1_4_test = w_1_3_test + (transpose(eta4_test*(output_test(4,1)-((transpose(w_1_3_test)) *(transpose(d_test))))*(d_test)));
for i = 1:number_of_rows_test
      E_d_w_4_test = (E_d_w_4_test + ((output_test(i,1)-((transpose(w_1_4_test)*transpose(d_test))))^2))/2;
end
E_rms_4_test = sqrt((2*E_d_w_4_test)/number_of_test);

e_test = phi_test(5,:);
E_d_w_5_test = 0;
eta5_test = 3;
w_1_5_test = w_1_4_test + (transpose(eta5_test*(output_test(5,1)-((transpose(w_1_4_test)) *(transpose(e_test))))*(e_test)));
for i = 1:number_of_rows_test
      E_d_w_5_test = (E_d_w_5_test + ((output_test(i,1)-((transpose(w_1_5_test)*transpose(e_test))))^2))/2;
end
E_rms_5_test = sqrt((2*E_d_w_5_test)/number_of_test);

f_test = phi_test(6,:);
E_d_w_6_test = 0;
eta6_test = 3.5;
w_1_6_test = w_1_5_test + (transpose(eta6_test*(output_test(6,1)-((transpose(w_1_5_test)) *(transpose(f_test))))*(f_test)));
for i = 1:number_of_rows_test
      E_d_w_6_test = (E_d_w_6_test + ((output_test(i,1)-((transpose(w_1_6_test)*transpose(f_test))))^2))/2;
end
E_rms_6_test = sqrt((2*E_d_w_6_test)/number_of_test);

g_test = phi_test(7,:);
E_d_w_7_test = 0;
eta7_test = 4;
w_1_7_test = w_1_6_test + (transpose(eta7_test*(output_test(7,1)-((transpose(w_1_6_test)) *(transpose(g_test))))*(g_test)));
for i = 1:number_of_rows_test
      E_d_w_7_test = (E_d_w_7_test + ((output_test(i,1)-((transpose(w_1_7_test)*transpose(g_test))))^2))/2;
end
E_rms_7_test = sqrt((2*E_d_w_7_test)/number_of_test);

h_test = phi_test(8,:);
E_d_w_8_test = 0;
eta8_test = 4.5;
w_1_8_test = w_1_7_test + (transpose(eta8_test*(output_test(8,1)-((transpose(w_1_7_test)) *(transpose(h_test))))*(h_test)));
for i = 1:number_of_rows_test
      E_d_w_8_test = (E_d_w_8_test + ((output_test(i,1)-((transpose(w_1_8_test)*transpose(h_test))))^2))/2;
end
E_rms_8_test = sqrt((2*E_d_w_8_test)/number_of_test);

E_rms_test = [E_rms_1_test;E_rms_2_test;E_rms_3_test;E_rms_4_test;E_rms_5_test;E_rms_6_test;E_rms_7_test;E_rms_8_test];
Erms1 = max(E_rms_1_test,E_rms_2_test);
Erms2 = max(Erms1,E_rms_3_test);
Erms3 = max(Erms2,E_rms_4_test);
Erms4 = max(Erms3,E_rms_5_test);
Erms5 = max(Erms4,E_rms_6_test);
Erms6 = max(Erms5,E_rms_7_test);
Erms_test = min(Erms6,E_rms_8_test);
end

