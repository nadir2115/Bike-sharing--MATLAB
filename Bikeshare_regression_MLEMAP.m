%Nadir Nibras Bike usage Regression problem
clear; clc; close all;
cd 'C:\Users\nadir\Documents\Machine Learning\Bikesharing-prediction-ML-Regression'; 
Dataset = csvread('bikehour.csv', 1,0);
[m n]=size(Dataset);                                        % find matrix size

Dataset(:,n)= log10(Dataset(:,n)+1);

Dataset = Dataset(randperm(m),:);                           % shuffle randomly
X= Dataset(:,1:(n-1));                                      % separate features
y= Dataset(:,n);


nl=1;     X = X.^nl;                                        % Polynomial feature option
Xtest= X(ceil(m*0.9):m,:);                                  % Test features
ytest= y(ceil(m*0.9):m);                                    % Test output

tic
%% MLE
p_error= 10000000000;
for k=1:100
rowindices= randperm(floor(m*0.9));
TDataset = Dataset(rowindices, :);
Xtrain= TDataset(1:floor(m*0.8),1:n-1); 
ytrain= TDataset(1:floor(m*0.8),n);                       % Create train set
Xvalidation= TDataset(ceil(m*0.8):floor(m*0.9),1:n-1);      % Create Validation sets
yvalidation= TDataset(ceil(m*0.8):floor(m*0.9),n); 
XtXtrain= (transpose(Xtrain))*Xtrain;
wMLEk = inv(XtXtrain)*(transpose(Xtrain))*ytrain;
ymle_est= Xvalidation*wMLEk; 
MSE_mle = immse(ymle_est,yvalidation);                      % MSE for MLE W
if MSE_mle<p_error
    wMLE=wMLEk;
    p_error= MSE_mle
end
end

ymle_est= Xtest*wMLE; 
toc


%% MAP

tic
previous_error= 1000000000;
for i=1:100
    for j = 0:100 %reduced from 1000
    I = eye(n-1); 
    mw= 0.01*j; tausq= 0.1*i; lambda = mw/tausq;            % increased from 0.001
    rowindices= randperm(floor(m*0.9));
    Dataset = Dataset(rowindices, :);                       % Scramble Dataset
    Xvalidation= Dataset(ceil(m*0.8):floor(m*0.9),1:n-1); 
    yvalidation= Dataset(ceil(m*0.8):floor(m*0.9),n);       % Create Validation set
    Xtrain= Dataset(1:floor(m*0.8),1:n-1);                  % Create train set
    ytrain= Dataset(1:floor(m*0.8),n); 
    XtXtrain= (transpose(Xtrain))*Xtrain;
    wMAP= (inv(XtXtrain+(lambda.*I)))*((lambda*mw)+((transpose(Xtrain))*ytrain));
    ymap_est_val= Xvalidation*wMAP;
    MSE_map(i) = immse(ymap_est_val,yvalidation); %MSE for MAP W
        if MSE_map(i)<previous_error
            i
            ideal_tausq= tausq
            ideal_lambda=lambda
            ideal_mw=mw
            previous_error=MSE_map(i)
            ideal_wMAP= wMAP;
        end
    end
end

ymap_est= Xtest*ideal_wMAP;

toc

%%
ytest=10.^(ytest)-1;
ymle_est=10.^(ymle_est)-1;
ymap_est=10.^(ymap_est)-1;
%% 
mean_error_MLE= mean(abs(ytest-ymle_est))
median_error_MLE= median(abs(ytest-ymle_est))
mean_error_MAP= mean(abs(ytest-ymap_est))
median_error_MAP= median(abs(ytest-ymap_est))



%for reference
Dataset_regression(:,1)= ytest; 
Dataset_regression(:,2)= ymap_est; 
Dataset_regression(:,3)= ymle_est;


Ym= [ymap_est ymle_est ytest];
csvwrite('Predictions_separate_cas_reg.csv', Ym);

i=300:400;
figure
plot(i,ymap_est(i))
hold on
plot(i, ytest(i))
hold on 
plot(i, abs(ytest(i)-ymap_est(i)))
ylabel("Bike count")
legend('MAP estimate','Actual count', 'Error for prediction')


figure
plot(i,ymap_est(i))
hold on
plot(i, ymle_est(i))
hold on
plot(i, ytest(i))
ylabel("Bike count")
legend('MAP estimate','MLE estimate','Actual count')

%Scatter plot of ytest and MAP estimate
figure
scatter(ytest,ymap_est)
hold on
plot(ytest, ytest)
xlim([0 400])
rows=find(ytest<400);
[p,s]=polyfit(ytest(rows),ymap_est(rows),1);
[yfit,dy]= polyconf(p,ytest(rows),s,'predopt', 'curve');
hold on
line(ytest(rows),yfit,'color','g')
