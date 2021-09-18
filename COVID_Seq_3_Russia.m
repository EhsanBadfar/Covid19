function Pre_Seq=COVID_Seq()
clc;
clear all;
close all;
data = sortrows(readtable('E:\PhD theses\new\Article#7-Covid\Codes\Data-1-Russia.xlsx'));
H=3;
data=table2array(data(:,H));
Name=["Date","The recovery rate","The death rate","The infected people",...
    "The people who don’t need hospitalization",...
    "The people with strong symptomatic condition", ...
    "The people to be hospitalized","New death"];
Column=["Date","Normalized Rate(%)","Normalized Rate(%)","Normalized Casses"...
    ,"Normalized Casses","Normalized Casses","Normalized Casses","Normalized Casses"];

numTimeStepsTrain = floor(0.7*size(data,1));

dataTrain = data(1:numTimeStepsTrain+1,:);
dataTest = data(numTimeStepsTrain+1:end,:);
Max = max(dataTrain);
Min = min(dataTrain);

dataTrainStandardized = (dataTrain - Min) ./ (Max-Min);
XTrain = dataTrainStandardized(1:end-1,:);
YTrain = dataTrainStandardized(2:end,:);
numFeatures = size(XTrain,2);
numResponses = size(YTrain,2);
numHiddenUnits = 30;

t = datetime(2020,7,31) + caldays(1:size(dataTrain,1));

% bar(t,dataTrain);
% ylabel(Column(H));
% title(Name(H));


layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',3500, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.1, ...
    'Verbose',1, ...
    'Shuffle','never');
net = trainNetwork(XTrain',YTrain',layers,options);
Max_T=max(dataTest);
Min_T=min(dataTest);
MM=Max_T-Min_T;
dataTestStandardized = (dataTest - Min_T) ./ (Max_T-Min_T);
XTest = dataTestStandardized(1:end-1,:);
net = predictAndUpdateState(net,XTrain');
[net,YPred] = predictAndUpdateState(net,YTrain(end,:)');


numTimeStepsTest = size(XTest(:,1),1);
for i = 2:numTimeStepsTest
        [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1));
end
YPred_Test = predict(net,XTest');
YPred_Test=double(YPred_Test');
MMM=Max-Min;
YY= (MM*YPred_Test+Min_T);

YTest = dataTest(2:end,:);
% rmse = sqrt(mean((YY-YTest).^2));
rmse = mse(YPred_Test,XTest);

T=datetime(t(end)) + caldays(1:size(dataTest,1));
t=1:size(dataTest,1);


% figure
% subplot(2,1,1)
% bar(t(1:end-1),dataTrain(1:end-1))
% hold on
% bar(T,[data(numTimeStepsTrain) YY(1:end)'],'r')
% ylabel(Column(H));
% title(Name(H));
% hold off
% subplot(2,1,2)
% plot(T(2:end),XTest)
plot(t(2:end),XTest)

hold on
% plot(T(2:end),YPred_Test,'.-')
plot(t(2:end),YPred_Test,'.-')

hold off
legend(["Observed" "Forecast"])
% ylabel(Column(H));
ylabel('Mortality Rate')
xlabel('Time (dayes)')
% title(Name(H));
save the_death_rate_Russia.mat XTest YPred_Test


disp('finish')