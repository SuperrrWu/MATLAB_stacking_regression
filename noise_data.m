%% create data
X=[-10:0.1:10];
Y=X.^2;
randn('seed',1);
%add some noise
addictives=randn(length(X),1)';
Y=Y+addictives;

%add some noise
train_X=X(1:190);
train_Y=Y(1:190);
test_X=X(191:200);
test_Y=Y(191:200);
plot(X,Y);
hold on;


%% using 2-order
[p2, S2 ]= polyfit(train_X,train_Y,2);
y2=polyval(p2,train_X,S2);
hold on;

%% using 3-order
[p3, S3 ]= polyfit(train_X,train_Y,3);
y3=polyval(p3,train_X,S3);


%% using 4-order
[p4, S4 ]= polyfit(train_X,train_Y,4);
y4=polyval(p4,train_X,S4);


%% define network
net=newff(train_X,train_Y,[3,1]);
net.trainParam.epochs=1000;
net.trainParam.goal=1e-3;
trainednet=train(net,train_X,train_Y);
ypre=sim(trainednet,train_X);


%% Get stacking training set
X_train=[y2;y3;y4;ypre]'
net=newff(X_train',train_Y,[3,1]);
net.trainParam.epochs=1000;
net.trainParam.goal=1e-6;
stacking_net=train(net,X_train',train_Y) ;


%% using stacking regression to predict on the test set
stacking_y2=polyval(p2,test_X,S2);
stacking_y3=polyval(p3,test_X,S3);
stacking_y4=polyval(p4,test_X,S4);
ypre=sim(trainednet,test_X);
stacking_train=[stacking_y2;stacking_y3;stacking_y4;ypre]
stacking_pre=sim(stacking_net,stacking_train);
plot(test_X,stacking_y2);
plot(test_X,stacking_y3);
plot(test_X,stacking_y4);
plot(test_X,stacking_pre);
RSS_2order=GetRSS(test_Y,stacking_y2);
RSS_3order=GetRSS(test_Y,stacking_y3);
RSS_4order=GetRSS(test_Y,stacking_y4);
RSS_network=GetRSS(test_Y,ypre);
RSS_stakcking=GetRSS(test_Y,stacking_pre);

%% define a function that return RSS
function RSS = GetRSS(ytrue,ypre)
if (length(ytrue)~=length(ypre))
    error('你要求的RSS两个向量长度不匹配!');
end
RSS=sum((ytrue-ypre).^2);
end