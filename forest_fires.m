X=[forestfires.X,forestfires.Y,forestfires.FFMC,forestfires.ISI,forestfires.temp,forestfires.RH,forestfires.wind,forestfires.rain]
y=log(forestfires.area+1);
train_X=X(1:500)';
test_X=X(500:517)';
train_Y=y(1:500);
test_Y=y(500:517);

%% using network
net=newff(train_X',train_Y',[3,1]);
net.trainParam.epochs=1000;
trainednet=train(net,train_X',train_Y');
ypre=sim(trainednet,train_X');

%% using regression tree
rtree=fitctree(train_X,train_Y');
stacking_ctree = predict(rtree, train_X);

%% using SVM regression
rsvm=fitrsvm(train_X,train_Y')
stacking_rsvm = predict(rsvm, train_X);

%% using normal distribution regression
glm=fitglm(train_X,train_Y','Distribution','normal')
stacking_glm= predict(rsvm, train_X);



%% using meta regressor
X_train=[ypre',stacking_ctree,stacking_rsvm,stacking_glm]
net=newff(X_train',train_Y',[3,1]);
net.trainParam.epochs=1000;
net.trainParam.goal=1e-6;
stacking_net=train(net,X_train',train_Y') ;

%% show results
ypre=sim(trainednet,test_X');
stacking_ctree = predict(rtree, test_X);
stacking_rsvm = predict(rsvm, test_X);
stacking_glm= predict(glm, test_X);
X_test=[ypre',stacking_ctree,stacking_rsvm,stacking_glm];
yfinal=sim(stacking_net,X_test');
RSS_rtree=GetRSS(test_Y,stacking_ctree);
RSS_rsvm=GetRSS(test_Y,stacking_rsvm);
RSS_glm=GetRSS(test_Y,stacking_glm);
RSS_stacking=GetRSS(test_Y,yfinal');
RSS_network=GetRSS(test_Y,ypre');
%% define a function that return RSS
function RSS = GetRSS(ytrue,ypre)
if (length(ytrue)~=length(ypre))
    error('你要求的RSS两个向量长度不匹配!');
end
RSS=sum((ytrue-ypre).^2);
end

