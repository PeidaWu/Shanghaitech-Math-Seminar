##运用BSS预测棒球手的薪水基于Hitter数据
library (ISLR)

##数据预处理
fix(Hitters)
names(Hitters)
dim (Hitters)
sum(is.na(Hitters$Salary))
Hitters = na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))

##BSS
library(leaps)
regfit.full = regsubsets(Salary~.,Hitters)
summary(regfit.full)
regfit.full = regsubsets(Salary~.,data=Hitters,nvmax=19)
reg.summary = summary(regfit.full)
names(reg.summary )
reg.summary$rsq

par(mfrow=c(2,2))
plot(reg.summary$rss,xlab ="Number of Variables",ylab =" RSS ",type ="l")
plot(reg.summary$adjr2,xlab ="Number of Variables",ylab ="Adjusted RSq",type ="l")

which.max(reg.summary$adjr2)
points(11,reg.summary$adjr2[11],col=" red ",cex =2,pch =20)

plot(reg.summary$cp,xlab ="Number of Variables", ylab ="Cp",type="l")
which.min(reg.summary$cp)
points(10,reg.summary$cp[10],col ="red",cex =2,pch =20)

which.min(reg.summary$bic)
plot(reg.summary$bic,xlab ="Number of Variables",ylab ="BIC",type="l")
points (6,reg.summary$bic[6],col ="red",cex =2,pch =20)

plot(regfit.full,scale ="r2")
plot(regfit.full,scale ="adjr2")
plot(regfit.full,scale ="Cp")
plot(regfit.full,scale ="bic")
coef(regfit.full,6)
##FSS
regfit.fwd = regsubsets(Salary~.,data= Hitters,nvmax =19, method ="forward")
summary(regfit.fwd)
#backSS
regfit.bwd=regsubsets(Salary~.,data= Hitters,nvmax =19,method ="backward")
summary(regfit.bwd)
##ridge
x = model.matrix(Salary~., Hitters)[,-1]
y = Hitters$Salary

library(glmnet)
grid = 10^seq(10,-2,length=100)
ridge.mod = glmnet(x,y,alpha=0, lambda=grid)

dim(coef(ridge.mod))

predict(ridge.mod, s = 4, type = "coefficients")[1:20,]

set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
y.test = y[test]

ridge.mod=glmnet(x[train,], y[train], alpha =0, lambda =grid ,thresh =1e-12)
ridge.pred=predict(ridge.mod,s=4,newx=x[test,])
mean((ridge.pred-y.test)^2)
mean((mean(y[train])-y.test)^2)

set.seed(1)
cv.out =cv.glmnet (x[train,], y[train], alpha=0)
plot(cv.out)
bestlam =cv.out$lambda.min
bestlam

ridge.pred= predict(ridge.mod ,s=bestlam , newx=x[test,])
mean((ridge.pred -y.test)^2)
out = glmnet(x,y,alpha =0)
predict(out , type ="coefficients",s= bestlam) [1:20,]
##lasso
lasso.mod =glmnet(x[train,], y[train], alpha=1, lambda=grid)
plot(lasso.mod)
set.seed (1)
cv.out =cv.glmnet (x[train ,], y[train], alpha=1)
plot(cv.out)
bestlam =cv.out$lambda.min
lasso.pred=predict(lasso.mod ,s=bestlam , newx=x[test,])
mean ((lasso.pred-y.test)^2)
out = glmnet(x,y, alpha =1, lambda = grid)
lasso.coef= predict (out ,type ="coefficients",s= bestlam ) [1:20,]
lasso.coef
bestlam

##pcr
library(pls)
set.seed(2)
pcr.fit=pcr(Salary~., data = Hitters, scale = TRUE, validation ="CV")
summary(pcr.fit)
##pls
set.seed (1)
pls.fit =plsr(Salary~.,dat =Hitters, scale=TRUE , validation ="CV")
summary (pls.fit )
