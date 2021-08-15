################# PROYECTO FINAL DE ESTADÍSTICA BAYESIANA #################
#-------------------------- 2021-2 ---------------------------------------
#--- Librerías ----
#install.packages("ROCR")
library(rjags)
library(MCMCpack) ### MCMC
library(ggplot2)
library(tidyverse)
library(car)
library(ROCR); library(boot);
# Colores
colores <- c("#00afbb","#ff5044") #Verde: 0, Rojo: 1
options(digits = 3)

###  FUNCIONES UTILIZADAS A LO LARGO DEL SCRIPT: -----------
# Función de cálculo de errores de clasificación globales y por grupo:
# Si test == TRUE, regresa los errores promediados sobre el conjunto de prueba.
# De lo contrario, regresa los errores sobre el conjunto de entrenamiento.
# Error global, error grupo 0, error grupo 1.
tasas_error = function(modelo,train, test=FALSE,corte) { 
  if (test) newdata = SAheart[-train,]
  else newdata = SAheart[train,]
  pred = as.numeric(predict(glm(modelo$formula, family = binomial(link = "logit"),
                                data = SAheart[train,]), newdata = newdata, 
                            type = "response") > corte)
  return(c(100*mean(newdata$chd != pred),
           100*mean(newdata$chd[newdata$chd==0] != pred[newdata$chd==0]), 
           100*mean(newdata$chd[newdata$chd==1] != pred[newdata$chd==1]))) }

#--- Datos ----
SAheart <- read.table("http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data",
                      sep=",",head=T,row.names=1)

str(SAheart) # 462 observaciones de 10 variables
which(is.na(SAheart)) # No hay datos faltantes

# Convertimos a factor la variable respuesta
SAheart$chd <- factor(SAheart$chd)

# Convertimos a factores las variables categóricas
SAheart$famhist <- factor(SAheart$famhist)

#--- Análisis Descriptivo ----
#---- Correlación entre variables
pairs(SAheart[,-c(5,10)],col=colores[SAheart$chd]) #Verde: 0, Rojo: 1
cor(SAheart[,-c(5,10)])
# La correlación más alta es entre adiposity y obesity


#---- Análisis del efecto de las variables sdp, tobacco, ldl, adiposity, famhist, 
# typea, obesity, alcohol y age en la variable respuesta chd

par(mfrow = c(2,3))
for (i in c(1:4,6,7)) {
  boxplot(SAheart[,i]~SAheart$chd, border=colores, col=0, main=names(SAheart)[i])}

par(mfrow = c(1,2))
for (i in c(8,9)) {
  boxplot(SAheart[,i]~SAheart$chd, border=colores, col=0, main=names(SAheart)[i])}

par(mfrow=c(1,1))
ggplot(SAheart,aes(x=famhist,fill=chd)) +
  geom_bar() +
  scale_fill_manual(values=colores) +
  theme_bw()
#---- Transformación de variables -----------
# transformación logaritmo para evitar concentración
#Se le suma 0.1 a las variables a transformar para evitar errores cuando se evalua en 0
SAheart$obesity=log(SAheart$obesity+0.1)
SAheart$tobacco=log(SAheart$tobacco+0.1)
SAheart$alcohol=log(SAheart$alcohol+0.1)

par(mfrow = c(1,3))
for (i in c(7,2,8)) {
  boxplot(SAheart[,i]~SAheart$chd, border=colores, col=0, main=names(SAheart)[i])}


pairs(SAheart[,-c(5,10)],col=colores[SAheart$chd]) #Verde: 0, Rojo: 1



#----- Ajuste del modelo con glm y selección de variables ---------------
# modelo con todas las variables
set.seed(1)
m1 = glm(chd ~ ., data=SAheart, family=binomial(link = "logit"))
summary(m1)

# seleccionamos tobacco,ldl,famhist, typea y age
drop1(m1, test = "Chisq")

set.seed(1)
m2 = glm(chd ~ tobacco + ldl + famhist + typea + age, data = SAheart,
         family = binomial(link = "logit")) 
summary(m2)
drop1(m2, test = "Chisq")
anova(m1, m2, test = "Chisq") 
# Al realizar la prueba de Devianza, se verifica no es necesario incluir las otras variables

# H0: betan1 = ... = betanm = 0 vs, H1: que alguna se distinta de cero. 
# Como no rechazo H0, es mejor modelo chico

# Agregamos interacciones
set.seed(1)
m3 = glm(chd ~ tobacco + ldl + famhist + typea + age
         + tobacco:typea + ldl:famhist, data = SAheart, 
         family = binomial(link = "logit"))
summary(m3) # ldl y famhistPresent no significativas
drop1(m3, test = "Chisq")
anova(m2, m3, test = "Chisq")  # mejor modelo 3

set.seed(1)
m4 = glm(chd ~ tobacco + typea + age
         + tobacco:typea + ldl:famhist, data = SAheart, 
         family = binomial(link = "logit"))
summary(m4)
drop1(m4, test = "Chisq")
anova(m3, m4, test = "Chisq")  # Se eliminan variables lhd y famhist

vif(m4)

# Seleccionamos el modelo

#Evaluación del desempeño del modelo 
# Se fija semilla y escoge el conjunto de entrenamiento.

set.seed(1); train=sample(1:462,393,replace = FALSE) #85% de las observaciones

# Selección del modelo mediante los criterios: AIC, BIC, tasas de error de 
# clasificación (globales y por grupo, aparentes, sobre cjto. de prueba y de 
# entrenamiento), AUC y ANOVA.

modelos = list(m1, m2, m3, m4)
(errores_aparentes = sapply(modelos, tasas_error, train = train,test=FALSE, corte = 0.5))  #TASAS APARENTES
(errores_test = sapply(modelos, tasas_error, train = train,test=TRUE, corte = 0.5))                                                      

c = cbind(sapply(modelos, function(x){x$deviance}), sapply(modelos, AIC), 
          sapply(modelos, BIC), t(errores_aparentes), t(errores_test))

######################### CONCLUSIONES SOBRE AJUSTE DEL MODELO #############################

colnames(c) = c("Devianza", "AIC", "BIC", "ApGlob", "Ap0", "Ap1", "TestGlob", 
                "Test0", "Test1")
row.names(c) = paste("Modelo", c(1,2,3,4)); c

# El AIC, BIC, DEVIANZA son medidas de bondad de ajuste 
# Utilizadas principalmente para clasificar, no predecir. 

# Un modelo saturado es aquél que tiene tasas aparentes (sobre todo el conjunto de datos)
# buenas, pero tasas validadas (sobre conjunto de entrenamiento) malas. 

# modelo 4 tiene todas las variables significativas
# Elección: modelo 4
round(cbind(Coeficiente = coef(m4), confint(m4)), 2)
round(cbind(OR = exp(coef(m4)), exp(confint(m4))), 2)[-1,] # Odds ratios

### MODELO BAYESIANO --------
attach(SAheart)
#modMCMC <- MCMClogit(chd~tobacco+ldl, burnin=1000, mcmc=1000, thin=1) 

##------ Modelo 1 con jags --------------------
data <- list(
  y=as.numeric(chd)-1,
  x1=tobacco,
  x2=ldl,
  x3=typea,
  x4=age,
  n=length(chd)
)

param <- c("alpha","Beta1","Beta2","Beta3", "Beta4")
inits <-  function() {list(
  "alpha"=rnorm(1),
  "Beta1"=rnorm(1),
  "Beta2"=rnorm(1),
  "Beta3"=rnorm(1),
  "Beta4"=rnorm(1)
)
  
}

modelo=" model {
  for(i in 1:n){
    y[i]~dbern(p[i])
    p[i] <- 1/(1.000001+exp(-(alpha+Beta1*x1[i]+Beta2*x2[i]+Beta3*x3[i]+Beta4*x4[i])))
  }

  alpha ~ dnorm(0.0,1.0E-2)
  Beta1 ~ dnorm(0.0,1.0E-2)
  Beta2 ~ dnorm(0.0,1.0E-2)
  Beta3 ~ dnorm(0.0,1.0E-2)
  Beta4 ~ dnorm(0.0,1.0E-2)
}

"

set.seed(1)
fit <- jags.model(textConnection(modelo),data,inits,n.chains=3)

update(fit,1000)

sample <- coda.samples(fit,param,n.iter = 4000,thin = 1)

dev.new()
plot(sample)

gelman.plot(sample)

summary(sample)

# Iterations = 2001:6000
# Thinning interval = 1 
# Number of chains = 3 
# Sample size per chain = 4000 
# 
# 1. Empirical mean and standard deviation for each variable,
# plus standard error of the mean:
#   
#   Mean     SD Naive SE Time-series SE
# Beta1  0.2440 0.0743 6.79e-04       0.001511
# Beta2  0.1671 0.0537 4.90e-04       0.001980
# Beta3  0.0373 0.0126 1.15e-04       0.001126
# Beta4  0.0545 0.0102 9.27e-05       0.000646
# alpha -6.0590 0.9736 8.89e-03       0.119751
# 
# 2. Quantiles for each variable:
#   
#   2.5%     25%     50%     75%   97.5%
# Beta1  0.1018  0.1940  0.2419  0.2939  0.3907
# Beta2  0.0633  0.1306  0.1668  0.2031  0.2734
# Beta3  0.0123  0.0289  0.0377  0.0459  0.0621
# Beta4  0.0350  0.0478  0.0545  0.0611  0.0746
# alpha -8.0104 -6.7054 -6.0259 -5.3672 -4.2202

#------------ Tasas de error de clasificación del modelo 1 con jags ----------
head(sample)
x=cbind(rep(1.0,length(chd)),tobacco,ldl,typea, age)
aux_cadenas =do.call(rbind,sample)
coeficientes =colMeans(aux_cadenas)

param_acomodados =c(coeficientes[5],coeficientes[1:4])

param_acomodados

y_hat <- drop(x%*%param_acomodados)

probas = 1/(1+exp(-y_hat))
head(probas)

matriz_jags <- ifelse(probas>=0.5,1,0)
head(matriz_jags)

table(SAheart$chd,matriz_jags)
(254+75)/462 #0.712
1-(254+75)/462 #Error de clasificación 0.288

errores_jags <- data.frame(modelo=c("Modelo 1","Modelo 2","Modelo 3"),
                           error_global=rep(NA,3),
                           error_0=rep(NA,3),
                           error_1=rep(NA,3))


errores_jags[1,2:4] <- c(100*mean(SAheart$chd != matriz_jags),
  100*mean(SAheart$chd[SAheart$chd==0] != matriz_jags[SAheart$chd==0]), 
  100*mean(SAheart$chd[SAheart$chd==1] != matriz_jags[SAheart$chd==1]))
errores_jags


plot(ecdf(probas))

#-------- Modelo 2 con jags -----------------
data.2<- list(
  y=as.numeric(SAheart$chd)-1,
  x1=SAheart$tobacco,
  x2=SAheart$ldl,
  x3=as.numeric(SAheart$famhist=="Present"),
  x4=SAheart$typea,
  x5=SAheart$age,
  n=length(SAheart$chd)
)

param.2 <- c("alpha","Beta1","Beta2","Beta3", "Beta4", "Beta5" )
inits.2 <-  function() {list(
  "alpha"=rnorm(1),
  "Beta1"=rnorm(1),
  "Beta2"=rnorm(1),
  "Beta3"=rnorm(1),
  "Beta4"=rnorm(1),
  "Beta5"=rnorm(1)
)
  
}

modelo.2=" model {
for(i in 1:n){
  y[i]~dbern(p[i])
  p[i] <- 1/(1.000001+exp(-(alpha+Beta1*x1[i]+Beta2*x2[i]+Beta3*x3[i]+Beta4*x4[i]+Beta5*x5[i])))
  }
  alpha ~ dnorm(0.0,1.0E-2)
  Beta1 ~ dnorm(0.0,1.0E-2)
  Beta2 ~ dnorm(0.0,1.0E-2)
  Beta3 ~ dnorm(0.0,1.0E-2)
  Beta4 ~ dnorm(0.0,1.0E-2)
  Beta5 ~ dnorm(0.0,1.0E-2)
  }
"

set.seed(1)
fit.2 <- jags.model(textConnection(modelo.2),data.2,inits.2,n.chains=3)

update(fit.2,1000)
#update(fit.2,4000) Para ver que el modelo 2 converge quitando las 4000 interacciones



sample.2 <- coda.samples(fit.2,param.2,n.iter = 4000,thin = 1)

dev.new()
plot(sample.2)

gelman.plot(sample.2)

summary(sample.2)

# Iterations = 2001:6000
# Thinning interval = 1 
# Number of chains = 3 
# Sample size per chain = 4000 
#
# 1. Empirical mean and standard deviation for each variable,
# plus standard error of the mean:
#  
#         Mean     SD Naive SE Time-series SE
# Beta1  0.248 0.0768 7.01e-04       0.001468
# Beta2  0.151 0.0568 5.19e-04       0.002286
# Beta3  0.888 0.2239 2.04e-03       0.004159
# Beta4  0.037 0.0133 1.21e-04       0.001299
# Beta5  0.051 0.0107 9.73e-05       0.000703
# alpha -6.224 1.0378 9.47e-03       0.131830
#
# 2. Quantiles for each variable:
#   
#         2.5%     25%     50%     75%   97.5%
# Beta1  0.1022  0.1942  0.2464  0.3000  0.4012
# Beta2  0.0435  0.1118  0.1505  0.1894  0.2645
# Beta3  0.4496  0.7383  0.8880  1.0376  1.3282
# Beta4  0.0127  0.0276  0.0364  0.0454  0.0649
# Beta5  0.0311  0.0436  0.0508  0.0581  0.0723
# alpha -8.4299 -6.8953 -6.1528 -5.4735 -4.3884

# ---------------- Tasas de error de clasificación del modelo 2 con jags ----------
head(sample.2)
x=cbind(rep(1.0,length(chd)),tobacco,ldl,as.numeric(SAheart$famhist=="Present"),typea, age)
aux_cadenas.2 =do.call(rbind,sample.2)
coeficientes.2 =colMeans(aux_cadenas.2)

param_acomodados.2 =c(coeficientes.2[6],coeficientes.2[1:5])

param_acomodados.2

y_hat.2 <- drop(x%*%param_acomodados.2)

probas.2 = 1/(1+exp(-y_hat.2))
head(probas.2)

matriz_jags.2 <- ifelse(probas.2>=0.5,1,0)
head(matriz_jags.2)

table(SAheart$chd,matriz_jags.2)

errores_jags[2,2:4] <- c(100*mean(SAheart$chd != matriz_jags.2),
                         100*mean(SAheart$chd[SAheart$chd==0] != matriz_jags.2[SAheart$chd==0]), 
                         100*mean(SAheart$chd[SAheart$chd==1] != matriz_jags.2[SAheart$chd==1]))
errores_jags


plot(ecdf(probas.2))

#-------- Modelo 3 con jags -----------------
data.3<- list(
  y=as.numeric(chd)-1,
  x1=tobacco,
  x2=ldl,
  x3=as.numeric(famhist=="Present"),
  x4=typea,
  x5=age,
  x6=tobacco*typea,
  x7=ldl*as.numeric(famhist=="Present"),
  n=length(chd)
)

param.3 <- c("alpha","Beta1","Beta2","Beta3", "Beta4", "Beta5", "Beta6", "Beta7" )
inits <-  function() {list(
  "alpha"=rnorm(1),
  "Beta1"=rnorm(1),
  "Beta2"=rnorm(1),
  "Beta3"=rnorm(1),
  "Beta4"=rnorm(1),
  "Beta5"=rnorm(1),
  "Beta6"=rnorm(1),
  "Beta7"=rnorm(1)
)
  
}

modelo.3=" model {
for(i in 1:n){
y[i]~dbern(p[i])
p[i] <- 1/(1.000001+exp(-(alpha+Beta1*x1[i]+Beta2*x2[i]+Beta3*x3[i]+Beta4*x4[i]+Beta5*x5[i]+Beta6*x6[i]+Beta7*x7[i])))}
alpha ~ dnorm(0.0,1.0E-2)
Beta1 ~ dnorm(0.0,1.0E-2)
Beta2 ~ dnorm(0.0,1.0E-2)
Beta3 ~ dnorm(0.0,1.0E-2)
Beta4 ~ dnorm(0.0,1.0E-2)
Beta5 ~ dnorm(0.0,1.0E-2)
Beta6 ~ dnorm(0.0,1.0E-2)
Beta7 ~ dnorm(0.0,1.0E-2)
}
"

set.seed(1)
fit.3 <- jags.model(textConnection(modelo.3),data.3,inits,n.chains=3)

update(fit.3,1000)

sample.3 <- coda.samples(fit.3,param.3,n.iter = 4000,thin = 1)

dev.new()
plot(sample.3)

gelman.plot(sample.3)

summary(sample.3)

# Iterations = 2001:6000
# Thinning interval = 1 
# Number of chains = 3 
# Sample size per chain = 4000 
# 
# 1. Empirical mean and standard deviation for each variable,
# plus standard error of the mean:
#   
#   Mean      SD Naive SE Time-series SE
# Beta1  1.40357 0.48502 4.43e-03       0.057890
# Beta2  0.00457 0.07713 7.04e-04       0.004723
# Beta3 -0.92669 0.66594 6.08e-03       0.048144
# Beta4  0.05759 0.01408 1.29e-04       0.001484
# Beta5  0.05485 0.01056 9.64e-05       0.000689
# Beta6 -0.02098 0.00864 7.89e-05       0.001015
# Beta7  0.36805 0.12429 1.13e-03       0.009164
# alpha -6.88180 1.09452 9.99e-03       0.167077
# 
# 2. Quantiles for each variable:
#   
#   2.5%     25%      50%     75%    97.5%
# Beta1  0.4386  1.0804  1.41434  1.7089  2.37491
# Beta2 -0.1460 -0.0487  0.00551  0.0572  0.15732
# Beta3 -2.2635 -1.3679 -0.91274 -0.4628  0.34551
# Beta4  0.0326  0.0479  0.05681  0.0660  0.08815
# Beta5  0.0346  0.0475  0.05473  0.0621  0.07526
# Beta6 -0.0382 -0.0264 -0.02112 -0.0152 -0.00375
# Beta7  0.1325  0.2788  0.36778  0.4495  0.61536
# alpha -9.4586 -7.5222 -6.83225 -6.1093 -5.01362


# ---------------- Tasas de error de clasificación del modelo 3 con jags ----------
head(sample.3)
x=cbind(rep(1.0,length(chd)),tobacco,ldl,as.numeric(SAheart$famhist=="Present"),
        typea, age, tobacco*typea,ldl*as.numeric(famhist=="Present"))
aux_cadenas.3 =do.call(rbind,sample.3)
coeficientes.3 =colMeans(aux_cadenas.3)

param_acomodados.3 =c(coeficientes.3[8],coeficientes.3[1:7])

param_acomodados.3

y_hat.3 <- drop(x%*%param_acomodados.3)

probas.3 = 1/(1+exp(-y_hat.3))
head(probas.3)

matriz_jags.3 <- ifelse(probas.3>=0.5,1,0)
head(matriz_jags.3)

table(SAheart$chd,matriz_jags.3)

errores_jags[3,2:4] <- c(100*mean(SAheart$chd != matriz_jags.3),
                         100*mean(SAheart$chd[SAheart$chd==0] != matriz_jags.3[SAheart$chd==0]), 
                         100*mean(SAheart$chd[SAheart$chd==1] != matriz_jags.3[SAheart$chd==1]))
errores_jags

plot(ecdf(probas.3))
