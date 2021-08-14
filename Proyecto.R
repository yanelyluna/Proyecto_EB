################# PROYECTO FINAL DE ESTADÍSTICA BAYESIANA #################
#-------------------------- 2021-2 ---------------------------------------
#--- Librerías ----
#install.packages("ROCR")
library(rjags)
library(MCMCpack) ### MCMC
library(ggplot2)
library(tidyverse)
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

# Los modelos 3 y 4 tienen un desempeño parecido
# Modelo 3 predice mejor en train

# Elección: modelo 3
round(cbind(Coeficiente = coef(m3), confint(m3)), 2)
round(cbind(OR = exp(coef(m3)), exp(confint(m3))), 2)[-1,] # Odds ratios

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
# Beta1  0.2403 0.0741 6.76e-04       0.001307
# Beta2  0.1721 0.0560 5.11e-04       0.002097
# Beta3  0.0390 0.0113 1.04e-04       0.000959
# Beta4  0.0554 0.0103 9.39e-05       0.000738
# alpha -6.2167 0.8898 8.12e-03       0.105540
# 
# 2. Quantiles for each variable:
#   
#   2.5%     25%     50%     75%   97.5%
# Beta1  0.0960  0.1909  0.2406  0.2894  0.3853
# Beta2  0.0637  0.1340  0.1720  0.2099  0.2833
# Beta3  0.0169  0.0315  0.0390  0.0465  0.0614
# Beta4  0.0356  0.0483  0.0555  0.0624  0.0756
# alpha -8.0559 -6.7696 -6.1653 -5.6207 -4.5531

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
(252+77)/462 #0.712
1-(252+77)/462 #Error de clasificación 0.288

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
#  Mean     SD Naive SE Time-series SE
# Beta1  0.2446 0.0764 6.97e-04        0.00140
# Beta2  0.1524 0.0556 5.08e-04        0.00211
# Beta3  0.8867 0.2302 2.10e-03        0.00448
# Beta4  0.0373 0.0136 1.24e-04        0.00134
# Beta5  0.0514 0.0108 9.87e-05        0.00076
# alpha -6.2644 1.0423 9.51e-03        0.14068
#
# 2. Quantiles for each variable:
#  
#  2.5%     25%     50%     75%   97.5%
# Beta1  0.0968  0.1935  0.2437  0.2949  0.3971
# Beta2  0.0451  0.1153  0.1523  0.1888  0.2651
# Beta3  0.4431  0.7296  0.8852  1.0418  1.3387
# Beta4  0.0129  0.0278  0.0367  0.0465  0.0645
# Beta5  0.0313  0.0440  0.0510  0.0587  0.0736
# alpha -8.4425 -6.9306 -6.2071 -5.5279 -4.3735

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
# Beta1  1.32044 0.44978 4.11e-03       0.052896
# Beta2 -0.00843 0.07481 6.83e-04       0.004315
# Beta3 -1.04085 0.61914 5.65e-03       0.040606
# Beta4  0.05233 0.01495 1.37e-04       0.001634
# Beta5  0.05366 0.01122 1.02e-04       0.000784
# Beta6 -0.01946 0.00804 7.34e-05       0.000888
# Beta7  0.38901 0.11704 1.07e-03       0.008158
# alpha -6.46515 1.16991 1.07e-02       0.175057
# 
# 2. Quantiles for each variable:
#   
#   2.5%     25%      50%     75%    97.5%
# Beta1  0.4902  1.0166  1.30433  1.5984  2.22982
# Beta2 -0.1574 -0.0557 -0.00771  0.0402  0.13787
# Beta3 -2.2218 -1.4441 -1.04705 -0.6491  0.21515
# Beta4  0.0232  0.0419  0.05242  0.0630  0.08084
# Beta5  0.0319  0.0459  0.05351  0.0612  0.07678
# Beta6 -0.0357 -0.0245 -0.01917 -0.0141 -0.00455
# Beta7  0.1575  0.3138  0.38945  0.4653  0.61515
# alpha -8.7248 -7.2998 -6.41006 -5.6478 -4.29011


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

# matriz_jags.3
# 0   1
# 0 262  40
# 1  75  85

#     modelo error_global error_0 error_1
# 1 Modelo 1           NA      NA      NA
# 2 Modelo 2           NA      NA      NA
# 3 Modelo 3         24.9    13.2    46.9

plot(ecdf(probas.3))
