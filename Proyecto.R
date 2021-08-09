################# PROYECTO FINAL DE ESTADÍSTICA BAYESIANA #################
#-------------------------- 2021-2 ---------------------------------------
#--- Librerías ----
#install.packages("ROCR")
library(rjags)
library(ggplot2)
library(tidyverse)
library(ROCR); library(boot);
# Colores
colores <- c("#00afbb","#ff5044") #Verde: 0, Rojo: 1
options(digits = 3)

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
m1 = glm(chd ~ ., data=SAheart, family=binomial(link = "logit"))
summary(m1)

# seleccionamos tobacco,ldl,famhist, typea y age
drop1(m1, test = "Chisq")

m2 = glm(chd ~ tobacco + ldl + famhist + typea + age, data = SAheart,
         family = binomial(link = "logit")) 
summary(m2)
drop1(m2, test = "Chisq")
anova(m1, m2, test = "Chisq") 
# Al realizar la prueba de Devianza, se verifica no es necesario incluir las otras variables

# H0: betan1 = ... = betanm = 0 vs, H1: que alguna se distinta de cero. 
# Como no rechazo H0, es mejor modelo chico

# Agregamos interacciones
m3 = glm(chd ~ tobacco + ldl + famhist + typea + age
         + tobacco:typea + ldl:famhist, data = SAheart, 
         family = binomial(link = "logit"))
summary(m3) # ldl y famhistPresent no significativas
drop1(m3, test = "Chisq")
anova(m2, m3, test = "Chisq")  # mejor modelo 3

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

### MODELO BAYESIANO ###
data <- list(
  y=SAheart$chd,
  x1=SAheart$tobacco,
  x2=SAheart$ldl,
  x3=SAheart$famhist,
  x4=SAheart$typea,
  x5=SAheart$age,
  n=length(SAheart$chd)
)

param <- c("alpha","Beta1","Beta2","Beta3", "Beta4", "Beta5" )
inits <-  function() {list(
  "alpha"=rnorm(1),
  "Beta1"=rnorm(1),
  "Beta2"=rnorm(1),
  "Beta3"=rnorm(1),
  "Beta4"=rnorm(1),
  "Beta5"=rnorm(1)
)
  
}

modelo=" model {
for(i in 1:n){
y[i]~dbern(p[i])
p[i] <- 1/(1.0001+exp(-(alpha+Beta1*x1[i]+Beta2*x2[i]+Beta3*x3[i]+Beta4*x4[i]+Beta5*x5[i])))
}

alpha ~ dnorm(0.0,1.0E-2)
Beta1 ~ dnorm(0.0,1.0E-2)
Beta2 ~ dnorm(0.0,1.0E-2)
Beta3 ~ dnorm(0.0,1.0E-2)
Beta4 ~ dnorm(0.0,1.0E-2)
Beta5 ~ dnorm(0.0,1.0E-2)
}

"
fit <- jags.model(textConnection(modelo),data,inits,n.chains=3)

update(fit,1000)

###  FUNCIONES UTILIZADAS A LO LARGO DEL SCRIPT:
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
