################# PROYECTO FINAL DE ESTADÍSTICA BAYESIANA #################
#-------------------------- 2021-2 ---------------------------------------
#--- Librerías ----
library(rjags)
library(ggplot2)
library(tidyverse)
# Colores
colores <- c("#00afbb","#ff5044") #Verde: 0, Rojo: 1
options(digits = 1)

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


#----- Ajuste del modelo con glm y selección de variables ---------------

