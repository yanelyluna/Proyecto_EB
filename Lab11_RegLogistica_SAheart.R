

###########################################################################
###################### LAB 11: REGRESIÓN LOGÍSTICA ########################
###########################################################################

# Objetivos: 

# 1. Identificación de los factores de riesgo asociados a la Enfermedad Coronaria de 
# Corazón mediante Regresión Logística.

# 2. Validación mediante Bootstrap y Cross-Validation de las tasas de error de 
# clasificación globales del modelo de Regresión Logística ajustado con la base de 
# datos SAheart. 

# La base de datos contiene observaciones de 462 hombres de Western Cape, 
# Sudáfrica, 10 atributos.

###  RESULTADOS:

# Mejor modelo ajustado en cuanto a bondad de ajuste y poder predictivo:

# Coeficientes y odds ratios estimados con intervalos al 95% de confianza:
# -----------------------------------------------------------------------------
# |      Variable     |Coeficiente  2.5 %  97.5 % |Odds Ratio  2.5 %   97.5 % |
# -----------------------------------------------------------------------------
# | (Intercept)       |   -3.12     -4.57  -1.73  |                           |
# | tobacco           |    0.24      0.10   0.40  |   1.28     1.10     1.49  |
# | ldl               |    0.15      0.05   0.26  |   1.17     1.05     1.30  |
# | famhistPresent    |    0.88      0.44   0.33  |   2.41     1.56     3.77  |
# | typea             |    0.03      0.01   0.66  |   1.03     1.01     1.06  |
# | age1              |   -0.89     -1.38  -0.42  |   0.41     0.25     0.66  |
# | age0              |   -1.83     -2.75  -1.01  |   0.16     0.06     0.36  |
# -----------------------------------------------------------------------------


#Nota: En regresión logística, la interpretación de los coeficientes puede resultar
# complicada, por lo que se suelen utilizar los Odds Ratios. 

# Tasas de error de clasificación por métodos de remuestreo

# ----------------------------------------------------------------
# |        Método             | Global   |  Grupo 0  |  Grupo 1  | 
# ----------------------------------------------------------------
# | Aparentes                 |  25.97   |   15.56   |   45.62   |
# | Validation Approach (85%) |  27.68   |   16.59   |   47.90   |  
# | Bootstrap B=500           |  25.80   |   15.87   |   44.78   |  
# | Bootstrap OOB B=500       |  28.16   |   17.85   |   47.43   |  
# | .632 Bootstrap            |  26.67   |   16.60   |   45.75   |  
# | 10-Fold CV                |  26.62   |   16.82   |   44.48   | 
# | Leave One Out CV          |  27.06   |     -     |     -     | 
# ----------------------------------------------------------------


par(pch = 16, cex.axis = 1.5, cex.lab = 1.2, lwd = 2); col = c(2,4)
library(ROCR); library(boot); options(digits = 4)

### Nota: Las funciones tasas_error, esp_sens y AUC son utilizadas a lo 
#         largo del script y están definidas al final del archivo.
SAheart <- read.table("http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data",
                      sep=",",head=T,row.names=1)

############################## INTRODUCCIÓN ###############################

#data(SAheart); 
which(is.na(SAheart)); SAheart$chd = factor(SAheart$chd) #Super importante definir
#variable respuesta como factor en la regresión logística
?SAheart; str(SAheart)

#Se utiliza la edad como variable contínua

round(cor(SAheart[,-c(5,9,10)]),2) 
# Se observa correlación alta: adiposidad con obesidad, por lo que será importante
# incluir interacción entre estas variables. 

table(SAheart$chd); table(SAheart$famhist, SAheart$chd) 

pairs(~ ., data = SAheart[,c(1:4,6:9)], col = col[SAheart$chd]) 
# Diagramas de dispersión sólo para variables continuas.
# Observaciones:
# 1. No se observa una separación entre las clases. 
# 2. Hay datos que se concentran en zonas, por lo que habrá problemas de bondad de ajuste.
# Para corregir el problema es necesario aplicar transformaciones, generalmente logaritmo 
# siempre es buena opción.

par(mfrow = c(3,3)); for (i in c(1:4,6:9)) {
  boxplot(SAheart[,i]~SAheart$chd, border=col, col=0, main=names(SAheart)[i])}

# Los diagramas de caja pueden mostrar datos atípicos. Las cajas son muy pequeñas, 
# lo que significa que los datos están muy cercanos, más evidencias de que se tienen
# que transformar las variables.

# Aplicamos transformaciones

SAheart$obesity=log(SAheart$obesity+0.1); SAheart$tobacco=log(SAheart$tobacco+0.1)
SAheart$alcohol=log(SAheart$alcohol+0.1); 
pairs(~ ., data = SAheart[,c(1:4,6:9)], col = col[SAheart$chd]) 

#La distribución de la variable mejora, ya no están concentrados.
# Sin embargo, no se distingue una separación entre las clases. 


# Categorizamos la variable age (por fines didácticos):
age2 = SAheart$age; age2[SAheart$age <= 30] = 0
age2[SAheart$age > 30 & SAheart$age <= 38] = 1
age2[SAheart$age > 38 & SAheart$age <= 48] = 2
age2[SAheart$age > 48 & SAheart$age <= 52] = 3
age2[SAheart$age > 52 & SAheart$age <= 60] = 4
age2[SAheart$age > 60] = 5
SAheart$age = as.factor(age2); table(SAheart$age); table(SAheart$age,SAheart$chd)

# La tabla nos muestra que hay sólo 8 y 15 personas con edades del grupo 0 y 1 que tienen
# la enfermedad, por lo que habrá problemas cuando se seleccione el conjunto de muestra y
# entrenamiento. (Puede que el conjunto no incluya estas observaciones y no estáran 
# representadas)
# Generalmente se usa como categoría de referencia a la variable que tiene mayor
# cardinalidad. 

par(mfrow = c(3,3)); for (i in c(1:4,6:8)) {
  boxplot(SAheart[,i]~SAheart$age, border=col, col=0, main=names(SAheart)[i])}
par(mfrow = c(1,1))


########################## SELECCIÓN DE MODELOS ###########################

m1 = glm(chd ~ .+adiposity:obesity, data=SAheart, family=binomial(link = "logit"))
summary(m1)

# Al igual que en regresión lineal mÃºltiple, se pueden colapsar categorías si son 
# significativas al mismo nivel y si sus coeficientes son parecidos. 

drop1(m1, test = "Chisq")

# Se deberían colapsar categorías de age, hay que recordar que es una 
# variable categórica ordinal:

SAheart$age = relevel(SAheart$age, "2"); m1 = update(m1); summary(m1) 

#Se juntan grupos de edades 1 y 2 por similitud en sus coeficientes.

SAheart$age[SAheart$age == 2] = 1; SAheart$age = droplevels(SAheart$age)
m1 = update(m1); summary(m1) 

# Después de cambiar el nivel de referencia y colapsar dos grupos se obtiene que 
# todas las categorías son estadísticamente diferentes a la categoría de referencias. 


SAheart$age = relevel(SAheart$age, "5"); m1 = update(m1); summary(m1) 

# Cambiamos categoría de referencia y se colapsan grupo 4 y5
SAheart$age[SAheart$age == 5] = 4; SAheart$age = droplevels(SAheart$age)
m1 = update(m1); summary(m1)
SAheart$age = relevel(SAheart$age, "3"); m1 = update(m1); summary(m1)
# No hay diferencia entre categorias 3 y 1, 4 y 3

SAheart$age[SAheart$age == 4] = 3; SAheart$age = droplevels(SAheart$age) 
# Se colapsan las categorías 3 y 4
m1 = update(m1); summary(m1)

table(SAheart$age) # Cambiamos la categoría de referencia
SAheart$age = factor(SAheart$age, levels = c(3,1,0)); m1 = update(m1)

### Ahora, procedemos a hacer selección de variables:
# Usar drop1 para seleccionarlas siempre
drop1(m1, test = "Chisq")

m2 = glm(chd ~ . + adiposity:obesity - 1, data = SAheart,
         family = binomial(link = "logit")) #Eliminamos intercepto porque no es significativo
summary(m2)
drop1(m2, test = "Chisq")
anova(m1, m2, test = "Chisq") 
# Al realizar la prueba de Devianza, se verifica que es necesario incluir el intercepto

# La prueba de devianza indica que: M2 c M1. 
# Las variables no incluidas en M2 pero sí en M1 son xn1, ..., xnm
# H0: betan1 = ... = betanm = 0 vs, H1: que alguna se distinta de cero. 
# Si no rechazo, es mejor modelo chico
# Si rechazo, es mejor modelo grande

# En este caso, se rechaza la prueba de hipótesis, por lo que el intercepto es 
# estadísticamente distinto de cero, por lo que será mejor incluirlo en el modelo.

#Nota: Entre menor devianza, mejor verosimilitud, mejor modelo. 

m3 = glm(chd ~ . + adiposity:obesity - alcohol, data = SAheart, 
         family = binomial(link = "logit"))
summary(m3)
drop1(m3, test = "Chisq")
anova(m1, m3, test = "Chisq")  # Se elimina variable alcohol

m4 = glm(chd ~ . - adiposity - alcohol - sbp, data = SAheart, 
         family = binomial(link = "logit"))
summary(m4)
drop1(m4, test = "Chisq")
anova(m3, m4, test = "Chisq")  # Se elimina variable sbp y el modelo pequeño es siendo mejor

m5 = glm(chd ~ . - adiposity - alcohol - sbp -obesity, data = SAheart, 
         family = binomial(link = "logit"))
summary(m5)
drop1(m5, test = "Chisq") # Todas las varibles incluidas en el modelo son significativas

anova(m3, m4, test = "Chisq") # Es mejor el modelo con menos variables. 
anova(m1, m5, test = "Chisq") # Pasa lo mismo si se compara con el modelo inicial. 


######################## EVALUACIÓN DE LOS MODELOS #########################
#Evaluación del desempeño del modelo (cambia segÃºn el conjunto de entrenamiento). 

# Se fija semilla y escoge el conjunto de entrenamiento.

set.seed(1); train = replicate(200, sample(1:425, 393)) #{train}=393~462*0.85

#train es una matriz. Columna es cada conjunto de entrenamiento (200 en total)

# Selección del modelo mediante los criterios: AIC, BIC, tasas de error de 
# clasificación (globales y por grupo, aparentes, sobre cjto. de prueba y de 
# entrenamiento), AUC y ANOVA.

modelos = list(m1, m3, m4, m5)
(errores_aparentes = sapply(modelos, tasas_error, train = NULL, corte = 0.5))  #TASAS APARENTES
(auc = sapply(modelos, function(modelo){mean(apply(train, 2, AUC, m = modelo))})) #
(ROC = lapply(modelos, function(modelo){matrix(apply(apply(train, 2, 
                                                           esp_sens, m = modelo), 1, mean), ncol=3)})) #ESPECIFICIDAD, SENSIBILIDAD Y PUNTO DE CORTE VALIDADAS
(errores_test = sapply(1:4, function(i) {apply(apply(train, 2, tasas_error,
                                                     m = modelos[[i]], corte = 0.5), 1, mean)})) #TASAS ERROR CONJUNTO DE PRUEBA


######################### CONCLUSIONES SOBRE AJUSTE DEL MODELO #############################

c = cbind(sapply(modelos, function(x){x$deviance}), sapply(modelos, AIC), 
          sapply(modelos, BIC), t(errores_aparentes), t(errores_test), auc)

colnames(c) = c("Devianza", "AIC", "BIC", "ApGlob", "Ap0", "Ap1", "TestGlob", 
                "Test0", "Test1", "AUC")
row.names(c) = paste("Modelo", c(1,3,4,5)); c

# El AIC, BIC, DEVIANZA son medidas de bondad de ajuste 
# Utilizadas principalmente para clasificar, no predecir. 

# Un modelo saturado es aquél que tiene tasas aparentes (sobre todo el conjunto de datos)
# buenas, pero tasas validadas (sobre conjunto de entrenamiento) malas. 

# Si el objetivo es discriminar: MODELO 4 Y 5
# Si el objetivo es predecir: MODELO 1 Y 2 

# Elección: modelo 5
round(cbind(Coeficiente = coef(m5), confint(m5)), 2)
round(cbind(OR = exp(coef(m5)), exp(confint(m5))), 2)[-1,] # Odds ratios


###########################  VALIDACIÓN DE TASAS DE ERROR   ###############################

# Modelo ajustado:
modelo = glm(chd ~ . - adiposity - alcohol - sbp - obesity, data = SAheart, 
             family = binomial(link = "logit"))


######################### VALIDATION APPROACH #############################

# 500 muestras de tamaño 462 con #{train} = 393 ~ 462Â· 0.85
set.seed(1); train = replicate(500, sample(1:462, 393)) 


# Tasas de error de clasificación globales en conjunto de prueba
(error_VA = apply(train, 2, error, test = T))   


# Nótese que al hacer más iteraciones (i), se estabilizan las estimaciones
# de las tasas de error de clasificación sobre el conjunto de prueba.
par(mfrow=c(1,3))
plot(cumsum(error_VA[1,])/1:500, type="l", ylab="TECG VA Global (%)", xlab="i")
plot(cumsum(error_VA[2,])/1:500, type="l", ylab="TECG VA Grupo 0 (%)", xlab="i")
plot(cumsum(error_VA[3,])/1:500, type="l", ylab="TECG VA Grupo 1 (%)", xlab="i")



############################## BOOTSTRAP ##################################
# Se utilizan B=500 muestras

set.seed(1)

# Tasas de error sobre conjunto de entrenamiento, al que le estamos ajustando el modelo.
(error_Boot = replicate(500, error(sample(1:462, 462, repl = TRUE), test=F)))
 


########################### BOOTSTRAP OUT OF BAG ##########################
# Muestra con reemplazo. Se prueba con las que quedaron fuera de la muestra. 
# Se utilizan B=500 muestran

set.seed(1)
(error_Boot_OOB = replicate(500, error(sample(1:462, 462, repl = TRUE), test=T)))


############################ .632 BOOTSTRAP ###############################
# Ponderación entre los dos métodos. 
# Â¿Cómo se obtiene la ponderación?

(error_Boot632 = 0.632 * error_Boot + 0.368 * error_Boot_OOB)


########################## K FOLD CROSS-VALIDATION ########################

set.seed(1); scrambled = sample(1:462, 462)  
# Scrambled ayuda a permutar los índices de las variables.

# La siguiente función regresa entradas de cada partición.
# Lista con conjunto de entrenamiento asociado a cada una de las particiones (fold)
train_cv = function(k) {
  set.seed(1)
  folds = cut(1:462, breaks = k, labels = F)  #Separar las observaciones en k grupos. 
  return(lapply(1:k, function(x){ scrambled[which(folds != x)] })) }

k = c(2, 5, 10, 22*(1:20))
sapply(train_cv(k[2]), error_global, test = T)
(error_CV = sapply(k, function(x){sapply(train_cv(x), error_global, test=T)}))

bias_CV = sapply(error_CV, function(x){mean(x-mean(x))}) # Sesgo
sd_CV = sapply(error_CV, sd) # Desviación estándar

par(mfrow = c(1,3))
plot(k, sapply(error_CV, mean), ylab="TECG CV (%)", ylim = c(26,29), 
     type="b")
plot(k, sd_CV, xlab="k", ylab = "Desviación estándar", type="b")
plot(k, abs(bias_CV), ylab = "|Sesgo|", type="b")

#La primera gráfica no es recomendable. 
#Fijarse en el sesgo de la desviación estándar. 
#Considerar sesgo, desviación estándar y cardinalidad de la base de datos. 


which.min(abs(bias_CV))
which.min(abs(bias_CV)[-23])
which.min(abs(bias_CV)[-c(20,23)])
k[c(1:3,10,20,23)] # k cuyas estimaciones tienen menor sesgo y desviaciÃƒÂ³n.

c=round(rbind(sapply(error_CV, mean), sd_CV, bias_CV*10^15),2)[,c(1:3,10,20,23)]
colnames(c)=k[c(1:3,10,20,23)]; row.names(c)=c("TECG (%)","SD","Sesgo (*10^15)")
c

#Ojo: Â¡buscamos una k en un punto medio! Tomamos k=10
error_CV10 = sapply(train_cv(10), error, test = T)


#################### LEAVE ONE OUT CROSS-VALIDATION #######################

(error_LOOCV = sapply(train_cv(462), error_global, test = T)) # k = 462


############################ CONCLUSIONES #################################

errores_global = list(error_VA[1,], error_Boot[1,], error_Boot_OOB[1,], error_Boot632[1,],
               error_CV10[1,], error_LOOCV)
errores_g0 = list(error_VA[2,], error_Boot[2,], error_Boot_OOB[2,], error_Boot632[2,],
                      error_CV10[2,])
errores_g1 = list(error_VA[3,], error_Boot[3,], error_Boot_OOB[3,], error_Boot632[3,],
                      error_CV10[3,])

### Resumen tasas de error global
 
c_global = cbind(Estimado = sapply(errores_global, mean),
                 SD = round(sapply(errores_global, sd),2),
          Sesgo = signif(sapply(errores_global, function(x){mean(x-mean(x))}),2),
          Lim_inf = c(sapply(errores_global,function(x){mean(x)-1.96*sd(x)})[-(5:6)],NA,NA),
          Lim_sup = c(sapply(errores_global,function(x){mean(x)+1.96*sd(x)})[-(5:6)],NA,NA))
row.names(c_global) = c("Validation Approach", "Bootstrap", "Bootstrap OOB", 
                 ".632 Bootstrap", "10-Fold CV", "Leave One Out CV"); c_global

### Resumen tasas de error grupo 0

c_g0 = cbind(Estimado = sapply(errores_g0, mean),
                 SD = round(sapply(errores_g0, sd),2),
                 Sesgo = signif(sapply(errores_g0, function(x){mean(x-mean(x))}),2),
                 Lim_inf = c(sapply(errores_g0,function(x){mean(x)-1.96*sd(x)})[-(5:6)],NA),
                 Lim_sup = c(sapply(errores_g0,function(x){mean(x)+1.96*sd(x)})[-(5:6)],NA))
row.names(c_g0) = c("Validation Approach", "Bootstrap", "Bootstrap OOB", 
                        ".632 Bootstrap", "10-Fold CV"); c_g0

### Resumen tasas de error grupo 1

c_g1 = cbind(Estimado = sapply(errores_g1, mean),
             SD = round(sapply(errores_g1, sd),2),
             Sesgo = signif(sapply(errores_g1, function(x){mean(x-mean(x))}),2),
             Lim_inf = c(sapply(errores_g1,function(x){mean(x)-1.96*sd(x)})[-(5:6)],NA),
             Lim_sup = c(sapply(errores_g1,function(x){mean(x)+1.96*sd(x)})[-(5:6)],NA))
row.names(c_g1) = c("Validation Approach", "Bootstrap", "Bootstrap OOB", 
                    ".632 Bootstrap", "10-Fold CV"); c_g1


################################## FIN ####################################

###  FUNCIONES UTILIZADAS A LO LARGO DEL SCRIPT:

# Cálculo de errores por clase.
tasas_error = function(modelo, train, corte) {
  if (is.null(train)) {
    pred = as.numeric(predict(modelo, type = "response") > corte)
    y = SAheart$chd }
  else {
    pred = as.numeric(predict(modelo, newdata = SAheart[-train,],
                              type = "response") > corte)
    y = SAheart$chd[-train] }
  return(100*c(mean(y != pred), mean(y[y == 0] != pred[y == 0]), 
               mean(y[y == 1] != pred[y == 1]))) }
# Error global, error grupo 0, error grupo 1.


# Obtiene especificidades, sensibilidades y puntos de corte del modelo m sobre el 
# conjunto de entrenamiento (train).
esp_sens = function(m, train) {
  pred = predict(m, SAheart[-train,], type = "response")
  perf = performance(prediction(pred, SAheart$chd[-train]), "tpr", "fpr")
  return(cbind(1-perf@x.values[[1]], perf@y.values[[1]], perf@alpha.values[[1]])) }


# Área bajo la curva ROC del modelo m sobre el conjunto de entrenamiento (train).
AUC = function(m, train) {
  pred = predict(m, SAheart[-train,], type = "response")
  return(performance(prediction(pred, SAheart$chd[-train]), "auc")@y.values[[1]]) }

# Función de cálculo de errores de clasificación globales y por grupo:
# Si test == TRUE, regresa los errores promediados sobre el conjunto de prueba.
# De lo contrario, regresa los errores sobre el conjunto de entrenamiento.

error = function(train, test) { 
  if (test) newdata = SAheart[-train,]
  else newdata = SAheart[train,]
  pred = as.numeric(predict(glm(modelo$formula, family = binomial(link = "logit"),
                                data = SAheart[train,]), newdata = newdata, 
                            type = "response") > 0.5)
  return(c(100*mean(newdata$chd != pred),
           100*mean(newdata$chd[newdata$chd==0] != pred[newdata$chd==0]), 
           100*mean(newdata$chd[newdata$chd==1] != pred[newdata$chd==1]))) }

error_global = function(train, test) { 
  if (test) newdata = SAheart[-train,]
  else newdata = SAheart[train,]
  pred = as.numeric(predict(glm(modelo$formula, family = binomial(link = "logit"),
                                data = SAheart[train,]), newdata = newdata, 
                            type = "response") > 0.5)
  return(100*mean(newdata$chd != pred)) }

