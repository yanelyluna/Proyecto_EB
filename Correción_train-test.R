
# Un solo conjunto entrenamiento
set.seed(1); train=sample(1:462,393,replace = FALSE) #85% de las observaciones

## Usando la función tasas_error
#MODELO 3
tasas_error(m3,train = train,test = FALSE,corte = 0.5) # Errores en train 22.9 12.5 42.6
tasas_error(m3,train = train,test = TRUE,corte = 0.5) #Errores en test 27.5 13.3 54.2

modelos = list(m1, m2, m3, m4)
(errores_aparentes = sapply(modelos, tasas_error, train = train,test=FALSE, corte = 0.5))  #TASAS APARENTES
(errores_test = sapply(modelos, tasas_error, train = train,test=TRUE, corte = 0.5))                                                      

c = cbind(sapply(modelos, function(x){x$deviance}), sapply(modelos, AIC), 
          sapply(modelos, BIC), t(errores_aparentes), t(errores_test))

colnames(c) = c("Devianza", "AIC", "BIC", "ApGlob", "Ap0", "Ap1", "TestGlob", 
                "Test0", "Test1")
row.names(c) = paste("Modelo", c(1,2,3,4)); c

###  FUNCIONES UTILIZADAS A LO LARGO DEL SCRIPT: ----------

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
