library(tidyverse)
library(caret)


#Loading data
heart_data <- read.csv('heart.csv')

#Data Exploration
dim(heart_data)
str(heart_data)

#Checking NAs
sum(is.na(heart_data) == TRUE)

#Data Visualization

#The relationship between age and presence of CAD
heart_data %>% ggplot(aes(target,age))+
  geom_boxplot(aes(fill= as.factor(target)),outlier.colour="blue",alpha=1/4)

#The relationship between resting blood pressure and presence of CAD
heart_data %>% ggplot(aes(target,trestbps))+
  geom_boxplot(aes(fill= as.factor(target)),outlier.colour="blue",alpha=1/4)

#The relationship between serum cholesterol and presence of CAD
heart_data %>% ggplot(aes(target,chol))+
  geom_boxplot(aes(fill= as.factor(target)),outlier.colour="blue",alpha=1/4)

#The relationship between maximum heart rate and presence of CAD
heart_data %>% ggplot(aes(target,thalach))+
  geom_boxplot(aes(fill= as.factor(target)),outlier.colour="blue",alpha=1/4)

#The relationship between patient's sex and presence of CAD
heart_data %>% ggplot(aes(sex)) + 
  geom_bar(mapping = aes(x=as.factor(sex),fill=as.factor(sex))) +
  facet_wrap(.~target)
#Male/Female ratio
table(heart_data$sex)

#The relationship between chest pain type and presence of CAD
heart_data %>% ggplot(aes(cp)) + 
  geom_bar(mapping = aes(x=as.factor(cp),fill=as.factor(cp))) +
  facet_wrap(.~target)

#The relationship between slope of the peak exercise ST segment and presence of CAD
heart_data %>% ggplot(aes(slope)) + 
  geom_bar(mapping = aes(x=as.factor(slope),fill=as.factor(slope))) +
  facet_wrap(.~target)

#The relationship between resting electrocardiographic results and presence of CAD
heart_data %>% ggplot(aes(restecg)) + 
  geom_bar(mapping = aes(x=as.factor(restecg),fill=as.factor(restecg))) +
  facet_wrap(.~target)

#The relationship between having a heart defect and presence of CAD
heart_data %>% ggplot(aes(thal)) + 
  geom_bar(mapping = aes(x=as.factor(thal),fill=as.factor(thal))) +
  facet_wrap(.~target)

#Creating test and train data sets 
set.seed(10,sample.kind = "Rounding")
test_index <- createDataPartition(heart_data$target ,times = 1, p=.3,list = FALSE)
train_heart <- heart_data[-test_index, ]
test_heart <- heart_data[test_index, ]

#Setting cross validation parameters
control <- trainControl(method = "cv", number = 10, p = .9)

#Logistic regression model
set.seed(10, sample.kind="Rounding")
train_glm <- train(as.factor(target) ~ ., method = "glm", 
                   data = train_heart,family = "binomial",
                   trControl = control) 
glm_pred <- predict(train_glm, test_heart)

logistic_regression <- confusionMatrix(glm_pred, as.factor(test_heart$target), 
                                       mode = "everything",
                                       positive = "1")
logistic_regression

results <- tibble(Model = "logistic_regression", 
                  Accuracy = logistic_regression$overall["Accuracy"], 
                  F1 = logistic_regression$byClass["F1"])
results %>% knitr::kable()

#K-nearest neighbors model
set.seed(10, sample.kind="Rounding")
train_knn <- train(as.factor(target) ~ ., method = "knn", 
                   data = train_heart,
                   tuneGrid = data.frame(k = seq(2, 200, 2)),
                   trControl = control)
#Find best tune
ggplot(train_knn, highlight = TRUE)
train_knn$bestTune
knn_pred <- predict(train_knn, test_heart)

knn <- confusionMatrix(knn_pred, as.factor(test_heart$target), 
                       mode = "everything",
                       positive = "1")
knn

results <- bind_rows(results,
                     data_frame(Model = "K-nearest neighbors", 
                                Accuracy = knn$overall["Accuracy"], 
                                F1 = knn$byClass["F1"]))
results %>% knitr::kable()

#Linear discriminant analysis
set.seed(10, sample.kind="Rounding")
train_lda <- train(as.factor(target) ~ ., method = "lda", data = train_heart,
                   trControl = control)
lda_pred <- predict(train_lda, test_heart)

lda <- confusionMatrix(lda_pred, as.factor(test_heart$target), 
                       mode = "everything",
                       positive = "1")
lda

results <- bind_rows(results,
                     data_frame(Model = "Linear discriminant analysis", 
                                Accuracy = lda$overall["Accuracy"], 
                                F1 = lda$byClass["F1"]))
results %>% knitr::kable()

#Quadratic discriminant analysis
set.seed(10, sample.kind="Rounding")
train_qda <- train(as.factor(target) ~ ., method = "qda", data = train_heart,
                   trControl = control)
qda_pred <- predict(train_qda, test_heart)

qda <- confusionMatrix(qda_pred, as.factor(test_heart$target), 
                       mode = "everything",
                       positive = "1")
qda

results <- bind_rows(results,
                     data_frame(Model = "Quadratic discriminant analysis", 
                                Accuracy = qda$overall["Accuracy"], 
                                F1 = qda$byClass["F1"]))
results %>% knitr::kable()

#Random Forest
set.seed(10, sample.kind="Rounding")
train_rf <- train(as.factor(target) ~ ., method = "rf", 
                  data = train_heart,ntree = 100,
                  tuneGrid = data.frame(mtry =seq(1,25,2)),
                  trControl = control,
                  importance = TRUE)
#Find best tune
ggplot(train_rf)      
train_rf$bestTune
rf_pred <- predict(train_rf, test_heart)
rf <- confusionMatrix(rf_pred, as.factor(test_heart$target), 
                      mode = "everything",
                      positive = "1")
rf

results <- bind_rows(results,
                     data_frame(Model = "Random Forest", 
                                Accuracy = rf$overall["Accuracy"], 
                                F1 = rf$byClass["F1"]))
results %>% knitr::kable()

#Find the most important variables affecting target
varImp(train_rf)
#------------------------------------------------------------------------------#
