#Installation of required packages:

if(!require(dplyr)) install.packages("dplyr", 
                                     repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret",                                   #Additional packages will be required for 
                                     repos = "http://cran.us.r-project.org")    #running machine learning models. A prompt for
                                                                                #the installation of those will be given while
                                                                                #the caret "train" command is executed.
                                                                                #Entering "Yes" as the response will install
                                                                                #the required package.
if(!require(data.table)) install.packages("data.table",                          
                                          repos = "http://cran.us.r-project.org") 
if(!require(gridExtra)) install.packages("gridExtra",                            
                                         repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats",                            
                                         repos = "http://cran.us.r-project.org")

#Loading required libraries for the script:
library(dplyr)
library(tidyverse)
library(caret)
library(data.table)
library(gridExtra)
library(matrixStats)

#The dataset required for this project can be downloaded in .RData format from
#https://github.com/adi297/CYOProject.git

load("liver_dat.RData")

#Preliminary examination of the dataset:
str(liver_dat)
head(liver_dat)

#The "Dataset" variable has the presence or absence of liver disease coded as
#1 and 2 respectively.

#The "Age" variable is to be examined for variation, by itself, and conditional
#gender:

liver_dat %>% ggplot(aes(Age)) + 
  geom_histogram(color = "black", fill = "blue", binwidth = 1)

liver_dat %>% filter(Gender == "Female") %>% 
  ggplot(aes(Age)) + geom_histogram(color = "black", fill = "blue", binwidth = 1)

liver_dat %>% filter(Gender == "Male") %>%
  ggplot(aes(Age)) + geom_histogram(color = "black", fill = "blue", 
                                                binwidth = 1)

#Presence of significant peaks is observed, but no fixed pattern.

#Presence or absence of liver disease conditional of gender is to be examined:

liver_dat %>% mutate(Dataset = as.factor(Dataset)) %>% 
  ggplot(aes(Gender, fill = Dataset)) + geom_bar(position = "dodge")

liver_dat_gender <- liver_dat %>% group_by(Gender) %>% 
  summarise(Count = n()) %>% as.data.frame()

liver_dat_female <- liver_dat %>% filter(Gender == "Female") %>% 
  group_by(Dataset) %>% 
  summarise(Count = n()) %>% as.data.frame()

liver_dat_male <- liver_dat %>% filter(Gender == "Male") %>% 
  group_by(Dataset) %>% 
  summarise(Count = n()) %>% as.data.frame()

liver_dat_gender <- liver_dat_gender %>% 
  mutate(Diseased = c(liver_dat_female[1,2], liver_dat_male[1,2]),
         Healthy = c(liver_dat_female[2,2], liver_dat_male[2,2]))

liver_dat_gender

#Different proportion of males and females, might suggest an effect on 
#prevalence of the disease. The differences could be quantified using odds
#ratios:

liver_dat_gender <- liver_dat_gender %>% 
  mutate(Odds_Ratio = ((Diseased/Count)/(Healthy/Count)))

liver_dat_gender %>% knitr::kable()

liver_dat_exp <- liver_dat %>% 
  mutate(Odds_Ratio = ifelse(Gender == "Female", liver_dat_gender$Odds_Ratio[1], 
                             liver_dat_gender$Odds_Ratio[2]), 
         y = as.factor(Dataset))


#Gender based odds ratios along with measurements of other parameters in the
#data table could serve as a good predictor space for machine learning 
#algorithms. The following dataset is accordingly prepared:

liver_dat_exp_set <- liver_dat_exp %>% select(-c(Gender, Dataset, 
                                                 Albumin_and_Globulin_Ratio))

names_x <- list(NULL, names(liver_dat_exp_set[,1:9]))

x <- matrix(c(liver_dat_exp_set$Age, liver_dat_exp_set$Total_Bilirubin, 
              liver_dat_exp_set$Direct_Bilirubin, 
              liver_dat_exp_set$Alkaline_Phosphotase, 
              liver_dat_exp_set$Alamine_Aminotransferase, 
              liver_dat_exp_set$Aspartate_Aminotransferase, 
              liver_dat_exp_set$Total_Protiens, liver_dat_exp_set$Albumin, 
              liver_dat_exp_set$Odds_Ratio), 583, 9, dimnames = names_x)

y <- liver_dat_exp_set$y

liv_exp_set <- list(x = x, y = y)


#Predictors are scaled to determine distances between them:

x_tmp <- with(liv_exp_set, sweep(x, 2, colMeans(x)))
x_scaled <- sweep(x_tmp, 2, colSds(liv_exp_set$x), FUN = "/")
labels <- colnames(liv_exp_set$x)
colnames(x_scaled) <- labels


d_samples <- dist(x_scaled)

dist_2to2 <- as.matrix(d_samples)[1, liv_exp_set$y == "2"]
mean(dist_2to2[2:length(dist_2to2)])
dist_2to1 <- as.matrix(d_samples)[1, liv_exp_set$y == "1"]
mean(dist_2to1)

#Heatmap of distances between scaled predictors:
heatmap(as.matrix(dist(t(x_scaled))), 
        col = RColorBrewer::brewer.pal(11, "Spectral"), 
        labRow = NA, labCol = NA)

#Cluster based grouping:
d<-dist(t(x_scaled))
h <- hclust(d)
groups <- cutree(h, k = 5)
groups %>% knitr::kable()

#Principal Component Analysis:
pca <- prcomp(x_scaled)

summary(pca)
plot(pca$sdev, type = "b", xlab = "Principal Component Number")
title(main = "Principal Component Analysis")
points(pca$sdev, cex = .5, col = "dark red")
points(9 , pca$sdev[9], cex = 3, col = "red")
lines(pca$sdev, col = "blue")
axis(1, 0:9, col.axis = "blue")
text(x = 9, y = 0.5, labels = "PC9") 
text(x = 7.8, y = 0.6, labels = "Full Cumulative Proportion Of Variance")

pca_dat <- as.data.frame(pca$x)

#Full variance is accounted for at the 9th principal component:
pca_9 <- pca_dat %>% gather("PC", "Val", factor_key = TRUE)
data.frame(PC = pca_9$PC, Val = pca_9$Val, type = factor(liv_exp_set$y)) %>% 
  ggplot(aes(PC, Val, fill = type)) + geom_boxplot()

#Training and test set creation:
set.seed(1, sample.kind = "Rounding") 
test_index <- createDataPartition(liv_exp_set$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- liv_exp_set$y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- liv_exp_set$y[-test_index]

train_set <- data.frame(x = train_x, y = train_y)
test_set <- data.frame(x = test_x, y = test_y)

#Training and tests sets both contain approximately equal proportions of 
#cases with and without liver disease:
mean(train_set$y == 2)

mean(test_set$y == 2)

#Predictions based on k-means:
predict_kmeans <- function(x, k) {
  centers <- k$centers  
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))
}

set.seed(1, sample.kind = "Rounding")
k <- kmeans(train_x, centers = 13)

res_kmeans <- predict_kmeans(test_x, k)
pred_kmeans <- ifelse(res_kmeans == 1, "2", "1")
pred_kmeans <- as.factor(pred_kmeans)
cm_kmeans <- confusionMatrix(pred_kmeans, test_y)
cm_kmeans$overall["Accuracy"]

#Predictions based on the "Boosted Classification Trees" model:
set.seed(1, sample.kind = "Rounding")
ada_fit <- train(y ~ ., method = "ada", data = train_set)
pred_ada <- predict(ada_fit, test_set, type = "raw")
cm_ada <- confusionMatrix(pred_ada, test_set$y)
cm_ada$overall["Accuracy"]

#Predictions based on the "AdaBoost Classification Trees" model:
set.seed(1, sample.kind = "Rounding")
adaboost_fit <- train(y ~ ., method = "adaboost", data = train_set)
pred_adaboost <- predict(adaboost_fit, test_set, type = "raw")
cm_adaboost <- confusionMatrix(pred_adaboost, test_set$y)
cm_adaboost$overall["Accuracy"]

#Predictions based on the "Distance Weighted Discrimination with 
#Radial Basis Function Kernel" model:
set.seed(1, sample.kind = "Rounding")
dwdRadial_fit <- train(y ~ ., method = "dwdRadial", data = train_set)
pred_dwdRadial <- predict(dwdRadial_fit, test_set, type = "raw")
cm_dwdRadial <- confusionMatrix(pred_dwdRadial, test_set$y)
cm_dwdRadial$overall["Accuracy"]

#Predictions based on the "k - nearest neighbors" model:
set.seed(1, sample.kind = "Rounding")
knn_fit <- train(y ~ ., method = "knn", data = train_set, 
                 tuneGrid = data.frame(k = seq(3, 21, 2)))
pred_knn <- predict(knn_fit, test_set, type = "raw")
cm_knn <- confusionMatrix(pred_knn, test_set$y)
cm_knn$overall["Accuracy"]

#Optimal k:
best_k <- knn_fit$bestTune
best_k

#k Optimization Plot:
ggplot(knn_fit)

#Predictions based on the "RandomForest" model:
set.seed(1, sample.kind = "Rounding")
rf_fit <- train(y ~ ., method = "rf", data = train_set, 
                tuneGrid = data.frame(mtry = c(3, 5, 7, 9)), importance = TRUE)
pred_rf <- predict(rf_fit, test_set, type = "raw")
cm_rf <- confusionMatrix(pred_rf, test_set$y)
cm_rf$overall["Accuracy"]

#Optimal predictors:
best_mtry_rf <- rf_fit$bestTune
best_mtry_rf

#Variable Importance:
varImp(rf_fit)

#Predictor number optimization:
ggplot(rf_fit)

#Predictions based on the "Oblique Random Forest" model:
set.seed(1, sample.kind = "Rounding")
ORFpls_fit <- train(y ~ ., method = "ORFpls", data = train_set, 
                tuneGrid = data.frame(mtry = c(3, 5, 7, 9)), importance = TRUE)
pred_ORFpls <- predict(ORFpls_fit, test_set, type = "raw")
cm_ORFpls <- confusionMatrix(pred_ORFpls, test_set$y)
cm_ORFpls$overall["Accuracy"]

#Optimal predictors:
best_mtry_ORFpls <- ORFpls_fit$bestTune
best_mtry_ORFpls

#Variable Importance:
varImp(ORFpls_fit)

#Predictor number optimization:
ggplot(ORFpls_fit)

#Model declaration for ensemble:
models <- c("kmeans", "ada", "adaboost", "dwdRadial", "knn", "rf")              #ORFpls was omitted due to time constraints.

#Ensemble model fitting function:
ens_fit <- lapply(models, function(model){
  if(model == "kmeans"){
    print(model)
    predict_kmeans <- function(x, k) {
      centers <- k$centers  
      distances <- sapply(1:nrow(x), function(i){
        apply(centers, 1, function(y) dist(rbind(x[i,], y)))
      })
      max.col(-t(distances))
    }
    
    set.seed(1, sample.kind = "Rounding")
    k <- kmeans(train_x, centers = 13)
    
    res_kmeans <- predict_kmeans(test_x, k)
    pred_kmeans <- ifelse(res_kmeans == 1, "2", "1")
  }
  else{
    print(model)
    train(y ~ ., method = model, data = train_set)
  }})

names(ens_fit) <- models

#Ensemble prediction generating function:
pred_ens <- sapply(ens_fit[2:6], function(object) 
  predict(object, newdata = test_set))


pred_ens_dat <- as.data.frame(pred_ens)
pred_ens_dat <- pred_ens_dat %>% 
  mutate(kmeans = as.factor(ens_fit$kmeans), ada = as.factor(ada), 
         adaboost = as.factor(adaboost), dwdRadial = as.factor(dwdRadial), 
         knn = as.factor(knn), 
         rf = as.factor(rf))

#Ensemble accuracy values:
pred_ens_dat <- as.matrix(pred_ens_dat)
num <- seq(1,6)
accuracy <- sapply(num, function(x){ 
  cm_acc <- confusionMatrix(data = as.factor(pred_ens_dat[,x]), 
                            reference = test_set$y)$overall["Accuracy"]
  tibble(accuracy = cm_acc)
}) 
names(accuracy) <- c("ada", "adaboost", "dwdRadial", "knn", "rf", "kmeans")
acc <- as.tibble(accuracy)
acc_all <- gather(acc) %>% pull(value) %>% mean()


#Confusion Matrix based on ensemble predictions:
ensemble <- ifelse(rowMeans(pred_ens_dat == "1") > 0.5, "1", "2")
cm_ens <- confusionMatrix(data = as.factor(ensemble), reference = test_set$y)
cm_ens$overall["Accuracy"]

all_models <- c("kmeans", "ada", "adaboost", "dwdRadial", 
                "knn", "rf", "ORFpls", "ensemble")
all_accuracies <- c(cm_kmeans$overall["Accuracy"], cm_ada$overall["Accuracy"], 
                    cm_adaboost$overall["Accuracy"], 
                    cm_dwdRadial$overall["Accuracy"], 
                    cm_knn$overall["Accuracy"], cm_rf$overall["Accuracy"], 
                    cm_ORFpls$overall["Accuracy"], cm_ens$overall["Accuracy"])

#Final result table:
results <- data.frame(models = all_models, 
                      accuracies = signif(all_accuracies, digits = 2))
results %>% knitr::kable()

#Overall mean accuracy of all models:
final_acc <- mean(results$accuracies)
final_acc

#Overall standard deviation of all models:
sd_acc <- sd(results$accuracies)
sd_acc

#Results Graph:
results_graph <- results %>% ggplot(aes(models, accuracies, fill = models)) + 
            geom_col() +
            geom_hline(yintercept = 0.73, lty = 2, size = 1) +
            geom_text(aes(x = 0, y = 0.66, label = "Mean Accuracy = 0.73"), 
                      nudge_x = 1.4, nudge_y = 0.02) +
            geom_text(aes(label = accuracies), nudge_y = -0.4) +
            geom_errorbar(aes(ymin = final_acc - 2*sd_acc, 
                              ymax = final_acc + 2*sd_acc))
results_graph

#Saving data for RMarkdown Document:
save(liver_dat, liver_dat_gender, train_set, test_set, dist_2to2, dist_2to1,
     x_scaled, liv_exp_set, predict_kmeans, knn_fit, rf_fit, ORFpls_fit,
      results, results_graph, pca, pca_9, cm_adaboost, file = "liver_rmd_dat.RData")
