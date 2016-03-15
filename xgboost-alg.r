
library(xgboost)

training_data = read.csv("data-science-puzzle/train.csv")
to_predict_data = read.csv("data-science-puzzle/test.csv")

names = names(training_data)

target = training_data['target']

training_data_size = nrow(training_data)

drops <- c("target")
training_data = training_data[ , !(names(training_data) %in% drops)]

to_predict_data_ids = to_predict_data['id']

data <- rbind(training_data, to_predict_data)

ids = data['id']
drops <- c("id")
data = data[ , !(names(data) %in% drops)]

data <- data.frame(lapply(data, function(x) as.numeric(x)))

data_avg <- apply(data, 2, mean)
data_std <- apply(data, 2, sd)
features_scaled <- scale(data, center= + data_avg, scale=data_std)

training_data <- features_scaled[1:training_data_size, ]
to_predict_data <- features_scaled[(training_data_size+1) : nrow(features_scaled), ]

training_data_matrix <- data.matrix(training_data)
to_predict_data_matrix <- data.matrix(to_predict_data)
target_data_matrix <- data.matrix(target)

train_size <- training_data_size * .75
train_target <- target_data_matrix[1:train_size,]
test_target <- target_data_matrix[(train_size+1):training_data_size,]

xg_to_predict_data <- xgb.DMatrix(data = to_predict_data_matrix)
xgtrain <- xgb.DMatrix(data = training_data_matrix[1:train_size,], label = train_target)

xgtest <-  xgb.DMatrix(data = training_data_matrix[(train_size+1):training_data_size,], label = test_target)

watchlist <- list(val=xgtest, train=xgtrain)

offset <- 2000
num_rounds <- 5000


evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- cor(preds, as.numeric(labels)) ^ 2
  return(list(metric = "error", value = err))
}

param <- list(objective = "reg:linear",
			        eta = 0.01,
              min_child_weight = 3,
              subsample = .8,
              colsample_bytree = .8,
              scale_pos_weight = 1.0,
			        gamma = 0,
              subsample = 0.5,
              colsample_bytree = 0.5,
              max_depth = 8,
              eval_metric=evalerror)

bst <- xgb.train(params = param, 
        				 data = xgtrain, 
        				 nround = num_rounds, 
        				 print.every.n = 20, 
        				 watchlist=watchlist, 
        				 early.stop.round = 80, 
        				 max_delta_step = 1,
        				 maximize = TRUE)

# val-rmse:1.830478

pred_train <- predict(bst, xgtrain)
pred_test <- predict(bst, xgtest)

prediction <- predict(bst, xg_to_predict_data)
result <- cbind(to_predict_data_ids, prediction)

write.csv(result, 'submission.csv', quote=FALSE, row.names=FALSE)
