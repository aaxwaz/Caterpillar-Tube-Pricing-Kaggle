setwd('c:\\users\\weimin-srx\\desktop\\CAT')
rm(list = ls())
require(xgboost)
require(Matrix)
require(TunePareto)
require(ROCR)


df = read.csv('train_filtered.csv', stringsAsFactors = F)
df$total_weight[is.na(df$total_weight)] = 0

CV_split <- function(tube_ids, K){
  temp = data.frame(ids = tube_ids, num = 1:length(tube_ids))
  ids_set = unique(tube_ids)
  result = generateCVRuns(1:length(ids_set), ntimes = 1, nfold = K, stratified = F)
  result = result[[1]]
  final_list = list()
  for(cv in 1:K){
    final_list = c(final_list, list(temp[temp$ids %in% ids_set[result[[cv]]],]$num))
  }
  final_list
}

evaluation = function(preds, actuals,min_limit = 0.0){
  preds[preds<min_limit] = min_limit
  n = length(preds)
  diffs = log(preds + 1) - log(actuals+1)
  sqrt(1/n*sum(diffs^2))
}

##### CV #####
eta = 0.1
rounds = 1000 #  5Fold 0.227~0.228, 0.2287, 0.2241
min_child_weight = 1
K = 5
rounds_of_running = 2

#### data processingdf = read.csv('train_add_factorized_NN.csv', stringsAsFactors = F)
df$total_weight[is.na(df$total_weight)] = 0

#### data processing
tube_ids = df$tube_assembly_id 
df = subset(df, select = -tube_assembly_id)

df = log(1 + df)

# start CV!
for (runs in 1:rounds_of_running){
  cat("Started rounds: ", runs, "\n")
  for (depth in seq(from = 7, to = 7, by = 1))
  {
    cat('\n')
    
    cat("\neta: ",eta," Depth: ", depth, " iter rounds: ", rounds, " child: ",min_child_weight," rmse: ")
    
    cv_splits = CV_split(tube_ids, K)
    
    logrmse_set = rep(0,K)
    for(cv in 1:K){
      valid_index = cv_splits[[cv]]
      valid_df = df[valid_index,]
      train_df = df[-valid_index,]
      
      #preparing training data 
      train_matrix = sparse.model.matrix(~., data = train_df[,-ncol(train_df)]) # sparse matrix 
      valid_matrix = sparse.model.matrix(~., data = valid_df[,-ncol(valid_df)])
      dtrain = xgb.DMatrix(train_matrix, label = train_df$cost)
      dvalid = xgb.DMatrix(valid_matrix, label = valid_df$cost)
      watchlist <- list(eval=dvalid)
      
      param = list(booster = 'gbtree', 
                   "objective"="reg:linear",
                   'nthreads' = 4,
                   #'lambda' = 0.00005,
                   #'alpha' = 0.000001,
                   min_child_weight = min_child_weight,
                   #subsample = 0.7,
                   #gamma = 0.9,
                   #colsample_bytree = 0.8,
                   max.depth = depth,
                   eta = eta)
      bst = xgb.train(param, dtrain, nround = rounds, label = train_df$cost)#, verbose = 1, watchlist = watchlist, early.stop.round = 50, eval_metric='rmse')
      
      #xgb.cv(param, dtrain, nround = rounds, label = train_df$cost, verbose = 1, watchlist = watchlist, early.stop.round= 2, nfold = 5)
      
      pred = predict(bst, valid_matrix)
      pred = exp(pred) - 1
      
      temp = evaluation(pred, exp(valid_df$cost)-1)
      logrmse_set[cv] = temp
      print(temp)
    }
    print(logrmse_set)
    cat("Avg is: ", mean(logrmse_set), " std: ", sd(logrmse_set), "\n")
  }
}
  

m = xgb.importance(train_matrix@Dimnames[[2]], model = bst)

