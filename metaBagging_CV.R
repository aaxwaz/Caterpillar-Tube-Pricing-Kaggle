setwd('c:\\users\\1989\\desktop\\CAT\\FINAL\\CV_best\\metaBagging\\CV')
rm(list = ls())
require(xgboost)
require(Matrix)
require(TunePareto)
require(ROCR)
require(randomForest)
require(data.table)
options( java.parameters = "-Xmx3g" )
require(extraTrees)

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

# meta-bagging sampling approach 
bagging_split <- function(tube_ids, percentage){
  temp = data.table(ids = tube_ids, num = 1:length(tube_ids))
  setkey(temp, ids)
  if(percentage == 100){ # with replacement 
    ids_set = unique(tube_ids)
    bag_sample = sample(1:length(ids_set), length(ids_set), replace=T)
    BAG = ids_set[bag_sample]
    OOB = ids_set[-bag_sample]
    bag_data = temp[BAG,allow.cartesian = TRUE]$num
    oob_data = temp[OOB,allow.cartesian = TRUE]$num
  }
  else{ # w/o replacement
    ids_set = unique(tube_ids)
    bag_sample = sample(1:length(ids_set), percentage/100 * length(ids_set), replace=F)
    BAG = ids_set[bag_sample]
    OOB = ids_set[-bag_sample]
    bag_data = temp[BAG,allow.cartesian = TRUE]$num
    oob_data = temp[OOB,allow.cartesian = TRUE]$num
  }
  list(bag_data, oob_data)
}


evaluation = function(preds, actuals,min_limit = 0.0){
  preds[preds<min_limit] = min_limit
  n = length(preds)
  diffs = log(preds + 1) - log(actuals+1)
  sqrt(1/n*sum(diffs^2))
}


##### start CV ##### 20 trees: 0.220-0.222(w/o NN); 15 trees: 0.2197(w/ NN); only xgb: 0.229
# params
### params
bag_size = 30
K = 5
N_transform = 15
bagging_ratio = 85
depth = 7
rounds = 900
eta = 0.1
min_child_weight = 1
# ET
et_ntree = 5
et_mtry = 70
# rf 
ntree = 15
mtry = 60

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

# load and process data 

### Start !!

logrmse_set = rep(0, K)
logrmse_set_median = rep(0, K)
for(cv in 1:K){
  
  valid_file = paste('CV_set_', as.character(cv), '.csv', sep='')
  valid_df = read.csv(valid_file, stringsAsFactors = F)
  if(exists('train_df')) rm(train_df)
  for(rest in 1:5){
    if(rest == cv) next
    temp_name = paste('CV_set_', as.character(rest), '.csv', sep='')
    temp_df = read.csv(temp_name)
    if(exists('train_df'))
    {
      train_df = rbind(train_df, temp_df)
    }
    else
    {
      train_df = temp_df
    }
    
  }
  
  # data transform
  # take out tube_assembly_ids
  train_tube_ids = as.character(train_df$tube_assembly_id)
  train_df = subset(train_df, select = -tube_assembly_id)
  valid_df = subset(valid_df, select = -tube_assembly_id)
  
  # data process/ N transform
  train_df = train_df^(1/N_transform)
  train_df = as.data.frame(train_df)
  valid_df = valid_df^(1/N_transform)
  valid_df = as.data.frame(valid_df)
  
  valid_df_cost = valid_df$cost # use as truth later
  valid_df = subset(valid_df, select=-cost) # take out cost column
  valid_fea_num = ncol(valid_df)
 
  valid_df = cbind(valid_df, data.frame(et_pred=0)) # create new col et_pred for valid_df
  valid_df = cbind(valid_df, data.frame(rf_pred=0)) # create new col rf_pred for valid_df
  #valid_df = cbind(valid_df, data.frame(base_pred=0)) # create new col base_pred
  
  # BAGGING START!
  final_result = 0
  for(b in 1:bag_size){
    
    cat("\n\nBagging round: ", b, "\n")
    
    temp_samples = bagging_split(train_tube_ids, bagging_ratio)
    BAG_train = train_df[temp_samples[[1]],] # around 30158 rows 
    OOB_train = train_df[temp_samples[[2]],] # around 11136 rows

    # base1 - rf
    rf = randomForest(cost~., OOB_train, ntree = ntree, do.trace = 0, mtry = mtry)
    rf_pred_BAG = predict(rf, BAG_train[,-ncol(BAG_train)]) # preditions on bag data, in log(1+x) level 
    rf_pred_valid = predict(rf, valid_df[,1:valid_fea_num]) # preditions on valid data, in log(1+x) level 
    
    # base3 - ET
    et = extraTrees(x = OOB_train[,-ncol(OOB_train)], y = OOB_train$cost, ntree = et_ntree, numThreads = 4, mtry = et_mtry, numRandomCuts = 3)
    et_pred_BAG = predict(et, BAG_train[,-ncol(BAG_train)])
    et_pred_valid = predict(et, valid_df[,1:valid_fea_num])
    
    # add in new col to BAG_train
    col_insert = which(names(BAG_train) == 'cost')[1]-1 # the col before cost
    BAG_train = cbind(BAG_train[,1:col_insert, drop=F], data.frame(et_pred=et_pred_BAG), data.frame(rf_pred=rf_pred_BAG),
                      BAG_train[,col_insert+1,drop=F])
    # update test_df$rf_pred col 
    valid_df$et_pred = et_pred_valid
    valid_df$rf_pred = rf_pred_valid
    
    # prepare for xgb train&predict
    train_matrix = sparse.model.matrix(~., data = BAG_train[,-ncol(BAG_train)]) # sparse matrix 
    valid_matrix = sparse.model.matrix(~., data = valid_df)
    dtrain = xgb.DMatrix(train_matrix, label = BAG_train$cost)
    
    bst = xgb.train(param, dtrain, nround = rounds, label = BAG_train$cost)
    
    pred = predict(bst, valid_matrix)
    pred[pred<0] = 0 # clipping 1
    
    cat(pred[1:10]^N_transform)
    
    final_result = cbind(final_result, pred)
    
  }
  
  final_result = final_result[,2:ncol(final_result)] # take out dummy col of zeros
  
  cat("Total cols for final_result: ", ncol(final_result), "\n")
  
  median_results = (apply(final_result, FUN=median, MARGIN=1))^N_transform
  mean_results = (apply(final_result, FUN=mean, MARGIN=1))^N_transform
  
  # clipping 2
  min_limit = 0.1
  median_results[median_results<min_limit] = min_limit
  mean_results[mean_results<min_limit] = min_limit
  
  temp1 = evaluation(mean_results, valid_df_cost^N_transform)
  temp2 = evaluation(median_results, valid_df_cost^N_transform)
  logrmse_set[cv] = temp1
  logrmse_set_median[cv] = temp2
  cat('\nMean result: ', temp1, ' Median result: ', temp2)
}
cat('\nMean: ')
print(logrmse_set)
cat("Avg is: ", mean(logrmse_set), " std: ", sd(logrmse_set), "\n")
cat('\nMedian: ')
print(logrmse_set_median)
cat("Avg is: ", mean(logrmse_set_median), " std: ", sd(logrmse_set_median), "\n")
### END CV!


### End!!
m = xgb.importance(train_matrix@Dimnames[[2]], model = bst)
write.csv(m, 'importance_85_15.csv')










