
###########################################################################################################
#
# Kaggle Instacart competition
# July 2017
##############################################################

library(data.table)
library(dplyr)
library(tidyr)
library(Ckmeans.1d.dp)# Products selection is based on product by product binary classification, with a global threshold (0.21)
#
#############################################
library(stringr)


# Load Data ---------------------------------------------------------------
path <- "C:/Users/Pamela/Documents/GitHub/Instacart data"

aisles <- fread(file.path(path, "aisles.csv"))
departments <- fread(file.path(path, "departments.csv"))
orderp <- fread(file.path(path, "order_products__prior.csv"))
ordert <- fread(file.path(path, "order_products__train.csv"))
orders <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))


# Reshape data ------------------------------------------------------------
aisles$aisle <- as.factor(aisles$aisle)
departments$department <- as.factor(departments$department)
orders$eval_set <- as.factor(orders$eval_set)
products$product_name <- as.factor(products$product_name)

products <- products %>% 
  inner_join(aisles) %>% inner_join(departments) %>% 
  select(-aisle_id, -department_id)

rm(aisles, departments)

products$perecedero <- ifelse(products$department=="produce", 1, ifelse(products$department=="dairy eggs", 1,0))

products <- products %>% 
    mutate(organic=ifelse(str_detect(str_to_lower(products$product_name),'organic'),1,0))




#### ONE HOT ENCODING####

#depts <- data.frame( product_id = products$product_id, dpt_ = gsub(" ", "_", products$department) )

#depts_matrix <- ( model.matrix( ~0+product_id+dpt_, data = depts ) )

#df_depts <- data.frame( depts_matrix )

#products <- products %>% inner_join( df_depts ) %>% select (-department)

#rm( depts, depts_matrix, df_depts ) 

#### ONE HOT ENCODING####


#print(head(orders_products))
print(head(products ))


ordert$user_id <- orders$user_id[match(ordert$order_id, orders$order_id)]

orders_products <- orders %>% inner_join(orderp, by = "order_id")

rm(orderp)
gc()


# Products ----------------------------------------------------------------
prd <- orders_products %>%
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(product_time = row_number()) %>%
  ungroup() %>%
  group_by(product_id) %>%
  summarise(
    prod_mean_days_since_prior = mean(days_since_prior_order, na.rm = T),
    prod_orders = n(),
    prod_distinct_users = n_distinct(user_id),
    prod_reorders = sum(reordered),
    prod_first_orders = sum(product_time == 1),
    prod_second_orders = sum(product_time == 2)
  )

print(head(prd))


prd$prod_reorder_probability <- prd$prod_second_orders / prd$prod_first_orders
prd$prod_reorder_times <- 1 + prd$prod_reorders / prd$prod_first_orders
prd$prod_reorder_ratio <- prd$prod_reorders / prd$prod_orders


prd <- prd %>% inner_join(products)

prd <- prd %>% select(-prod_reorders, -prod_first_orders, -prod_second_orders, -product_name)

print(head(prd))


rm(products)
gc()

# Users -------------------------------------------------------------------
#user's prior data & behaviour
users <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    user_orders = max(order_number),
    user_period = sum(days_since_prior_order, na.rm = T),
    user_mean_days_since_prior = mean(days_since_prior_order, na.rm = T)
  )

print(head(users))

#user's order prior data: buying data
us <- orders_products %>%
  group_by(user_id) %>%
  summarise(
    user_total_products = n(),
    user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
    user_distinct_products = n_distinct(product_id)
  )

users <- users %>% inner_join(us)
users$user_average_basket <- users$user_total_products / users$user_orders


#user + order: keep train and test data

us <- orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set,
         time_since_last_order = days_since_prior_order
	   )
print(head(us))

us$days_since_prior_7 <- ifelse(us$time_since_last_order==7,1,0)

#To train and test data attach the calculus from Prior info
users <- users %>% inner_join(us)

rm(us)
gc()

print(head(orders_products))

# Database ----------------------------------------------------------------
data <- orders_products %>%
  group_by(user_id, product_id) %>% 
  summarise(
    up_orders = n(), #seems like an error, the dataset is at product level, not order
    #up_orders = n_distinct(order_number), #same result as above
    up_first_order = min(order_number),
    up_last_order = max(order_number),
    up_average_cart_position = mean(add_to_cart_order))

rm(orders_products, orders)

print(head(data))

data <- data %>%
  group_by(user_id) %>%
  mutate(up_best_selling = dense_rank(desc(up_orders)))

data$up_best_selling <- ifelse(data$up_best_selling==1,1,0)
data$up_all_orders <- ifelse(data$up_orders==data$up_last_order,1,0)


print(head(data))


data <- data %>% 
  inner_join(prd, by = "product_id") %>%
  inner_join(users, by = "user_id")

data$up_order_rate <- data$up_orders / data$user_orders
data$up_orders_since_last_order <- data$user_orders - data$up_last_order
data$up_order_rate_since_first_order <- data$up_orders / (data$user_orders - data$up_first_order + 1)

# # ordenes con el producto/sum(ordenes) desde que empecé a pedir el producto

#agrego el target reordered para train
data <- data %>% 
  left_join(ordert %>% select(user_id, product_id, reordered), 
            by = c("user_id", "product_id"))

data$last_purchase_same_day <- ifelse(data$user_orders==data$up_last_order & data$time_since_last_order==0,1,0)


rm(ordert, prd, users)
gc()




#factor to number for xgBoost

data$aisle <- as.numeric(data$aisle)
data$department <- as.numeric(data$department)


#data<- data%>% select(-aisle,-days_since_prior_7)
print(head(data))


# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$eval_set == "train",])
train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

print(head(train))


test <- as.data.frame(data[data$eval_set == "test",])
test$eval_set <- NULL
test$user_id <- NULL
test$reordered <- NULL

print(head(test))

rm(data)
gc()


# Model -------------------------------------------------------------------
library(xgboost)

params <- list(
  "objective"           = "reg:logistic",
  "eval_metric"         = "logloss",
  "eta"                 = 0.15,#0.1
  "max_depth"           = 6,
  "min_child_weight"    = 10,
  "gamma"               = 0.70,
  "subsample"           = 1, #0.76,
  "colsample_bytree"    = 1, #0.95,
  "alpha"               = 2e-05,
  "lambda"              = 10,
  "seed"			= 102191
)

subtrain <- train 
#subtrain <- train %>% sample_frac(0.1)
X <- xgb.DMatrix(as.matrix(subtrain %>% select(-reordered)), label = subtrain$reordered)
model <- xgboost(data = X, params = params, nrounds = 500)

importance <- xgb.importance(colnames(X), model = model)
xgb.ggplot.importance(importance)

rm(X, importance, subtrain)
gc()


# Apply model -------------------------------------------------------------
X <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))
test$reordered <- predict(model, X)

test$reordered <- (test$reordered > 0.21) * 1

submission <- test %>%
  filter(reordered == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(submission, file = "submit_015eta_500_trees.csv", row.names = F)

