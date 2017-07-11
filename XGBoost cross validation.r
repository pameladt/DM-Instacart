

library("xgboost")

archivo_grid    <- "C:/Users/Pamela/Documents/GitHub/DM-Instacart/grid.txt"

archivo_salida  <- "C:/Users/Pamela/Documents/GitHub/DM-Instacart/salida.txt"

archivo_entrada <- "Instacart"

if( !file.exists( archivo_salida) )
{
cat( "fecha", "entrada", "algoritmo", "obs", 
     "eta",  "alpha", "lambda", "gamma",
     "colsample_bytree", "min_child_weight", "max_depth", "iteracion", "tiempo", "logloss_min",  
     "\r\n", sep="\t", file=archivo_salida, fill=FALSE, append=FALSE 
    ) 
}


library(data.table)
library(dplyr)
library(tidyr)
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
    #prod_mean_days_since_prior = mean(days_since_prior_order, na.rm = T),
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
         time_since_last_order = days_since_prior_order)

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

rm(ordert, prd, users)
gc()

#factor to number for xgBoost

data$aisle <- as.numeric(data$aisle)
data$department <- as.numeric(data$department)


data<- data%>% select(-aisle)
print(head(data))


# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$eval_set == "train",])
train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

print(head(train))

rm(data)
gc()


#subtrain <- train 
subtrain <- train %>% sample_frac(0.1)

nrow(subtrain)

dtrain <- xgb.DMatrix(as.matrix(subtrain %>% select(-reordered)), label = subtrain$reordered)

vnround <- 500

peta               =  0.1
palpha             =  2e-05 
plambda            =  10
pgamma             =  0.70
pmin_child_weight  =  10
pmax_depth         =  6


set.seed( 102191  )

t0 =  Sys.time()
cv = xgb.cv( 
		data = dtrain,           missing = 0 ,
                #scale_pos_weight = vscale_pos_weight,
		stratified = TRUE,       nfold = 5 ,
		eta = peta, 
 		subsample =  1, #0.76, 
 		colsample_bytree = 1, #0.95, 
 		min_child_weight = pmin_child_weight, 
 		max_depth = pmax_depth,
 		alpha = palpha, lambda = plambda, gamma = pgamma,
 		#objective="multi:softprob",         num_class=2,
        	#objective="binary:logistic",   
		objective = "reg:logistic",
 		eval_metric = "logloss",            maximize =FALSE,
                nround= vnround,   early_stopping_rounds = 100,
                nthread=32
		)

t1 = Sys.time()





#Comparo

tiempo        <- as.numeric(  t1 - t0, units = "secs")
logloss_min       <- min( cv$evaluation_log[ , test_logloss_mean] )
iteracion_max <- which.min(  cv$evaluation_log[ , test_logloss_mean] )


cat( format(Sys.time(), "%Y%m%d %H%M%S"), archivo_entrada, "xgboost", "sinpeso",
     peta,  palpha, plambda, pgamma,
     1,  pmin_child_weight, pmax_depth, iteracion_max, tiempo, logloss_min,  
     "\r\n", sep="\t", file=archivo_salida, fill=FALSE, append=TRUE 
    ) 

