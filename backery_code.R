## Packages

library(tidyverse)
library(lubridate)
library(janitor)
library(corrplot)
library(cowplot)
library(skimr)
library(tidymodels)
library(modeltime)
library(timetk)
library(vip)
library(parallel)
library(ranger)
library(glmnet)
library(tictoc)


# Data

## Raw data

# load data

bakery_sales_raw <- read_csv("bakery_sales_2021-2022.csv")

weather_raw <- read_csv("weather_2021.01.01-2022.10.31.csv")



## Data inspection

### Bakery

# bakery data

# Convert variable names into snake case type
bakery_sales <- bakery_sales_raw %>% 
  clean_names()

#bakery_sales %>% glimpse()
bakery_sales %>% skim()


* There is a dot, which is not an article
* *negative quantity may be the quantity of products discarded (Ask the bakery owner). For this analysis, negative quantity will not be considered.
* *prices have changed along the year
* *purchase per person
** What is price == 0?
  
# data_prep2

# Covert price into number

bakery_sales <- bakery_sales %>% 
  mutate(
    unit_price     = str_replace_all(unit_price    , " €", ""),  # Remove the currency symbol
    unit_price     = str_replace_all(unit_price    , ",", "."),  # Replace comma with dot
    unit_price     = as.numeric(unit_price)                  # Convert to numeric
  )

bakery_sales %>% select(article) %>% distinct()


bakery_sales <- bakery_sales %>% 
  filter(article != ".") %>% 
  filter(unit_price > 0 & quantity > 0)

bakery_sales %>% skim()



### Wheather

# wheather data

#weather_raw %>% glimpse()

weather_raw %>% skim()



* tsun has not available data all NAs
* snow - there are 15 days of snow, the others (NAs) will be zero
* wspd and wdir have two NAs, meaning no-wind day - replace NAs with zero
* wpgt is zero when no wind-day. Other NAs will be replace by their average.
* wpgt and press  - NAs replace by their average.

# wheather data

wheather <- weather_raw %>% 
  replace_na(list(snow = 0, wspd = 0, wdir = 0)) %>% 
  mutate(wpgt = if_else(wspd == 0, 0, wpgt),
         wpgt = if_else(is.na(wpgt), mean(wpgt, na.rm = TRUE), wpgt),
         pres = if_else(is.na(pres), mean(pres, na.rm = TRUE), pres)) %>% 
  select(-tsun)

wheather %>% skim()



### Daily bakery data

This task will be carry out with daily data.
Extra features created: day of week, month, customers, revenue

# merge

#How many customers per day?

customer_per_day <- bakery_sales %>% 
  filter(article != ".") %>% 
  filter(unit_price > 0 & quantity > 0) %>%  
  group_by(date, ticket_number) %>% 
  distinct(ticket_number) %>% 
  tally() %>% 
  group_by(date) %>% 
  summarise(customers = sum(n)) %>% 
  ungroup()

bakery_daily_data <- bakery_sales %>% 
  mutate(revenue = unit_price*quantity) %>% 
  group_by(date) %>% 
  summarise(quantity = sum(quantity),
            revenue  = sum(revenue)) %>% 
  ungroup()

df_data <- bakery_daily_data %>% 
  left_join(customer_per_day, by = "date") %>% 
  left_join(wheather, by = "date") %>%
  mutate(
    day_of_week = wday(date, label = TRUE),
    month = month(date, label = TRUE)
  ) 

df_data %>% skim()


#saveRDS(df_data, "df_data_prep.rds")




## Seasonality plots

# seasonality

p1 <- ggplot(df_data, aes(x = date, y = quantity)) +
  geom_line()+
  labs(title = "Products Sold by day",
       x = "",
       y = "quantity") +
  theme_minimal()

p2 <-  ggplot(df_data, aes(x = day_of_week, y = quantity)) +
  geom_boxplot() +
  labs(title = "Distribution of Daily Sales by Day of the Week",
       x = "",
       y = "quantity") +
  theme_minimal()

p3 <- ggplot(df_data, aes(x = month, y = quantity)) +
  geom_boxplot() +
  labs(title = "Distribution of Daily Sales by Month",
       x = "",
       y = "quantity") +
  theme_minimal()

cowplot::plot_grid(p1,p2,p3, ncol = 1)

plot1 <- cowplot::plot_grid(p2,p3, ncol = 1)



## Correlation among predictors

There is a high correlation between max, min and avg temp, and wind speed and wind speed gust, as expected.
Also, customers and revenue are correlated (Endogeneity)

# correl

correlation <- cor(df_data %>% select(-date, -quantity) %>% select_if(is.numeric))

corrplot(correlation)



# Modelling

## Split data into training and testing for time-series modelling

# split_data

set.seed(123)

#initial_split initial_time_split
splits <- initial_time_split(df_data, prop = 0.80)


## Recipes

# recipies

# Recipe base
# This recipe only includes the week and month as time-derived features
# One hot encoding = FALSE to create n-1 dummy variables for week and month (avoid perfect multicollinearity)

recipe_base <- recipe(quantity ~ ., training(splits)) %>%
  recipes::step_string2factor(all_nominal()) %>%
  step_rm(date, revenue)  %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE)%>% 
  step_corr(all_numeric_predictors(), threshold = .5) %>% 
  step_normalize(all_numeric_predictors())

# Recipe base2
# Similar the previous, but hot encoding = TRUE. RF and XGBoost

recipe_base2 <- recipe(quantity ~ ., training(splits)) %>%
  recipes::step_string2factor(all_nominal()) %>%
  step_rm(date, revenue)  %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)%>% 
  step_corr(all_numeric_predictors(), threshold = .5) %>% 
  step_normalize(all_numeric_predictors())

# This recipe incorporates Fourier series and natural splines to account for seasonal effects in the linear regression model.
# One-hot encoding is set to FALSE to avoid multicollinearity issues in the linear regression.

recipe_lm <- recipe(quantity ~ ., training(splits)) %>%
  recipes::step_string2factor(all_nominal()) %>%
  step_fourier(date, period = c(7, 14, 30, 365/2), K = 2) %>%
  step_rm(date, revenue, customers)  %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE)%>%
  step_ns(tavg, deg_free = 5) %>%
  step_corr(all_numeric_predictors(), threshold = .5) %>%
  step_normalize(all_numeric_predictors())

# Similar to the previous recipe, but One-hot encoding = TRUE for XGBoost and Random Forest.

recipe_xgboost <- recipe(quantity ~ ., training(splits)) %>%
  step_fourier(date, period = c(7, 14, 30, 365/2), K = 2) %>%
  recipes::step_string2factor(all_nominal()) %>%
  step_rm(date, revenue, customers)  %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)%>% 
  step_ns(tavg, deg_free = 5) %>% 
  #step_corr(all_numeric_predictors(), threshold = .5) %>% 
  step_normalize(all_numeric_predictors(),-all_outcomes())

# This recipe is specific for ARIMA and uses date to estimate auto-regressive moving -average model

recipe_arima <- recipe(quantity ~ ., data = training(splits)) %>%
  recipes::step_string2factor(all_nominal()) %>%
  step_fourier(date, period = c(7, 14, 30, 365/2), K = 2) %>%
  step_rm(revenue, customers)  %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE)%>% 
  step_ns(tavg, deg_free = 5) %>% 
  step_corr(all_numeric_predictors(), threshold = .5) %>% 
  step_normalize(all_numeric_predictors(),-all_outcomes())


#recipe_xgboost %>% prep() %>% juice() %>% glimpse()


# Models

## LM regression model

# lm

# Base recipe model

wflw_lm_fit1 <- workflow() %>%
  add_model(
    linear_reg("regression") %>% set_engine("lm")
  ) %>%
  add_recipe(recipe_base) %>%
  fit(training(splits))

# Linear reg recipe model

wflw_lm_fit2 <- workflow() %>%
  add_model(
    linear_reg("regression") %>% set_engine("lm")
  ) %>%
  add_recipe(recipe_lm) %>%
  fit(training(splits))



## Regularisation model

This model aims to reduce multicollinearity in the linear regression by shrinking the coefficients.
Elastic net regression balances between Lasso and Ridge regression through the alpha parameter, which combines the penalties of both models.

# elastic_net

elastic_net_spec <- linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet")

wflw_en <- workflow() %>% 
  add_recipe(recipe_lm) %>% 
  add_model(elastic_net_spec)

cores <- makePSOCKcluster(detectCores()-1)
doParallel::registerDoParallel(cores)

tic()

set.seed(345)


cv_folds <- rolling_origin(training(splits), initial = 50, assess = 10, skip = 5, cumulative = FALSE)


tune_en <- tune_grid(
  wflw_en,
  resamples = cv_folds,
  grid      = grid_regular(penalty(), mixture(), levels = 10)
)

toc()

stopCluster(cores)
registerDoSEQ()


select_en <- select_best(tune_en, metric = "rmse")

workflow_en_best <- finalize_workflow(wflw_en, select_en)

wflw_en_fit <- workflow_en_best %>%
  fit(data = training(splits))

predict_en_train <- predict(wflw_en_fit, new_data = training(splits)) %>% 
  bind_cols(training(splits)) %>% 
  metrics(truth = quantity, estimate = .pred)




## Arima Model

# arima

arima_spec <- arima_reg() %>%
  set_engine("auto_arima")

# Create a workflow
wflw_arima_fit <- workflow() %>%
  add_recipe(recipe_arima) %>%
  add_model(arima_spec)%>%
  fit(training(splits))



## Random Forest

Model 1 for base recipe and model 2 for XGboost/RF recipe (additional date features included)
Both models were run in parallel (around 490 secs to run with 7 cores). See after this snippet the final specifications (tuning) for the model.

# rf

spec_rf <- rand_forest(mtry = tune(), 
                       trees = tune(), 
                       min_n = tune()) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("regression")

workflow_rf1 <- workflow() %>%
  add_recipe(recipe_base2) %>%
  add_model(spec_rf)

workflow_rf2 <- workflow() %>%
  add_recipe(recipe_xgboost) %>%
  add_model(spec_rf)

#rolling_origin - validation respects the temporal order of data. The model is trained only on past data to predict the future.

cv_splits <- rolling_origin(training(splits), initial = 50, assess = 10, skip = 5, cumulative = FALSE)

grid_rf1 <- grid_regular(
  mtry(range = c(1,5)),
  trees(),
  min_n(),
  levels = 5
)

grid_rf2 <- grid_rf1

cores <- makePSOCKcluster(detectCores()-1)
doParallel::registerDoParallel(cores)

tic()

tuned_rf1 <- tune_grid(
  workflow_rf1,
  resamples = cv_splits,
  grid = grid_rf1,
)

tuned_rf2 <- tune_grid(
  workflow_rf2,
  resamples = cv_splits,
  grid = grid_rf2,
)

toc()

stopCluster(cores)
registerDoSEQ()

## Model 1

select_params1 <- select_best(tuned_rf1, metric = "rmse")

workflow_rf_best1 <- finalize_workflow(workflow_rf1, select_params1)

wflw_rf_fit1 <- workflow_rf_best1 %>%
  fit(data = training(splits))

## Model 2

select_params2 <- select_best(tuned_rf2, metric ="rmse")

workflow_rf_best2 <- finalize_workflow(workflow_rf2, select_params2)

wflw_rf_fit2 <- workflow_rf_best2 %>%
  fit(data = training(splits))



## Random Forest tuned

# rf_tuned

# Base model

spec_rf1 <- rand_forest(mtry = 5, min_n = 2, trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

workflow_rf1 <- workflow() %>%
  add_recipe(recipe_base2) %>%
  add_model(spec_rf1)

wflw_rf_fit1 <- workflow_rf1 %>%
  fit(data = training(splits))

# XGBoost/RF recipe

spec_rf2 <- rand_forest(mtry = 5, min_n = 2, trees = 1500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

workflow_rf2 <- workflow() %>%
  add_recipe(recipe_xgboost) %>%
  add_model(spec_rf2)

wflw_rf_fit2 <- workflow_rf2 %>%
  fit(data = training(splits))




## XGBoost models

Model 1 represents the base recipe, and Model 2 represents the XGBoost recipe (with additional date features included). Both models were run in parallel (around 560 secs to run with 7 cores). See after this snippet the final specifications (tuning) for the model.

# xgb

spec_xgboost <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) %>%
  set_mode("regression") %>%
  set_engine("xgboost")


workflow_xgboost1 <- workflow() %>%
  add_recipe(recipe_base2) %>%
  add_model(spec_xgboost)

workflow_xgboost2 <- workflow() %>%
  add_recipe(recipe_xgboost) %>%
  add_model(spec_xgboost)

#rolling_origin - validation respects the temporal order of data. The model is trained only on past data to predict the future.

cv_splits <- rolling_origin(training(splits), initial = 50, assess = 10, skip = 5, cumulative = FALSE)

grid_xgboost1 <- grid_latin_hypercube(
  trees(),
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), training(splits)),
  learn_rate(),
  size = 30
)

grid_xgboost2 <- grid_xgboost1

cores <- makePSOCKcluster(detectCores()-1)
doParallel::registerDoParallel(cores)

tic()

# Tune the model with cross-validation
set.seed(234)

tuned_results1 <- tune_grid(
  workflow_xgboost1,
  resamples = cv_splits,
  grid = grid_xgboost1,
  control = control_grid(save_pred = TRUE, parallel_over = "everything")
)

toc()

stopCluster(cores)
registerDoSEQ()

cores <- makePSOCKcluster(detectCores()-1)
doParallel::registerDoParallel(cores)

tic()

# Tune the model with cross-validation
set.seed(234)

tuned_results2 <- tune_grid(
  workflow_xgboost2,
  resamples = cv_splits,
  grid = grid_xgboost2,
  control = control_grid(save_pred = TRUE, parallel_over = "everything")
)

toc()

stopCluster(cores)
registerDoSEQ()

### Model 1

select_params1 <- tune::select_best(tuned_results1, metric = "rmse")

workflow_xgboost_best1 <- finalize_workflow(workflow_xgboost1, select_params1)


wflw_xgboost_fit1 <- workflow_xgboost_best1 %>%
  fit(data = training(splits))

### Model 2

select_params2 <- select_best(tuned_results2, metric = "rmse")

workflow_xgboost_best2 <- finalize_workflow(workflow_xgboost2, select_params2)


wflw_xgboost_fit2 <- workflow_xgboost_best2 %>%
  fit(data = training(splits))



## XGBoost tuned

# xgb_tune1
set.seed(456)

spec_xgboost <- boost_tree(
  mode = "regression",
  mtry = 15,
  trees = 217,
  min_n = 17,
  tree_depth = 7,
  learn_rate = 0.00786773244058574,
  loss_reduction = 8.04243507128875,
  sample_size = 0.790286398630124
  
) %>%
  set_engine("xgboost")

workflow_xgboost1 <- workflow() %>%
  add_recipe(recipe_base2) %>%
  add_model(spec_xgboost)

wflw_xgboost_fit1 <- workflow_xgboost1 %>%
  fit(data = training(splits))

workflow_xgboost2 <- workflow() %>%
  add_recipe(recipe_xgboost) %>%
  add_model(spec_xgboost)

wflw_xgboost_fit2 <- workflow_xgboost2 %>%
  fit(data = training(splits))




## XGBoost manually tuned
# xgb_tune2

spec_xgboost3 <- boost_tree(
  mode = "regression",
  mtry = 3,
  trees = 1600,
  min_n = 26,
  tree_depth = 2,
  learn_rate = 0.0303,
  loss_reduction = 0.0000110,
  sample_size = 0.988
  
) %>%
  set_engine("xgboost")

workflow_xgboost3 <- workflow() %>%
  add_recipe(recipe_xgboost) %>%
  add_model(spec_xgboost3)

wflw_xgboost_fit3 <- workflow_xgboost3 %>%
  fit(data = training(splits))



# Models evaluation

## Metrics table

Models will be assessed based on metrics RMSE, MAE, etc..
(Please note that the model results may slightly change when tuned)

# assessment

models_tbl <- modeltime_table(
  wflw_lm_fit1,
  wflw_lm_fit2,
  wflw_en_fit,
  wflw_arima_fit,
  wflw_rf_fit1, 
  wflw_rf_fit2,
  wflw_xgboost_fit1,
  wflw_xgboost_fit2,
  wflw_xgboost_fit3)


calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits), quiet = FALSE) 

calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = TRUE)



## Forecast plot in the test set
(Please note that the model results may slightly change when tuned)


# forecast plot

total_forec <- bind_rows(training(splits), testing(splits))

plot_total <- calibration_tbl %>%
  modeltime_forecast(new_data    = testing(splits), 
                     actual_data = df_data) %>%
  plot_modeltime_forecast()

plot_fore <- calibration_tbl %>%
  modeltime_forecast(new_data    = testing(splits), 
                     actual_data = df_data) %>%
  filter(.model_desc == "ACTUAL" | .model_id %in% c(1,9)) %>%
  filter(.index >= "2022-01-02") %>% 
  plot_modeltime_forecast(.interactive = FALSE, .title = "LM and XGBoost Forecast - 2022")

plot_total


## Feature importance
(Please note that the model results may slightly change when tuned)

# feature_importance

p1 <- wflw_xgboost_fit2 %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 14)

p2 <- wflw_xgboost_fit3 %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 14)

## XGboost - best model plot
p2


