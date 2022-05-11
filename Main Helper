splitter <- function(x, var, folds) {
  to_rm <- c(dplyr::setdiff(outcomes, var))
  
  x <- x %>%
    ## Create separate DF with keys and remove them from data
    mutate(keys = map(data,  ~ select(.x, any_of(ids)) %>% 
                        mutate(.row = row_number()))) %>%
    mutate(data = map(data, ~ select(.x, -any_of(ids)) %>%
                        remove_constant())) %>%
    ## Filter
    filter(map_dbl(data, nrow) > 100) %>%
    ## remove outcomes not in var
    mutate(data = map(data, ~ select(.x,-any_of(to_rm)))) %>%
    ## Initial Split
    mutate(data_split = map(data, initial_split, strata = all_of(var))) %>%
    ## Training Split
    mutate(data_train = map(data_split, training)) %>%
    ## Testing Split
    mutate(data_test = map(data_split, testing)) %>%
    ## CV Folds
    mutate(cv_folds = map(data_train, vfold_cv, strata = var, v = folds))
  
  return(x)
}

folds <- 2

train_tune_predict <-
  function(x,
           model,
           desired_metric = 'rmse',
           mode = 'regression') {
    
    ## Loop
    
    output <- tibble()
    
    count <- 0
    
    var <- sym(var)
    
    for (i in rep(1:nrow(x))) {
      
      count <- count + 1
      
      this <- x[i,]
      
      ## Globals
      var <- var
      name <- this$name
      control <- control_resamples(save_pred = TRUE, verbose = TRUE)
      mode <- mode
      rec <- this$rec[[1]]
      cv_folds <- this$cv_folds[[1]]
      data_split <- this$data_split[[1]]
      data_train <- this$data_train[[1]]
      data_test <- this$data_test[[1]]
      keys <- this$keys[[1]]
      ## Messages
      message_spec <- glue('Spec done n = {count}')
      message_grid <- glue('Grid done n = {count}')
      message_wf <- glue('WF done n = {count}')
      message_tune <- glue('Tune done n = {count}')
      message_best_tune <- glue('Best Tune done n = {count}')
      message_final_wf <- glue('Final WF done n = {count}')
      message_last_fit <- glue('Last Fit done n = {count}')
      message_preds <- glue('Preds done n = {count}')
      message_metrics <- glue('Metrics done n = {count}')
      
      message_loop <- glue('----------------------------------Loop done n = {count}')
      
      message_function <- glue('----------------------------------Function done')
      
      ## If Statement controlling the logic
      
      ## Cubist Rules
      if (model == 'cubist_rules') {
        ## Specifies Model
        spec <-
          cubist_rules(committees = tune()) %>%
          set_engine("Cubist")
        print(message_spec)
        ## Grid
        grid <- grid_regular(committees())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(rec) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(
            wf,
            resamples = cv_folds,
            control = control,
            grid = grid
          )
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = data_split,
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
      }
      
      ## Bagged Tree
      
      else if (model == "bag_tree") {
        ## Specifies Model
        spec <-
          bag_tree(
            tree_depth = tune(),
            cost_complexity = tune(),
            min_n = tune()
          ) %>%
          set_mode('regression') %>%
          set_engine('rpart')
        print(message_spec)
        ## Grid
        grid <-
          grid_regular(cost_complexity(), tree_depth(), min_n(), levels = 3)
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(this$rec[[1]]) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(
            wf,
            resamples = this$cv_folds[[1]],
            control = control,
            grid = grid
          )
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = this$data_split[[1]],
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
      }
      
      ## MLP
      
      else if (model == 'mlp') {
        ## Specifies Model
        spec <-
          mlp(hidden_units = tune(),
              epochs = tune(),
              penalty = tune()) %>%
          set_mode('regression') %>%
          set_engine('nnet',
                     MaxNWts = 5000)
        print(message_spec)
        ## Grid
        grid <-
          grid_regular(hidden_units(), epochs(), penalty())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(rec) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(wf,
                    resamples = cv_folds,
                    control = control,
                    grid = grid)
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = this$data_split[[1]],
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
        
        ## Boost Tree (XGboost)
        
      } else if (model == 'boost_tree'){
        
        ## Specifies Model
        spec <-
          boost_tree(trees = 1000,
                     tree_depth = tune(),
                     min_n = tune()) %>%
          set_engine('xgboost') %>%
          set_mode(mode)
        print(message_spec)
        ## Grid
        grid <-
          grid_regular(tree_depth(),
                       min_n())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(this$rec[[1]]) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(wf,
                    resamples = this$cv_folds[[1]])
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = this$data_split[[1]],
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
        
      }
      
      ## Bart
      
      else if (model == 'bart') {
        ## Specifies Model
        spec <-
          parsnip::bart(trees = tune()) %>%
          set_engine("dbarts") %>%
          set_mode(mode)
        print(message_spec)
        ## Grid
        grid <- grid_regular(trees())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(rec) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(
            wf,
            resamples = cv_folds,
            control = control,
            grid = grid
          )
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = data_split,
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
      }
      
      ## Ranger
      
      else if (model == 'ranger') {
        ## Specifies Model
        spec <-
          rand_forest(
            min_n = tune(),
            trees = tune()) %>%
          set_mode(mode) %>%
          set_engine("ranger")
        print(message_spec)
        ## Grid
        grid <- grid_regular(min_n(),trees())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(rec) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(
            wf,
            resamples = cv_folds,
            control = control,
            grid = grid
          )
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = data_split,
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
      }
      
      ## GLM
      
      else if (model == 'glm') {
        ## Specifies Model
        spec <-
          linear_reg(penalty = tune()) %>%
          set_mode(mode) %>%
          set_engine("glmnet")
        print(message_spec)
        ## Grid
        grid <- grid_regular(penalty())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(rec) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(
            wf,
            resamples = cv_folds,
            control = control,
            grid = grid
          )
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = data_split,
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
      }
      
      ## GLM
      
      else if (model == 'mars') {
        ## Specifies Model
        spec <-
          mars(
            prod_degree = tune(),
            prune_method = "none"
          ) %>%
          set_mode(mode) %>%
          set_engine("earth") 
        print(message_spec)
        ## Grid
        grid <- grid_regular(prod_degree())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(rec) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(
            wf,
            resamples = cv_folds,
            control = control,
            grid = grid
          )
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = data_split,
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
      }
      
      ## Bag Mars
      
      else if (model == 'bag_mars') {
        ## Specifies Model
        spec <-
          bag_mars(
            prod_degree = tune()
          ) %>%
          set_mode(mode) %>%
          set_engine("earth") 
        print(message_spec)
        ## Grid
        grid <- grid_regular(prod_degree())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(rec) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(
            wf,
            resamples = cv_folds,
            control = control,
            grid = grid
          )
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = data_split,
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
      }
      
      ## SVM Poly
      
      else if (model == 'svm_poly') {
        ## Specifies Model
        spec <-
          svm_poly(cost = tune(),
                   degree = tune(),
                   scale_factor = tune()) %>%
          set_mode(mode)
        print(message_spec)
        ## Grid
        grid <- grid_regular(cost(),degree(),scale_factor())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(rec) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(
            wf,
            resamples = cv_folds,
            control = control,
            grid = grid
          )
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = data_split,
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
      }
      
      ## SVM Linear
      
      else if (model == 'svm_linear') {
        ## Specifies Model
        spec <-
          svm_linear(cost = tune()) %>%
          set_mode(mode) %>%
          set_engine('LiblineaR')
        print(message_spec)
        ## Grid
        grid <- grid_regular(cost())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(rec) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(
            wf,
            resamples = cv_folds,
            control = control,
            grid = grid
          )
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = data_split,
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
      }
      
      #KNN
      
      else if (model == 'knn') {
        ## Specifies Model
        spec <-
          nearest_neighbor(neighbors = tune(), weight_func = tune()) %>%
          set_mode(mode) %>%
          set_engine("kknn")
        print(message_spec)
        ## Grid
        grid <- grid_regular(neighbors(),weight_func())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(rec) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(
            wf,
            resamples = cv_folds,
            control = control,
            grid = grid
          )
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = data_split,
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
      }
      
      #SVM rbj
      
      else if (model == 'svm_rbj') {
        ## Specifies Model
        spec <-
          svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
          set_mode(mode) 
        print(message_spec)
        ## Grid
        grid <- grid_regular(cost(),rbf_sigma())
        print(message_grid)
        ## Workflow
        wf <-
          workflow() %>%
          add_recipe(rec) %>%
          add_model(spec)
        print(message_wf)
        ## Tune
        tune <-
          tune_grid(
            wf,
            resamples = cv_folds,
            control = control,
            grid = grid
          )
        print(message_tune)
        best <- select_best(tune, metric = desired_metric)
        print(message_best_tune)
        ## Finalize WF
        final_wf  <-
          finalize_workflow(wf, best)
        print(message_final_wf)
        ## Last Fit
        last_fit <- final_wf %>%
          last_fit(split = data_split,
                   metrics = metric_set(rmse, rsq))
        print(message_last_fit)
        ## Predictions
        preds <- collect_predictions(last_fit)
        print(message_preds)
        ## Metrics
        metrics <- collect_metrics(last_fit)
        print(message_metrics)
      }
      
      ## Appending DF
      
      preds <- preds %>%
        left_join(keys, by = '.row')
      
      preds <- preds %>%
        mutate(name = name) %>%
        mutate(difference = abs(.data[[var]] - .pred)) %>%
        mutate(in_range = difference < 66) %>%
        arrange(difference) %>%
        select(in_range, .pred, .row, var, name, difference,key,name_stripped) %>%
        relocate(in_range, difference, var, .pred, name, .row) %>%
        mutate(rmse = metrics %>% filter(.metric == 'rmse') %>% select(.estimate)  %>% pull()) %>%
        mutate(rsq = metrics %>% filter(.metric == 'rsq') %>% select(.estimate) %>% pull()) %>%
        group_by(name) %>%
        mutate(hit_rate = sum(in_range, na.rm=T) / nrow(.)) %>%
        ungroup()
      
      output <- output %>%
        bind_rows(preds)
      
      print(message_loop)
      
    }
    
    print(message_function)
    
    return(output)
    
  }
all_models <-
  c(
    'cubist_rules',
    'bag_tree',
    'mlp',
    'boost_tree',
    'bart',
    'ranger',
    'glm',
    'mars',
    'bag_mars',
    'svm_poly',
    'svm_linear',
    'knn',
    'svm_rbj'
  )

models_to_test <- c('glm')

nested_results <- tibble()

for(i in models_to_test) {
  
  set.seed(1)
  
  start.time <- Sys.time()
  
  print(glue("----------------------------------Starting {i}"))
  
  new <- advanced %>%
    train_tune_predict(i)
  
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  
  new <- new %>%
    mutate(model = i) %>%
    mutate(time = time.taken)
  
  nested_results <- nested_results %>%
    bind_rows(new)
}
