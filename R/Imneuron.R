#' Fitting of AI based Neural Network Model
#'
#' @param data dataset containing the information about all the variables which are continuous in nature
#' @param target_variable response variable
#' @param hidden_neurons_range This is a range of values specifying the number of hidden neurons to explore in the neural network's two layers (Layer 1 and Layer 2)
#' @param cv_type This argument is used to apply cross validation like "5_fold" for 5 folded cross validation and "10-fold" for 10 folded cross validation
#'
#' @return Average values of R2, RMSE, MAE, and PER across the cross-validation folds.    The trained neural network models for each fold.    A data frame containing the evaluation metrics for each fold
#'
#' @export
#' @import neuralnet ggplot2 MLmetrics
#' @examples
#' # 5-fold cross-validation
#' \donttest{
#' data(fruit)
#' results_5fold <- Imneuron(fruit, "Fruit.Yield", hidden_neurons_range = c(2,2), cv_type = "5-fold")
# 10-fold cross-validation
#' results_10fold <- Imneuron(fruit, "Fruit.Yield", hidden_neurons_range = c(2,2), cv_type = "10-fold")
#' }
#' @references Jeelani, M.I., Tabassum, A., Rather, K and Gul, M. (2023). Neural Network Modeling of Height Diameter Relationships for Himalayan Pine through Back Propagation Approach. Journal of The Indian Society of Agricultural Statistics. 76(3): 169.
#' Tabassum, A., Jeelani, M.I., Sharma,M., Rather, K R ., Rashid, I and Gul, M. (2022). Predictive Modelling of Height and Diameter Relationships of Himalayan Chir Pine. Agricultural Science Digest - A Research Journal. <doi:10.18805/ag.D-5555>
#'

 Imneuron<- function(data, target_variable, hidden_neurons_range, cv_type = "5-fold") {
  # Load required libraries
  requireNamespace("neuralnet")
  requireNamespace("MLmetrics")
  requireNamespace("ggplot2")

  # Data normalization
  normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
  }

  maxmindf <- as.data.frame(lapply(data, normalize))

  results <- list()

  # Define number of folds based on cv_type
  if (cv_type == "5-fold") {
    num_folds <- 5
  } else if (cv_type == "10-fold") {
    num_folds <- 10
  } else {
    stop("Invalid cv_type. Supported values are '5-fold' or '10-fold'")
  }

  # Split data into folds
  folds <- cut(seq(1, nrow(maxmindf)), breaks = num_folds, labels = FALSE)

  for (n1 in hidden_neurons_range) {
    for (n2 in hidden_neurons_range) {
      # Initialize data frames to store evaluation metrics
      results_df <- data.frame(
        Fold = 1:num_folds,
        R2 = rep(NA, num_folds),
        RMSE = rep(NA, num_folds),
        MAE = rep(NA, num_folds),
        PER = rep(NA, num_folds)
      )

      # Initialize lists to store neural network models and plots
      nn_models <- list()
      nn_plots <- list()

      # Initialize variables to store plot with lowest error
      min_error <- Inf
      min_error_index <- NULL

      # Perform cross-validation
      for (i in 1:num_folds) {
        # Split data into train and validation sets
        validation_indices <- which(folds == i, arr.ind = TRUE)
        validation_data <- maxmindf[validation_indices, ]
        train_data <- maxmindf[-validation_indices, ]

        # Train neural network model
        formula <- stats::as.formula(paste(target_variable, "~."))
        nn <-neuralnet::neuralnet(formula, data = train_data,
                        hidden = c(n1, n2),
                        rep = 2,
                        act.fct = "logistic",
                        err.fct = "sse",
                        linear.output = FALSE,
                        lifesign = "minimal",
                        stepmax = 1000000,
                        threshold = 0.001)

        # Store neural network model
        nn_models[[i]] <- nn

        # Predict on validation set
        predictions <- stats::predict(nn, validation_data)

        # Evaluate model performance
        error <-MLmetrics::RMSE(predictions, validation_data[[target_variable]])

        # Update minimum error and index
        if (error < min_error) {
          min_error <- error
          min_error_index <- i
        }

        # Evaluate model performance
        results_df[i, "R2"] <-MLmetrics::R2_Score(predictions, validation_data[[target_variable]])
        results_df[i, "RMSE"] <- error
        results_df[i, "MAE"] <-MLmetrics::MAE(predictions, validation_data[[target_variable]])
        results_df[i, "PER"] <-MLmetrics::RMSE(predictions, validation_data[[target_variable]] / mean(validation_data[[target_variable]]))
      }

      # Plot neural network with lowest error
      plot_name <- paste("NN_Plot_Hidden_", n1, "_", n2, "_Fold_", min_error_index, ".png", sep="")
      grDevices::png(plot_name)
      plot(nn_models[[min_error_index]])
      grDevices::dev.off()

      # Store plot file name
      nn_plots[[1]] <- plot_name

      # Compute average evaluation metrics across folds
      avg_R2 <- mean(results_df$R2)
      avg_RMSE <- mean(results_df$RMSE)
      avg_MAE <- mean(results_df$MAE)
      avg_PER <- mean(results_df$PER)

      # Store results for this combination of hidden neurons
      evaluation <- list(
        Hidden_Neurons_Layer1 = n1,
        Hidden_Neurons_Layer2 = n2,
        R2 = avg_R2,
        RMSE = avg_RMSE,
        MAE = avg_MAE,
        PER = avg_PER,
        NN_Models = nn_models,
        NN_Plots = nn_plots,
        Results_DF = results_df
      )

      results <- c(results, list(evaluation))
    }
  }

  return(results)
}
