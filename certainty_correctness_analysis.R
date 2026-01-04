library(tidyverse)

# Define models with ensemble+MTP results
models <- c(
  "deepseekv31671b" = "DeepSeek V3.1",
  "gptoss120b" = "GPT-OSS 120B",
  "llama33b" = "Llama 3.2 Instruct",
  "qwen3" = "Qwen 3 Instruct"
)

features <- read_csv("data/features_all.csv", show_col_types = FALSE)
excluded_features <- c("ParticipantPeakDysphoria", "IndividualExegesis", "DissolveUnion", "Disgust")
feature_vars <- features$feature_variable[!(features$feature_variable %in% excluded_features)]
exclude <- read_csv("data/exclude.csv", show_col_types = FALSE)

# All (Morphospace dataset)

cat("=== Processing 'all' dataset ===\n\n")

all_results <- list()

for (model_id in names(models)) {
  model_name <- models[[model_id]]
  file_path <- paste0("all/results_", model_id, "_ensemble_mtp.csv")
  
  if (!file.exists(file_path)) {
    cat("  Skipping", model_name, "- file not found\n")
    next
  }
  
  cat("Processing:", model_name, "\n")
  
  df <- read_csv(file_path, show_col_types = FALSE)
  
  # Filter exclusions
  df <- df %>% filter(!(ritual_number %in% exclude$exclude))
  
  # Find certainty columns
  certainty_cols <- grep("_llm_certainty$", names(df), value = TRUE)
  
  # For each feature, extract correctness and certainty
  for (feat in feature_vars) {
    true_col <- feat
    pred_col <- paste0(feat, "_llm")
    cert_col <- paste0(feat, "_llm_certainty")
    
    if (!(true_col %in% names(df)) || !(pred_col %in% names(df)) || !(cert_col %in% names(df))) {
      next
    }
    
    feat_data <- df %>%
      select(ritual_number, all_of(c(true_col, pred_col, cert_col))) %>%
      rename(true_val = !!true_col, pred_val = !!pred_col, certainty = !!cert_col) %>%
      filter(!is.na(true_val), !is.na(pred_val), !is.na(certainty)) %>%
      filter(true_val != -1, true_val != 999, as.character(pred_val) != "") %>%
      mutate(
        true_val = as.numeric(true_val),
        pred_val = as.numeric(pred_val),
        certainty = as.numeric(certainty),
        correct = as.integer(true_val == pred_val),
        model = model_name,
        feature = feat,
        dataset = "all"
      )
    
    if (nrow(feat_data) > 0) {
      all_results[[length(all_results) + 1]] <- feat_data
    }
  }
}


# Synchrony dataset

sync_features <- c("singing", "chanting", "praying", "marching", "dancing", "generic_mvmt")

for (model_id in names(models)) {
  model_name <- models[[model_id]]
  file_path <- paste0("synchrony/results_", model_id, "_ensemble_mtp.csv")
  
  if (!file.exists(file_path)) {
    cat("  Skipping", model_name, "- file not found\n")
    next
  }
  
  cat("Processing:", model_name, "\n")
  
  df <- read_csv(file_path, show_col_types = FALSE)
  
  # Filter exclusions
  df <- df %>% filter(!(ritual_number %in% exclude$exclude))
  
  for (feat in sync_features) {
    true_col <- paste0(feat, "_human")
    pred_col <- paste0(feat, "_llm")
    cert_col <- paste0(feat, "_llm_certainty")
    
    if (!(true_col %in% names(df)) || !(pred_col %in% names(df)) || !(cert_col %in% names(df))) {
      next
    }
    
    feat_data <- df %>%
      select(ritual_number, all_of(c(true_col, pred_col, cert_col))) %>%
      rename(true_val = !!true_col, pred_val = !!pred_col, certainty = !!cert_col) %>%
      filter(!is.na(true_val), !is.na(pred_val), !is.na(certainty)) %>%
      filter(true_val != -1, true_val != 999, as.character(pred_val) != "") %>%
      mutate(
        true_val = as.numeric(true_val),
        pred_val = as.numeric(pred_val),
        certainty = as.numeric(certainty),
        correct = as.integer(true_val == pred_val),
        model = model_name,
        feature = paste0("sync_", feat),
        dataset = "synchrony"
      )
    
    if (nrow(feat_data) > 0) {
      all_results[[length(all_results) + 1]] <- feat_data
    }
  }
}


combined <- bind_rows(all_results)

cat("Total predictions:", nrow(combined), "\n")
cat("Unique models:", n_distinct(combined$model), "\n")
cat("Unique features:", n_distinct(combined$feature), "\n\n")

# Overall correlation
cat("=== Overall Correlation: Certainty vs Correctness ===\n")
overall_cor <- cor.test(combined$certainty, combined$correct, method = "pearson")
cat(sprintf("Pearson r = %.3f, 95%% CI [%.3f, %.3f], p < %.4f\n", 
            overall_cor$estimate, 
            overall_cor$conf.int[1], 
            overall_cor$conf.int[2],
            overall_cor$p.value))

# Point-biserial correlation (same as Pearson for binary outcome)
cat(sprintf("\nN = %d predictions\n", nrow(combined)))

# Correlation by model
cat("\n=== Correlation by Model ===\n")
model_cors <- combined %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean_certainty = mean(certainty, na.rm = TRUE),
    accuracy = mean(correct, na.rm = TRUE),
    r = cor(certainty, correct, use = "complete.obs"),
    .groups = "drop"
  ) %>%
  arrange(desc(r))

print(model_cors)

# Correlation by dataset
cat("\n=== Correlation by Dataset ===\n")
dataset_cors <- combined %>%
  group_by(dataset) %>%
  summarise(
    n = n(),
    mean_certainty = mean(certainty, na.rm = TRUE),
    accuracy = mean(correct, na.rm = TRUE),
    r = cor(certainty, correct, use = "complete.obs"),
    .groups = "drop"
  )

print(dataset_cors)

# Accuracy by certainty bins
cat("\n=== Accuracy by Certainty Bins ===\n")
certainty_bins <- combined %>%
  mutate(cert_bin = cut(certainty, 
                        breaks = c(0, 50, 60, 70, 80, 90, 100),
                        labels = c("10-50%", "51-60%", "61-70%", "71-80%", "81-90%", "91-100%"),
                        include.lowest = TRUE)) %>%
  group_by(cert_bin) %>%
  summarise(
    n = n(),
    pct = n() / nrow(combined) * 100,
    accuracy = mean(correct, na.rm = TRUE) * 100,
    .groups = "drop"
  ) %>%
  arrange(cert_bin)

print(certainty_bins)

# Logistic regression: certainty predicting correctness
cat("\n=== Logistic Regression: Certainty -> Correctness ===\n")
logit_model <- glm(correct ~ certainty, data = combined, family = binomial)
summary_logit <- summary(logit_model)
cat(sprintf("Coefficient (log-odds): %.4f\n", coef(logit_model)["certainty"]))
cat(sprintf("Odds ratio per 10%% increase: %.3f\n", exp(coef(logit_model)["certainty"] * 10)))
cat(sprintf("p-value: %.2e\n", summary_logit$coefficients["certainty", "Pr(>|z|)"]))

# Create output directory
dir.create("figures_R/certainty_analysis", showWarnings = FALSE, recursive = TRUE)

write_csv(combined, "figures_R/certainty_analysis/prediction_level_data.csv")
write_csv(model_cors, "figures_R/certainty_analysis/correlation_by_model.csv")
write_csv(certainty_bins, "figures_R/certainty_analysis/accuracy_by_certainty_bin.csv")

overall_summary <- tibble(
  metric = c("overall_r", "overall_r_lower", "overall_r_upper", "overall_p", 
             "n_predictions", "mean_certainty", "mean_accuracy",
             "odds_ratio_per_10pct", "logit_p"),
  value = c(overall_cor$estimate, overall_cor$conf.int[1], overall_cor$conf.int[2], 
            overall_cor$p.value, nrow(combined), 
            mean(combined$certainty), mean(combined$correct),
            exp(coef(logit_model)["certainty"] * 10),
            summary_logit$coefficients["certainty", "Pr(>|z|)"])
)

write_csv(overall_summary, "figures_R/certainty_analysis/overall_summary.csv")

cat("Results saved to figures_R/certainty_analysis")

# Accuracy by certainty bin plot
p1 <- certainty_bins %>%
  ggplot(aes(x = cert_bin, y = accuracy)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  geom_text(aes(label = sprintf("%.1f%%\n(n=%d)", accuracy, n)), 
            vjust = -0.3, size = 3) +
  labs(
    title = "Accuracy Increases with Model Certainty",
    subtitle = "Ensemble+MTP condition (10 repetitions)",
    x = "Certainty (% agreement across repetitions)",
    y = "Accuracy (%)"
  ) +
  ylim(0, 100) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave("figures_R/certainty_analysis/accuracy_by_certainty.pdf", p1, width = 8, height = 6)

cat("Visualisation saved at figures_R/certainty_analysis/accuracy_by_certainty.pdf.\n")
