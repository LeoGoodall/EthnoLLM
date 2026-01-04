library(tidyverse)
library(irr)
library(ggplot2)
library(gridExtra)
library(patchwork)

output_dir <- "llm_human_reliability"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Load data
rituals_data <- read_csv("data/rituals_codes.csv", show_col_types = FALSE)

# Define features
features <- c("singing", "chanting", "praying", "marching", "dancing", "generic_mvmt")
feature_labels <- c(
  "Synchronous Singing",
  "Synchronous Chanting",
  "Synchronous Praying",
  "Synchronous Marching",
  "Synchronous Dancing",
  "Synchronous Generic Movement"
)

# Convert coder columns to numeric
for (feat in features) {
  rituals_data[[paste0(feat, "_coder_1")]] <- as.numeric(rituals_data[[paste0(feat, "_coder_1")]])
  rituals_data[[paste0(feat, "_coder_2")]] <- as.numeric(rituals_data[[paste0(feat, "_coder_2")]])
}

# Load LLM results
llm_models <- c(
  "claudesonnet45" = "Claude Sonnet 4.5",
  "deepseekv31671b" = "DeepSeek V3.1",
  "gptoss120b" = "GPT-OSS 120B",
  "gpt5nano" = "GPT-5 Nano",
  "llama33b" = "Llama 3.2 Instruct (3B)",
  "qwen3" = "Qwen 3 Instruct (4B)",
  "perplexity" = "Perplexity Sonar"
)

# Load LLM predictions (baseline condition)
llm_data_list <- map(names(llm_models), function(model) {
  file_path <- file.path("synchrony", paste0("results_", model, ".csv"))
  if (file.exists(file_path)) {
    df <- read_csv(file_path, show_col_types = FALSE) %>%
      mutate(model = model)
    # Convert all _llm columns to numeric to handle type mismatches across files
    llm_cols <- grep("_llm$", names(df), value = TRUE)
    for (col in llm_cols) {
      df[[col]] <- as.numeric(df[[col]])
    }
    df
  } else {
    NULL
  }
})
llm_data <- bind_rows(llm_data_list)

# Merge with rituals data
full_data <- rituals_data %>%
  select(ritual_number, all_of(paste0(features, "_human")),
         all_of(paste0(features, "_coder_1")),
         all_of(paste0(features, "_coder_2"))) %>%
  inner_join(llm_data %>% 
               select(ritual_number, model, 
                      all_of(paste0(features, "_llm"))),
             by = "ritual_number")


# 1. HUMAN INTER-CODER RELIABILITY AS CEILING

# Compute human-human agreement (Cohen's kappa)
human_reliability <- map_dfr(features, function(feat) {
  c1 <- rituals_data[[paste0(feat, "_coder_1")]]
  c2 <- rituals_data[[paste0(feat, "_coder_2")]]
  
  complete <- !is.na(c1) & !is.na(c2)
  kappa_result <- kappa2(data.frame(c1[complete], c2[complete]))
  
  tibble(
    feature = feat,
    human_kappa = kappa_result$value,
    human_agreement = sum(c1[complete] == c2[complete]) / sum(complete)
  )
})

# Compute LLM-human agreement for each model
compute_f1 <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)
  complete <- !is.na(y_true) & !is.na(y_pred)
  y_true <- y_true[complete]
  y_pred <- y_pred[complete]
  
  tp <- sum(y_true == 1 & y_pred == 1)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  
  if (tp + fp == 0) precision <- 0 else precision <- tp / (tp + fp)
  if (tp + fn == 0) recall <- 0 else recall <- tp / (tp + fn)
  if (precision + recall == 0) f1 <- 0 else f1 <- 2 * precision * recall / (precision + recall)
  
  return(f1)
}

llm_performance <- full_data %>%
  group_by(model) %>%
  group_split() %>%
  map_dfr(function(model_data) {
    model_name <- unique(model_data$model)
    map_dfr(features, function(feat) {
      tibble(
        model = model_name,
        feature = feat,
        llm_f1 = compute_f1(
          model_data[[paste0(feat, "_human")]],
          model_data[[paste0(feat, "_llm")]]
        )
      )
    })
  })

# Combine human reliability and LLM performance
combined_data <- llm_performance %>%
  left_join(human_reliability, by = "feature") %>%
  mutate(
    feature_label = feature_labels[match(feature, features)],
    model_label = llm_models[model]
  )

# Summary statistics
cat("Correlation between human kappa and LLM F1:\n")
correlation_results <- combined_data %>%
  group_by(model_label) %>%
  summarise(
    correlation = cor(human_kappa, llm_f1, use = "complete.obs"),
    .groups = "drop"
  )
print(correlation_results)
cat("\n")

# Overall correlation
overall_cor <- cor(combined_data$human_kappa, combined_data$llm_f1, use = "complete.obs")
cat(sprintf("Overall correlation (all models): %.3f\n\n", overall_cor))

# Save results
write_csv(combined_data, 
          file.path(output_dir, "human_reliability_llm_performance.csv"))

# Plot: Human reliability vs LLM performance
p1 <- ggplot(combined_data, aes(x = human_kappa, y = llm_f1)) +
  geom_smooth(method = "lm", se = TRUE, color = "grey70", linetype = "dashed") +
  geom_point(aes(color = model_label), size = 3, alpha = 0.7) +
  geom_text(aes(label = feature), size = 2.5, vjust = -0.8, hjust = 0.5) +
  labs(
    title = "Human Inter-Coder Reliability as LLM Performance Ceiling",
    subtitle = sprintf("Correlation: r = %.3f", overall_cor),
    x = "Human Inter-Coder Agreement (Cohen's \u03BA)",
    y = "LLM Performance (F1 Score)",
    color = "LLM Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "grey40"),
    legend.position = "bottom"
  ) +
  scale_color_brewer(palette = "Set1")

ggsave(file.path(output_dir, "human_ceiling_analysis.pdf"), 
       p1, width = 10, height = 8, device = cairo_pdf)

# 2. LLM AGREEMENT WITH INDIVIDUAL CODERS

# Compute agreement with each coder separately
llm_individual_agreement <- full_data %>%
  group_by(model) %>%
  group_split() %>%
  map_dfr(function(model_data) {
    model_name <- unique(model_data$model)
    map_dfr(features, function(feat) {
      pred <- model_data[[paste0(feat, "_llm")]]
      true_c1 <- model_data[[paste0(feat, "_coder_1")]]
      true_c2 <- model_data[[paste0(feat, "_coder_2")]]
      true_resolved <- model_data[[paste0(feat, "_human")]]
      
      # Coder 1
      complete_c1 <- !is.na(pred) & !is.na(true_c1)
      kappa_c1 <- if (sum(complete_c1) > 10) {
        kappa2(data.frame(true_c1[complete_c1], pred[complete_c1]))$value
      } else NA_real_
      
      # Coder 2
      complete_c2 <- !is.na(pred) & !is.na(true_c2)
      kappa_c2 <- if (sum(complete_c2) > 10) {
        kappa2(data.frame(true_c2[complete_c2], pred[complete_c2]))$value
      } else NA_real_
      
      # Resolved
      complete_resolved <- !is.na(pred) & !is.na(true_resolved)
      kappa_resolved <- if (sum(complete_resolved) > 10) {
        kappa2(data.frame(true_resolved[complete_resolved], pred[complete_resolved]))$value
      } else NA_real_
      
      tibble(
        model = model_name,
        feature = feat,
        coder1_kappa = kappa_c1,
        coder2_kappa = kappa_c2,
        resolved_kappa = kappa_resolved
      )
    })
  }) %>%
  pivot_longer(cols = ends_with("_kappa"),
               names_to = "agreement_type",
               values_to = "kappa") %>%
  mutate(
    feature_label = feature_labels[match(feature, features)],
    model_label = llm_models[model],
    agreement_type = case_when(
      agreement_type == "coder1_kappa" ~ "vs Coder 1",
      agreement_type == "coder2_kappa" ~ "vs Coder 2",
      agreement_type == "resolved_kappa" ~ "vs Resolved"
    )
  )

# Add human-human agreement for reference
human_human_reference <- human_reliability %>%
  mutate(
    feature_label = feature_labels[match(feature, features)],
    agreement_type = "Human vs Human",
    model_label = "Human Baseline"
  ) %>%
  rename(kappa = human_kappa) %>%
  select(feature, feature_label, kappa, agreement_type, model_label)

# Summary: Are LLMs within human variation range?
cat("LLM agreement with individual coders vs resolved ground truth:\n\n")
agreement_summary <- llm_individual_agreement %>%
  group_by(model_label, feature_label) %>%
  summarise(
    coder1 = kappa[agreement_type == "vs Coder 1"],
    coder2 = kappa[agreement_type == "vs Coder 2"],
    resolved = kappa[agreement_type == "vs Resolved"],
    mean_individual = mean(c(coder1, coder2), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(human_reliability %>% 
              mutate(feature_label = feature_labels[match(feature, features)]) %>%
              select(feature_label, human_kappa),
            by = "feature_label")

print(agreement_summary %>% 
        mutate(across(where(is.numeric), ~round(., 3))),
      n = Inf)

write_csv(agreement_summary, 
          file.path(output_dir, "llm_individual_coder_agreement.csv"))

# Plot: LLM agreement with individual coders
p2 <- ggplot(llm_individual_agreement, 
             aes(x = feature_label, y = kappa, fill = agreement_type)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_point(position = position_jitterdodge(jitter.width = 0.1), 
             alpha = 0.5, size = 2) +
  geom_hline(data = human_reliability %>% 
               mutate(feature_label = feature_labels[match(feature, features)]),
             aes(yintercept = human_kappa),
             linetype = "dashed", color = "red", size = 0.7) +
  facet_wrap(~feature_label, scales = "free_x", nrow = 2) +
  coord_flip() +
  labs(
    title = "LLM Agreement with Individual Coders vs Resolved Labels",
    subtitle = "Red line = Human-human agreement (baseline)",
    x = NULL,
    y = "Cohen's Kappa",
    fill = "Comparison"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "bottom",
    strip.text = element_text(size = 9, face = "bold")
  ) +
  scale_fill_brewer(palette = "Set2")

ggsave(file.path(output_dir, "llm_individual_coder_agreement.pdf"), 
       p2, width = 14, height = 8, device = cairo_pdf)



# 3. ERROR PATTERN COMPARISON: HUMANS VS LLMS

# Function to compute error patterns
compute_error_pattern <- function(coder1, coder2, name1, name2) {
  complete <- !is.na(coder1) & !is.na(coder2)
  c1 <- coder1[complete]
  c2 <- coder2[complete]
  
  tibble(
    name1 = name1,
    name2 = name2,
    false_positive_rate = sum(c1 == 0 & c2 == 1) / sum(c1 == 0),
    false_negative_rate = sum(c1 == 1 & c2 == 0) / sum(c1 == 1),
    both_present = sum(c1 == 1 & c2 == 1),
    both_absent = sum(c1 == 0 & c2 == 0),
    disagree_fp = sum(c1 == 0 & c2 == 1),
    disagree_fn = sum(c1 == 1 & c2 == 0)
  )
}

# Human error patterns
human_error_patterns <- map_dfr(features, function(feat) {
  compute_error_pattern(
    rituals_data[[paste0(feat, "_coder_1")]],
    rituals_data[[paste0(feat, "_coder_2")]],
    "Coder 1", "Coder 2"
  ) %>% mutate(
    feature = feat,
    comparison_type = "Human-Human"
  )
})

# LLM error patterns (vs resolved ground truth)
llm_error_patterns <- full_data %>%
  group_by(model) %>%
  group_split() %>%
  map_dfr(function(model_data) {
    model_name <- unique(model_data$model)
    map_dfr(features, function(feat) {
      compute_error_pattern(
        model_data[[paste0(feat, "_human")]],
        model_data[[paste0(feat, "_llm")]],
        "Ground Truth", model_name
      ) %>% mutate(
        feature = feat,
        comparison_type = "LLM-Human"
      )
    })
  })

# Combine error patterns
all_error_patterns <- bind_rows(
  human_error_patterns %>% mutate(annotator = "Human-Human"),
  llm_error_patterns %>% mutate(annotator = name2)
) %>%
  mutate(
    feature_label = feature_labels[match(feature, features)],
    fp_fn_ratio = ifelse(false_negative_rate > 0, 
                         false_positive_rate / false_negative_rate, 
                         NA_real_)
  )

cat("Error pattern comparison (FP rate vs FN rate):\n\n")
error_summary <- all_error_patterns %>%
  select(annotator, feature_label, false_positive_rate, false_negative_rate, fp_fn_ratio) %>%
  arrange(feature_label, annotator)

print(error_summary %>%
        mutate(across(where(is.numeric), ~round(., 3))),
      n = 30)

write_csv(all_error_patterns, 
          file.path(output_dir, "error_pattern_comparison.csv"))

# Plot: FP vs FN rates
p3 <- ggplot(all_error_patterns, 
             aes(x = false_positive_rate, y = false_negative_rate)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
  geom_point(aes(color = annotator, shape = comparison_type), 
             size = 3, alpha = 0.7) +
  facet_wrap(~feature_label, scales = "free", nrow = 2) +
  labs(
    title = "Error Pattern Comparison: Humans vs LLMs",
    subtitle = "Dashed line = balanced errors",
    x = "False Positive Rate",
    y = "False Negative Rate",
    color = "Annotator",
    shape = "Comparison Type"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "bottom",
    strip.text = element_text(size = 9, face = "bold")
  ) +
  scale_color_brewer(palette = "Set1")

ggsave(file.path(output_dir, "error_pattern_comparison.pdf"), 
       p3, width = 14, height = 8, device = cairo_pdf)


# 4. SUMMARY: COMBINED DIAGNOSTIC PLOT

# Prepare data for combined plot
diagnostic_data <- combined_data %>%
  group_by(feature_label) %>%
  summarise(
    human_kappa = first(human_kappa),
    mean_llm_f1 = mean(llm_f1, na.rm = TRUE),
    min_llm_f1 = min(llm_f1, na.rm = TRUE),
    max_llm_f1 = max(llm_f1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    difficulty_category = factor(
      case_when(
        human_kappa >= 0.8 ~ "Easy (\u03BA ≥ 0.8)",
        human_kappa >= 0.6 ~ "Moderate (0.6 ≤ \u03BA < 0.8)",
        human_kappa >= 0.4 ~ "Difficult (0.4 ≤ \u03BA < 0.6)",
        TRUE ~ "Very Difficult (\u03BA < 0.4)"
      ),
      levels = c("Very Difficult (\u03BA < 0.4)", "Difficult (0.4 ≤ \u03BA < 0.6)", 
                 "Moderate (0.6 ≤ \u03BA < 0.8)", "Easy (\u03BA ≥ 0.8)")
    )
  )

p4a <- ggplot(diagnostic_data, 
              aes(x = reorder(feature_label, human_kappa), y = human_kappa)) +
  geom_col(aes(fill = difficulty_category), alpha = 0.8) +
  geom_text(aes(label = sprintf("%.2f", human_kappa)), 
            hjust = 1.5, size = 4) +
  coord_flip() +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
  labs(
    title = "A. Human Inter-Coder Reliability",
    x = NULL,
    y = "Cohen's Kappa",
    fill = "Difficulty"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    legend.position = "none"
  ) +
  scale_fill_manual(values = c(
    "Easy (\u03BA ≥ 0.8)" = "#27ae60",
    "Moderate (0.6 ≤ \u03BA < 0.8)" = "#f39c12",
    "Difficult (0.4 ≤ \u03BA < 0.6)" = "#e67e22",
    "Very Difficult (\u03BA < 0.4)" = "#e74c3c"
  ))

p4b <- ggplot(diagnostic_data, 
              aes(x = reorder(feature_label, human_kappa), y = mean_llm_f1)) +
  geom_col(aes(fill = difficulty_category), alpha = 0.8) +
  geom_errorbar(aes(ymin = min_llm_f1, ymax = max_llm_f1), 
                width = 0.3, alpha = 0.6) +
  coord_flip() +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
  labs(
    title = "B. LLM Performance (Mean F1 across models)",
    x = NULL,
    y = "F1 Score",
    fill = "Difficulty"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    legend.position = "bottom"
  ) +
  scale_fill_manual(values = c(
    "Easy (\u03BA ≥ 0.8)" = "#27ae60",
    "Moderate (0.6 ≤ \u03BA < 0.8)" = "#f39c12",
    "Difficult (0.4 ≤ \u03BA < 0.6)" = "#e67e22",
    "Very Difficult (\u03BA < 0.4)" = "#e74c3c"
  ))

p4_combined <- p4a + p4b + 
  plot_layout(ncol = 2, widths = c(1, 1.2), guides = "collect") +
  plot_annotation(theme = theme(legend.position = "bottom"))

ggsave(file.path(output_dir, "combined_diagnostic_plot.pdf"), 
       p4_combined, width = 14, height = 6, device = cairo_pdf)

# ============================================================================
# 5. FINAL SUMMARY STATISTICS (Human-LLM)
# ============================================================================

cat("\n=== FINAL SUMMARY (Human-LLM) ===\n\n")

final_summary <- tibble(
  metric = c(
    "Mean human inter-coder kappa",
    "Features with substantial agreement (κ ≥ 0.8)",
    "Features with poor agreement (κ < 0.4)",
    "Correlation: human kappa vs LLM F1",
    "Mean LLM F1 (easy features, κ ≥ 0.8)",
    "Mean LLM F1 (difficult features, κ < 0.4)"
  ),
  value = c(
    mean(human_reliability$human_kappa),
    sum(human_reliability$human_kappa >= 0.8),
    sum(human_reliability$human_kappa < 0.4),
    overall_cor,
    diagnostic_data %>% 
      filter(human_kappa >= 0.8) %>% 
      pull(mean_llm_f1) %>% 
      mean(),
    diagnostic_data %>% 
      filter(human_kappa < 0.4) %>% 
      pull(mean_llm_f1) %>% 
      mean()
  )
)

print(final_summary %>%
        mutate(value = sprintf("%.3f", value)))

write_csv(final_summary, 
          file.path(output_dir, "final_summary.csv"))

# ============================================================================
# 6. LLM-LLM INTER-CODER AGREEMENT ANALYSIS
# ============================================================================

cat("\n=== ANALYSIS 6: LLM-LLM Agreement ===\n\n")

# Load all LLM predictions into a wide format for pairwise comparison
llm_predictions_wide <- map(names(llm_models), function(model) {
  file_path <- file.path("synchrony", paste0("results_", model, ".csv"))
  if (file.exists(file_path)) {
    df <- read_csv(file_path, show_col_types = FALSE)
    # Rename _llm columns to include model name
    for (feat in features) {
      llm_col <- paste0(feat, "_llm")
      if (llm_col %in% names(df)) {
        df[[paste0(feat, "_", model)]] <- as.numeric(df[[llm_col]])
      }
    }
    df %>%
      select(ritual_number, all_of(paste0(features, "_", model)))
  } else {
    NULL
  }
}) %>%
  compact() %>%
  reduce(full_join, by = "ritual_number")

# Function to compute agreement metrics between two coders
compute_pairwise_agreement <- function(coder1_vec, coder2_vec) {
  complete <- !is.na(coder1_vec) & !is.na(coder2_vec)
  c1 <- coder1_vec[complete]
  c2 <- coder2_vec[complete]
  
  if (length(c1) < 10) {
    return(list(kappa = NA_real_, agreement = NA_real_, n = length(c1)))
  }
  
  kappa_result <- tryCatch(
    kappa2(data.frame(c1, c2))$value,
    error = function(e) NA_real_
  )
  agreement <- sum(c1 == c2) / length(c1)
  
  list(kappa = kappa_result, agreement = agreement, n = length(c1))
}

# Compute pairwise LLM-LLM agreement for each feature
model_names <- names(llm_models)
llm_llm_agreement <- map_dfr(features, function(feat) {
  # Get all pairwise combinations
  pairs <- combn(model_names, 2, simplify = FALSE)
  
  map_dfr(pairs, function(pair) {
    model1 <- pair[1]
    model2 <- pair[2]
    
    col1 <- paste0(feat, "_", model1)
    col2 <- paste0(feat, "_", model2)
    
    if (col1 %in% names(llm_predictions_wide) && col2 %in% names(llm_predictions_wide)) {
      result <- compute_pairwise_agreement(
        llm_predictions_wide[[col1]],
        llm_predictions_wide[[col2]]
      )
      
      tibble(
        feature = feat,
        model1 = model1,
        model2 = model2,
        model1_label = llm_models[model1],
        model2_label = llm_models[model2],
        kappa = result$kappa,
        agreement = result$agreement,
        n = result$n
      )
    } else {
      NULL
    }
  })
})

# Add human-human agreement for comparison
llm_llm_agreement <- llm_llm_agreement %>%
  left_join(human_reliability %>% 
              select(feature, human_kappa, human_agreement),
            by = "feature") %>%
  mutate(feature_label = feature_labels[match(feature, features)])

# Summary statistics
cat("LLM-LLM Pairwise Agreement Summary (Mean Kappa by Feature):\n\n")
llm_llm_summary_by_feature <- llm_llm_agreement %>%
  group_by(feature_label) %>%
  summarise(
    mean_llm_llm_kappa = mean(kappa, na.rm = TRUE),
    min_llm_llm_kappa = min(kappa, na.rm = TRUE),
    max_llm_llm_kappa = max(kappa, na.rm = TRUE),
    human_human_kappa = first(human_kappa),
    n_pairs = n(),
    .groups = "drop"
  ) %>%
  mutate(
    llm_vs_human_diff = mean_llm_llm_kappa - human_human_kappa
  )

print(llm_llm_summary_by_feature %>%
        mutate(across(where(is.numeric) & !n_pairs, ~round(., 3))),
      n = Inf)

# Summary by model pair
cat("\n\nMean LLM-LLM Agreement by Model Pair (across all features):\n\n")
llm_llm_summary_by_pair <- llm_llm_agreement %>%
  group_by(model1_label, model2_label) %>%
  summarise(
    mean_kappa = mean(kappa, na.rm = TRUE),
    min_kappa = min(kappa, na.rm = TRUE),
    max_kappa = max(kappa, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_kappa))

print(llm_llm_summary_by_pair %>%
        mutate(across(where(is.numeric), ~round(., 3))),
      n = Inf)

# Save results
write_csv(llm_llm_agreement, 
          file.path(output_dir, "llm_llm_pairwise_agreement.csv"))
write_csv(llm_llm_summary_by_feature, 
          file.path(output_dir, "llm_llm_agreement_by_feature.csv"))
write_csv(llm_llm_summary_by_pair, 
          file.path(output_dir, "llm_llm_agreement_by_pair.csv"))

# ============================================================================
# Plot: LLM-LLM Agreement Heatmap
# ============================================================================

# Create symmetric matrix for heatmap
create_agreement_matrix <- function(data, model_labels) {
  models <- names(model_labels)
  n_models <- length(models)
  mat <- matrix(NA, nrow = n_models, ncol = n_models,
                dimnames = list(model_labels, model_labels))
  
  # Fill diagonal with 1 (self-agreement)
  diag(mat) <- 1
  
  # Aggregate mean kappa across features for each pair
  pair_means <- data %>%
    group_by(model1, model2) %>%
    summarise(mean_kappa = mean(kappa, na.rm = TRUE), .groups = "drop")
  
  for (i in 1:nrow(pair_means)) {
    m1_label <- model_labels[pair_means$model1[i]]
    m2_label <- model_labels[pair_means$model2[i]]
    mat[m1_label, m2_label] <- pair_means$mean_kappa[i]
    mat[m2_label, m1_label] <- pair_means$mean_kappa[i]
  }
  
  return(mat)
}

agreement_matrix <- create_agreement_matrix(llm_llm_agreement, llm_models)

# Convert to long format for ggplot
agreement_matrix_long <- as.data.frame(as.table(agreement_matrix)) %>%
  rename(Model1 = Var1, Model2 = Var2, Kappa = Freq)

# Get mean human-human kappa for reference
mean_human_kappa <- mean(human_reliability$human_kappa)

p5 <- ggplot(agreement_matrix_long, aes(x = Model1, y = Model2, fill = Kappa)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = sprintf("%.2f", Kappa)), size = 3.5, color = "white") +
  scale_fill_gradient2(
    low = "#e74c3c", mid = "#f39c12", high = "#27ae60",
    midpoint = mean_human_kappa,
    limits = c(0, 1),
    name = "Cohen's κ"
  ) +
  labs(
    title = "LLM-LLM Inter-Coder Agreement",
    subtitle = sprintf("Mean kappa across all features (Human-Human baseline: κ = %.2f)", mean_human_kappa),
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 10, color = "grey40"),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(hjust = 1),
    panel.grid = element_blank()
  ) +
  coord_fixed()

ggsave(file.path(output_dir, "llm_llm_agreement_heatmap.pdf"), 
       p5, width = 10, height = 8, device = cairo_pdf)

# ============================================================================
# Plot: LLM-LLM vs Human-Human Agreement Comparison
# ============================================================================

comparison_data <- llm_llm_summary_by_feature %>%
  select(feature_label, mean_llm_llm_kappa, human_human_kappa) %>%
  pivot_longer(cols = c(mean_llm_llm_kappa, human_human_kappa),
               names_to = "comparison_type",
               values_to = "kappa") %>%
  mutate(
    comparison_type = ifelse(comparison_type == "mean_llm_llm_kappa",
                            "LLM-LLM (Mean)", "Human-Human")
  )

p6 <- ggplot(comparison_data, 
             aes(x = reorder(feature_label, kappa), y = kappa, fill = comparison_type)) +
  geom_col(position = "dodge", alpha = 0.85, width = 0.7) +
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "darkgreen", size = 0.5) +
  geom_hline(yintercept = 0.6, linetype = "dashed", color = "orange", size = 0.5) +
  coord_flip() +
  scale_fill_manual(
    values = c("Human-Human" = "#3498db", "LLM-LLM (Mean)" = "#9b59b6"),
    name = "Comparison"
  ) +
  labs(
    title = "Human-Human vs LLM-LLM Agreement",
    subtitle = "Green line: Substantial (κ ≥ 0.8), Orange line: Moderate (κ ≥ 0.6)",
    x = NULL,
    y = "Cohen's Kappa"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 10, color = "grey40"),
    legend.position = "bottom"
  )

ggsave(file.path(output_dir, "llm_llm_vs_human_comparison.pdf"), 
       p6, width = 10, height = 6, device = cairo_pdf)

# ============================================================================
# Plot: Feature-wise LLM-LLM Agreement Distribution
# ============================================================================

p7 <- ggplot(llm_llm_agreement, 
             aes(x = reorder(feature_label, kappa, FUN = median), y = kappa)) +
  geom_boxplot(aes(fill = feature_label), alpha = 0.7, outlier.shape = NA) +
  geom_jitter(alpha = 0.4, width = 0.15, size = 2) +
  geom_hline(data = human_reliability %>% 
               mutate(feature_label = feature_labels[match(feature, features)]),
             aes(yintercept = human_kappa),
             linetype = "dashed", color = "red", size = 0.8) +
  facet_wrap(~feature_label, scales = "free_x", nrow = 2) +
  coord_flip() +
  labs(
    title = "LLM-LLM Agreement Distribution by Feature",
    subtitle = "Red dashed line = Human-Human agreement (baseline)",
    x = NULL,
    y = "Cohen's Kappa"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "none",
    strip.text = element_text(size = 9, face = "bold")
  ) +
  scale_fill_brewer(palette = "Set2")

ggsave(file.path(output_dir, "llm_llm_agreement_by_feature.pdf"), 
       p7, width = 14, height = 8, device = cairo_pdf)

# ============================================================================
# Summary: LLM-LLM Analysis
# ============================================================================

cat("\n=== LLM-LLM AGREEMENT SUMMARY ===\n\n")

llm_llm_final_summary <- tibble(
  metric = c(
    "Mean LLM-LLM kappa (all pairs, all features)",
    "Mean Human-Human kappa (all features)",
    "LLM-LLM vs Human-Human difference",
    "Number of LLM pairs",
    "Number of features",
    "Features where LLM-LLM > Human-Human",
    "Most agreeing LLM pair",
    "Least agreeing LLM pair"
  ),
  value = c(
    sprintf("%.3f", mean(llm_llm_agreement$kappa, na.rm = TRUE)),
    sprintf("%.3f", mean(human_reliability$human_kappa)),
    sprintf("%.3f", mean(llm_llm_agreement$kappa, na.rm = TRUE) - mean(human_reliability$human_kappa)),
    as.character(nrow(llm_llm_summary_by_pair)),
    as.character(length(features)),
    as.character(sum(llm_llm_summary_by_feature$llm_vs_human_diff > 0, na.rm = TRUE)),
    paste0(llm_llm_summary_by_pair$model1_label[1], " & ", 
           llm_llm_summary_by_pair$model2_label[1],
           " (κ = ", sprintf("%.3f", llm_llm_summary_by_pair$mean_kappa[1]), ")"),
    paste0(llm_llm_summary_by_pair$model1_label[nrow(llm_llm_summary_by_pair)], " & ", 
           llm_llm_summary_by_pair$model2_label[nrow(llm_llm_summary_by_pair)],
           " (κ = ", sprintf("%.3f", llm_llm_summary_by_pair$mean_kappa[nrow(llm_llm_summary_by_pair)]), ")")
  )
)

print(llm_llm_final_summary, n = Inf)

write_csv(llm_llm_final_summary, 
          file.path(output_dir, "llm_llm_summary.csv"))

cat("\n=== ANALYSIS COMPLETE ===\n")
cat(sprintf("All outputs saved to: %s\n", output_dir))
