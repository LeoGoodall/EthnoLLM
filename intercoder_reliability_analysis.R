# Inter-coder Reliability Analysis for Synchrony Dataset
# For Nature Machine Intelligence paper
# Compares dual independent human coder annotations

library(tidyverse)
library(irr)        # For Cohen's kappa and other reliability metrics
library(psych)      # For additional reliability metrics
library(ggplot2)
library(gridExtra)
library(knitr)

output_dir <- "intercoder_reliability"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Load data
data <- read_csv("data/rituals_codes.csv")

# Define synchrony features
features <- c("singing", "chanting", "praying", "marching", "dancing", "generic_mvmt")

# Convert all coder columns to numeric (some may be read as character)
for (feat in features) {
  coder1_col <- paste0(feat, "_coder_1")
  coder2_col <- paste0(feat, "_coder_2")
  data[[coder1_col]] <- as.numeric(data[[coder1_col]])
  data[[coder2_col]] <- as.numeric(data[[coder2_col]])
}
feature_labels <- c(
  "Synchronous Singing",
  "Synchronous Chanting",
  "Synchronous Praying",
  "Synchronous Marching",
  "Synchronous Dancing",
  "Synchronous Generic Movement"
)

# 1. INTER-RATER RELIABILITY METRICS
cat("Computing inter-rater reliability metrics...\n")

# Function to compute comprehensive reliability metrics for a single feature
compute_reliability_metrics <- function(coder1, coder2, feature_name) {
  # Remove missing values
  complete_cases <- !is.na(coder1) & !is.na(coder2)
  c1 <- coder1[complete_cases]
  c2 <- coder2[complete_cases]
  
  n <- length(c1)
  
  # Raw agreement
  agreement <- sum(c1 == c2) / n
  
  # Cohen's Kappa (for binary nominal data)
  kappa_result <- kappa2(data.frame(c1, c2), weight = "unweighted")
  kappa_value <- kappa_result$value
  
  # Confusion matrix elements
  both_present <- sum(c1 == 1 & c2 == 1)
  both_absent <- sum(c1 == 0 & c2 == 0)
  c1_only <- sum(c1 == 1 & c2 == 0)
  c2_only <- sum(c1 == 0 & c2 == 1)
  
  # Prevalence and bias
  prevalence_c1 <- sum(c1 == 1) / n
  prevalence_c2 <- sum(c2 == 1) / n
  prevalence_diff <- abs(prevalence_c1 - prevalence_c2)
  
  # Positive and negative agreement (for binary data)
  if ((both_present + c1_only + c2_only) > 0) {
    positive_agreement <- (2 * both_present) / (2 * both_present + c1_only + c2_only)
  } else {
    positive_agreement <- NA
  }
  
  if ((both_absent + c1_only + c2_only) > 0) {
    negative_agreement <- (2 * both_absent) / (2 * both_absent + c1_only + c2_only)
  } else {
    negative_agreement <- NA
  }
  
  # Return results
  tibble(
    feature = feature_name,
    n = n,
    raw_agreement = agreement,
    cohen_kappa = kappa_value,
    positive_agreement = positive_agreement,
    negative_agreement = negative_agreement,
    prevalence_c1 = prevalence_c1,
    prevalence_c2 = prevalence_c2,
    prevalence_diff = prevalence_diff,
    both_present = both_present,
    both_absent = both_absent,
    c1_only = c1_only,
    c2_only = c2_only
  )
}

# Compute for all features
reliability_results <- map_dfr(features, function(feat) {
  coder1_col <- paste0(feat, "_coder_1")
  coder2_col <- paste0(feat, "_coder_2")
  
  compute_reliability_metrics(
    data[[coder1_col]],
    data[[coder2_col]],
    feat
  )
})

# Add feature labels
reliability_results <- reliability_results %>%
  mutate(feature_label = feature_labels)

# Print summary table
cat("\n=== INTER-RATER RELIABILITY SUMMARY ===\n\n")
print(reliability_results %>%
  select(feature_label, n, raw_agreement, cohen_kappa, positive_agreement, negative_agreement) %>%
  mutate(across(where(is.numeric) & !n, ~round(., 3))),
  n = Inf)

# Save detailed results
write_csv(reliability_results, 
          file.path(output_dir, "intercoder_reliability_metrics.csv"))


# 2. CONFUSION MATRICES
cat("\n\nGenerating confusion matrices...\n")

# Function to create confusion matrix for a feature
create_confusion_matrix <- function(coder1, coder2, feature_name) {
  # Remove missing values
  complete_cases <- !is.na(coder1) & !is.na(coder2)
  c1 <- coder1[complete_cases]
  c2 <- coder2[complete_cases]
  
  # Create confusion matrix
  conf_mat <- table(Coder1 = c1, Coder2 = c2)
  
  # Convert to data frame for plotting
  conf_df <- as.data.frame(conf_mat) %>%
    mutate(
      Coder1 = factor(Coder1, levels = c("1", "0"), labels = c("Present", "Absent")),
      Coder2 = factor(Coder2, levels = c("1", "0"), labels = c("Present", "Absent")),
      feature = feature_name
    )
  
  return(list(matrix = conf_mat, df = conf_df))
}

# Generate confusion matrices for all features
confusion_matrices <- map(features, function(feat) {
  coder1_col <- paste0(feat, "_coder_1")
  coder2_col <- paste0(feat, "_coder_2")
  
  create_confusion_matrix(
    data[[coder1_col]],
    data[[coder2_col]],
    feat
  )
})
names(confusion_matrices) <- features

# Print confusion matrices
cat("\n=== CONFUSION MATRICES ===\n\n")
for (i in seq_along(features)) {
  cat(feature_labels[i], ":\n")
  print(confusion_matrices[[i]]$matrix)
  cat("\n")
}


# 3. DISAGREEMENT ANALYSIS
cat("\nAnalysing disagreement patterns...\n")

# Create disagreement dataset
disagreement_data <- map_dfr(features, function(feat) {
  coder1_col <- paste0(feat, "_coder_1")
  coder2_col <- paste0(feat, "_coder_2")
  
  data %>%
    select(ritual_number, ritual_name, 
           coder1 = all_of(coder1_col),
           coder2 = all_of(coder2_col),
           text) %>%
    filter(!is.na(coder1) & !is.na(coder2)) %>%
    mutate(
      feature = feat,
      feature_label = feature_labels[which(features == feat)],
      agree = coder1 == coder2,
      disagreement_type = case_when(
        coder1 == coder2 ~ "Agreement",
        coder1 == 1 & coder2 == 0 ~ "Coder1 Only",
        coder1 == 0 & coder2 == 1 ~ "Coder2 Only"
      )
    )
})

# Overall disagreement summary
disagreement_summary <- disagreement_data %>%
  group_by(feature, feature_label) %>%
  summarise(
    total_cases = n(),
    agreements = sum(agree),
    disagreements = sum(!agree),
    disagreement_rate = mean(!agree),
    coder1_only = sum(disagreement_type == "Coder1 Only"),
    coder2_only = sum(disagreement_type == "Coder2 Only"),
    .groups = "drop"
  )

cat("\n=== DISAGREEMENT SUMMARY ===\n\n")
print(disagreement_summary %>%
  mutate(across(where(is.numeric) & !c(total_cases, agreements, disagreements, coder1_only, coder2_only), 
                ~round(., 3))),
  n = Inf)

write_csv(disagreement_summary, 
          file.path(output_dir, "disagreement_summary.csv"))

# Test for systematic bias (Chi-square test)
cat("\n=== SYSTEMATIC BIAS TEST ===\n\n")
bias_tests <- map_dfr(features, function(feat) {
  coder1_col <- paste0(feat, "_coder_1")
  coder2_col <- paste0(feat, "_coder_2")
  
  c1 <- data[[coder1_col]]
  c2 <- data[[coder2_col]]
  
  # Remove missing values
  complete_cases <- !is.na(c1) & !is.na(c2)
  c1 <- c1[complete_cases]
  c2 <- c2[complete_cases]
  
  # McNemar's test for systematic bias in binary paired data
  disagreements <- data.frame(c1, c2)
  contingency <- table(c1, c2)
  
  # Only run if there are disagreements
  if (sum(contingency[1,2], contingency[2,1]) > 0) {
    mcnemar_result <- mcnemar.test(contingency, correct = TRUE)
    
    tibble(
      feature = feat,
      feature_label = feature_labels[which(features == feat)],
      chi_square = mcnemar_result$statistic,
      p_value = mcnemar_result$p.value,
      significant = p_value < 0.05,
      interpretation = ifelse(significant, 
                             "Systematic bias detected",
                             "No systematic bias")
    )
  } else {
    tibble(
      feature = feat,
      feature_label = feature_labels[which(features == feat)],
      chi_square = NA,
      p_value = NA,
      significant = FALSE,
      interpretation = "Perfect agreement"
    )
  }
})

print(bias_tests %>%
  mutate(across(where(is.numeric), ~round(., 4))),
  n = Inf)

write_csv(bias_tests, 
          file.path(output_dir, "systematic_bias_tests.csv"))


# 4. VISUALISATIONS
cat("\nCreating visualisations...\n")

# Plot 1: Reliability metrics comparison
p1 <- ggplot(reliability_results, 
             aes(x = reorder(feature_label, cohen_kappa), y = cohen_kappa)) +
  geom_col(fill = "#3498db", alpha = 0.8) +
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "darkgreen", size = 0.7) +
  geom_hline(yintercept = 0.6, linetype = "dashed", color = "orange", size = 0.7) +
  geom_hline(yintercept = 0.4, linetype = "dashed", color = "red", size = 0.7) +
  coord_flip() +
  labs(
    title = "Inter-coder Reliability (Cohen's Kappa)",
    subtitle = "Green: Substantial (>0.8), Orange: Moderate (>0.6), Red: Fair (>0.4)",
    x = NULL,
    y = "Cohen's Kappa"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "grey40")
  )

ggsave(file.path(output_dir, "kappa_comparison.pdf"), 
       p1, width = 10, height = 6, device = "pdf")

# Plot 2: Raw agreement vs Kappa
p2 <- ggplot(reliability_results, 
             aes(x = raw_agreement, y = cohen_kappa)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
  geom_point(size = 4, color = "#e74c3c", alpha = 0.7) +
  geom_text(aes(label = feature_label), vjust = -0.8, size = 3) +
  labs(
    title = "Raw Agreement vs Cohen's Kappa",
    subtitle = "Kappa adjusts for chance agreement",
    x = "Raw Agreement",
    y = "Cohen's Kappa"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14)
  ) +
  coord_cartesian(xlim = c(0.7, 1), ylim = c(0.4, 1))

ggsave(file.path(output_dir, "agreement_vs_kappa.pdf"), 
       p2, width = 8, height = 6, device = "pdf")

# Plot 3: Confusion matrices heatmaps
confusion_plots <- map2(features, feature_labels, function(feat, label) {
  conf_df <- confusion_matrices[[feat]]$df
  
  ggplot(conf_df, aes(x = Coder2, y = Coder1, fill = Freq)) +
    geom_tile(color = "white", size = 1) +
    geom_text(aes(label = Freq), size = 6, fontface = "bold") +
    scale_fill_gradient(low = "#ecf0f1", high = "#3498db", 
                       name = "Count") +
    labs(
      title = label,
      x = "Coder 2",
      y = "Coder 1"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
      axis.text = element_text(size = 10),
      legend.position = "right"
    ) +
    coord_fixed()
})

# Arrange confusion matrices in grid
p3 <- arrangeGrob(grobs = confusion_plots, ncol = 3)
ggsave(file.path(output_dir, "confusion_matrices.pdf"), 
       p3, width = 15, height = 10, device = "pdf")

# Plot 4: Disagreement patterns
p4 <- ggplot(disagreement_summary, 
             aes(x = reorder(feature_label, disagreement_rate))) +
  geom_col(aes(y = coder1_only), fill = "#e74c3c", alpha = 0.7, width = 0.7) +
  geom_col(aes(y = -coder2_only), fill = "#3498db", alpha = 0.7, width = 0.7) +
  coord_flip() +
  labs(
    title = "Disagreement Patterns by Coder",
    subtitle = "Red: Coder 1 detected only | Blue: Coder 2 detected only",
    x = NULL,
    y = "Number of Disagreements"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "grey40")
  ) +
  geom_hline(yintercept = 0, color = "black", size = 0.5)

ggsave(file.path(output_dir, "disagreement_patterns.pdf"), 
       p4, width = 10, height = 6, device = "pdf")

# Plot 5: Positive vs Negative Agreement
agreement_long <- reliability_results %>%
  select(feature_label, positive_agreement, negative_agreement) %>%
  pivot_longer(cols = c(positive_agreement, negative_agreement),
               names_to = "agreement_type",
               values_to = "value") %>%
  mutate(agreement_type = ifelse(agreement_type == "positive_agreement",
                                "Positive Agreement", "Negative Agreement"))

p5 <- ggplot(agreement_long, 
             aes(x = reorder(feature_label, value), y = value, fill = agreement_type)) +
  geom_col(position = "dodge", alpha = 0.8) +
  coord_flip() +
  scale_fill_manual(values = c("Positive Agreement" = "#27ae60", 
                               "Negative Agreement" = "#95a5a6"),
                   name = NULL) +
  labs(
    title = "Positive vs Negative Agreement",
    subtitle = "Positive: Both code as present | Negative: Both code as absent",
    x = NULL,
    y = "Agreement Rate"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "grey40"),
    legend.position = "bottom"
  )

ggsave(file.path(output_dir, "positive_negative_agreement.pdf"), 
       p5, width = 10, height = 6, device = "pdf")


# 5. OVERALL SUMMARY STATISTICS

overall_stats <- tibble(
  metric = c(
    "Mean Cohen's Kappa",
    "Median Cohen's Kappa",
    "Mean Raw Agreement",
    "Mean Positive Agreement",
    "Mean Negative Agreement",
    "Total Disagreements",
    "Mean Disagreement Rate"
  ),
  value = c(
    mean(reliability_results$cohen_kappa, na.rm = TRUE),
    median(reliability_results$cohen_kappa, na.rm = TRUE),
    mean(reliability_results$raw_agreement, na.rm = TRUE),
    mean(reliability_results$positive_agreement, na.rm = TRUE),
    mean(reliability_results$negative_agreement, na.rm = TRUE),
    sum(disagreement_summary$disagreements),
    mean(disagreement_summary$disagreement_rate)
  )
)

print(overall_stats %>%
  mutate(value = round(value, 3)))

write_csv(overall_stats, 
          file.path(output_dir, "overall_summary.csv"))


# 6. EXPORT DISAGREEMENT CASES FOR QUALITATIVE REVIEW
cat("\nExporting disagreement cases...\n")

# Export all disagreement cases with text excerpts (first 200 chars)
disagreement_cases <- disagreement_data %>%
  filter(!agree) %>%
  mutate(text_excerpt = str_sub(text, 1, 200)) %>%
  select(ritual_number, ritual_name, feature_label, 
         coder1, coder2, disagreement_type, text_excerpt) %>%
  arrange(feature_label, ritual_number)

write_csv(disagreement_cases, 
          file.path(output_dir, "disagreement_cases.csv"))

cat(sprintf("\nExported %d disagreement cases for qualitative review.\n", 
            nrow(disagreement_cases)))


# 7. GENERATE LATEX TABLE FOR MANUSCRIPT
cat("\nGenerating LaTeX table...\n")

latex_table <- reliability_results %>%
  select(feature_label, n, raw_agreement, cohen_kappa, 
         positive_agreement, negative_agreement) %>%
  mutate(across(where(is.numeric) & !n, ~sprintf("%.3f", .)))

latex_output <- kable(latex_table, 
                     format = "latex",
                     col.names = c("Feature", "N", "Raw Agreement", 
                                 "Cohen's κ", "Positive Agreement", 
                                 "Negative Agreement"),
                     booktabs = TRUE,
                     caption = "Inter-coder reliability metrics for synchrony features")

cat(latex_output, 
    file = file.path(output_dir, "reliability_table.tex"))

cat(sprintf("All outputs saved to: %s\n", output_dir))
