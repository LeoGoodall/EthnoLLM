suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(ggplot2)
  library(purrr)
  library(forcats)
  library(cowplot)
})

# Inputs and outputs
results_dir <- "synchrony"
features_csv <- "data/features_synchrony.csv"
fig_dir <- file.path("figures_R", "synchrony")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

# Model and condition setup
model_bases <- c("llama33b", "qwen3", "gptoss120b", "deepseekv31671b", "gpt5nano", "claudesonnet45", "perplexity")
conditions <- c("baseline", "mtp", "ensemble_mtp")
cond_label <- c(baseline = "Baseline", mtp = "MTP", ensemble_mtp = "MTP+Ensemble")
model_display <- c(
  llama33b = "Llama 3.2 Instruct (3B)",
  qwen3 = "Qwen 3 Instruct (4B)",
  gptoss120b = "GPT-OSS (120B)",
  deepseekv31671b = "DeepSeek V3.1 (671B)",
  gpt5nano = "GPT-5 Nano",
  claudesonnet45 = "Claude Sonnet 4.5",
  perplexity = "Perplexity Sonar"
)

# Color palette
palette_map <- c(
  "qwen3|baseline" = "#B2E6E6",
  "qwen3|mtp" = "#80D4D4",
  "qwen3|ensemble_mtp" = "#26A69A",
  "gptoss120b|baseline" = "#CAE6D0",
  "gptoss120b|mtp" = "#A8D3B3",
  "gptoss120b|ensemble_mtp" = "#55A868",
  "deepseekv31671b|baseline" = "#F1B8BA",
  "deepseekv31671b|mtp" = "#E19699",
  "deepseekv31671b|ensemble_mtp" = "#C44E52",
  "gpt5nano|baseline" = "#D6B8E9",
  "gpt5nano|mtp" = "#B89ACF",
  "gpt5nano|ensemble_mtp" = "#7A4C9A",
  "llama33b|baseline" = "#FFE0B2",
  "llama33b|mtp" = "#FFB74D",
  "llama33b|ensemble_mtp" = "#F57C00",
  "claudesonnet45|baseline" = "#C6DAF7",
  "claudesonnet45|mtp" = "#7FB3E5",
  "claudesonnet45|ensemble_mtp" = "#186BB6",
  "perplexity|baseline" = "#F8BBD0",
  "perplexity|mtp" = "#F48FB1",
  "perplexity|ensemble_mtp" = "#E91E63"
)

# Utility: parse model/cond from filename
parse_model_condition <- function(path) {
  fn <- basename(path)
  stem <- sub("^results_", "", sub("\\.csv$", "", fn))
  parts <- strsplit(stem, "_")[[1]]
  base <- parts[1]
  cond <- "baseline"
  if (length(parts) > 1) {
    rest <- paste(parts[-1], collapse = "_")
    if (rest %in% conditions[-1]) cond <- rest
  }
  list(base = base, condition = cond)
}

# Load feature metadata (6 binary features)
features_meta <- read_csv(features_csv, show_col_types = FALSE) %>%
  transmute(
    feature_variable = feature_variable,
    feature_name = feature_name,
    type = "binary"
  )

to_num <- function(x) suppressWarnings(as.numeric(x))

metrics_binary <- function(y_true, y_pred) {
  y_true <- as.integer(ifelse(is.na(y_true), NA, ifelse(to_num(y_true) != 0, 1L, 0L)))
  y_pred <- as.integer(ifelse(is.na(y_pred), NA, ifelse(to_num(y_pred) != 0, 1L, 0L)))
  keep <- !is.na(y_true) & !is.na(y_pred)
  y_true <- y_true[keep]
  y_pred <- y_pred[keep]
  if (length(y_true) == 0) return(c(precision = NA_real_, recall = NA_real_, f1 = NA_real_))
  tp <- sum(y_true == 1L & y_pred == 1L)
  fp <- sum(y_true == 0L & y_pred == 1L)
  fn <- sum(y_true == 1L & y_pred == 0L)
  # If model never predicts positive (tp+fp=0), precision=0
  precision <- if ((tp + fp) > 0) tp/(tp + fp) else 0
  recall <- if ((tp + fn) > 0) tp/(tp + fn) else 0
  f1 <- if ((precision + recall) > 0) 2 * precision * recall/(precision + recall) else 0
  c(precision = precision, recall = recall, f1 = f1)
}

compute_feature_metrics <- function(res_df, feature_row) {
  feat <- feature_row$feature_variable
  fname <- feature_row$feature_name
  col_true <- feat
  col_pred <- sub("_human$", "_llm", feat)
  if (!(col_true %in% names(res_df)) || !(col_pred %in% names(res_df))) return(NULL)
  yt <- res_df[[col_true]]
  yp <- res_df[[col_pred]]
  m <- metrics_binary(yt, yp)
  tibble(Feature = feat, FeatureName = fname, Precision = m[["precision"]], Recall = m[["recall"]], F1 = m[["f1"]])
}

all_files <- list.files(results_dir, pattern = "^results_.*\\.csv$", full.names = TRUE)
all_files <- all_files[grepl(paste(model_bases, collapse = "|"), all_files)]
if (length(all_files) == 0) stop("No synchrony result files found.")

# Load exclude list
exclude_rituals <- read_csv("data/exclude.csv", show_col_types = FALSE) %>%
  pull(exclude)

per_feature_list <- list()
per_model_metrics <- list()

for (f in all_files) {
  info <- parse_model_condition(f)
  base <- info$base; cond <- info$condition
  if (!(base %in% model_bases)) next
  df <- read_csv(f, show_col_types = FALSE) %>%
    filter(!ritual_number %in% exclude_rituals)

  feat_metrics <- purrr::map_dfr(seq_len(nrow(features_meta)), function(i) compute_feature_metrics(df, features_meta[i, ]))
  if (nrow(feat_metrics) > 0) {
    feat_metrics <- feat_metrics %>% mutate(ModelBase = base, Condition = cond)
    per_feature_list[[paste(base, cond, sep = "|")]] <- feat_metrics
  }

  if (nrow(feat_metrics) > 0) {
    mm <- feat_metrics %>% summarise(
      F1_se = sd(F1, na.rm = TRUE) / sqrt(sum(!is.na(F1))),
      Precision_se = sd(Precision, na.rm = TRUE) / sqrt(sum(!is.na(Precision))),
      Recall_se = sd(Recall, na.rm = TRUE) / sqrt(sum(!is.na(Recall))),
      F1 = mean(F1, na.rm = TRUE),
      Precision = mean(Precision, na.rm = TRUE),
      Recall = mean(Recall, na.rm = TRUE)
    ) %>% mutate(ModelBase = base, Condition = cond)
    per_model_metrics[[paste(base, cond, sep = "|")]] <- mm
  }
}

per_feature <- bind_rows(per_feature_list)
per_model <- bind_rows(per_model_metrics)

if (nrow(per_model) == 0) stop("No metrics computed for synchrony.")

# Save per-feature and per-model metrics
write_csv(per_feature, file.path(fig_dir, "per_feature_metrics.csv"))
write_csv(per_model, file.path(fig_dir, "per_model_metrics.csv"))
message("Saved: ", file.path(fig_dir, "per_feature_metrics.csv"))
message("Saved: ", file.path(fig_dir, "per_model_metrics.csv"))

# Ordering
per_model <- per_model %>% mutate(ModelBase = factor(ModelBase, levels = model_bases), Condition = factor(Condition, levels = conditions))
# Pivot scores and SEs separately, then join
scores_long <- per_model %>% 
  select(ModelBase, Condition, F1, Precision, Recall) %>%
  pivot_longer(cols = c(F1, Precision, Recall), names_to = "Metric", values_to = "Score")

se_long <- per_model %>% 
  select(ModelBase, Condition, F1_se, Precision_se, Recall_se) %>%
  pivot_longer(cols = c(F1_se, Precision_se, Recall_se), names_to = "Metric", values_to = "SE") %>%
  mutate(Metric = gsub("_se$", "", Metric))

perf_long <- scores_long %>%
  left_join(se_long, by = c("ModelBase", "Condition", "Metric")) %>%
  mutate(Metric = factor(Metric, levels = c("F1", "Precision", "Recall"))) %>%
  # Pad missing conditions per model and metric so dodged bars keep fixed slots
  tidyr::complete(ModelBase, Metric, Condition = factor(conditions, levels = conditions), fill = list(Score = NA_real_, SE = NA_real_))
fill_levels <- as.vector(outer(model_bases, conditions, paste, sep = "|"))
perf_long <- perf_long %>% mutate(fill_key = factor(paste(ModelBase, Condition, sep = "|"), levels = fill_levels))

# Performance bars
bar_plot_base <- ggplot(perf_long, aes(x = ModelBase, y = Score, fill = fill_key)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  geom_errorbar(aes(ymin = Score - SE, ymax = Score + SE), 
                position = position_dodge(width = 0.75), width = 0.2, linewidth = 0.3, alpha = 0.3) +
  facet_wrap(~ Metric, nrow = 1) +
  scale_fill_manual(values = palette_map[levels(perf_long$fill_key)], breaks = levels(perf_long$fill_key)) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = NULL, y = "Score", title = "Model Performance Comparison — Synchrony Features") +
  theme_bw() +
  theme(legend.position = "none", strip.background = element_rect(fill = "white"), strip.text = element_text(face = "bold"), axis.text.x = element_text(angle = 45, hjust = 1), plot.margin = margin(t = 5, r = 5, b = 5, l = 15)) +
  scale_x_discrete(labels = model_display[levels(perf_long$ModelBase)])

cond_cols <- setNames(palette_map[paste0("gptoss20b|", conditions)], conditions)
legend_df <- tibble(Condition = factor(names(cond_label), levels = conditions), y = 1)
legend_plot <- ggplot(legend_df, aes(x = Condition, y = y, fill = Condition)) +
  geom_col() +
  scale_fill_manual(values = c(baseline = "#e0e0e0", ensemble = "#bdbdbd", mtp = "#757575", ensemble_mtp = "#424242"), labels = cond_label, name = "Condition") +
  theme_void() + theme(legend.position = "bottom")
legend_g <- cowplot::get_legend(legend_plot)
bar_plot <- cowplot::plot_grid(bar_plot_base, legend_g, ncol = 1, rel_heights = c(1, 0.12))
ggsave(file.path(fig_dir, "model_performance_bar_synchrony.pdf"), bar_plot, width = 11, height = 5.2)

# Heatmap by feature (rows), model+condition columns
cat_f1 <- per_feature %>% group_by(FeatureName, ModelBase, Condition) %>% summarise(F1 = mean(F1, na.rm = TRUE), .groups = "drop") %>%
  mutate(ModelBase = factor(ModelBase, levels = model_bases), Condition = factor(Condition, levels = conditions))

# Order features by average F1 across all models/conditions (highest at top)
feat_order <- cat_f1 %>%
  group_by(FeatureName) %>%
  summarise(avgF1 = mean(F1, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(avgF1)) %>%
  pull(FeatureName)
cat_f1 <- cat_f1 %>% mutate(FeatureName = factor(FeatureName, levels = rev(feat_order)))

cat_f1 <- cat_f1 %>% mutate(Col = paste(ModelBase, cond_label[as.character(Condition)], sep = " | "))
col_order <- unlist(lapply(model_bases, function(m) paste(m, cond_label[conditions], sep = " | ")))
heat_df <- cat_f1 %>% select(FeatureName, Col, F1) %>% pivot_wider(names_from = Col, values_from = F1)
heat_long <- heat_df %>% pivot_longer(-FeatureName, names_to = "ModelCond", values_to = "F1") %>% mutate(ModelCond = factor(ModelCond, levels = col_order))
x_levels <- levels(droplevels(heat_long$ModelCond))
boundaries <- numeric(0)
if (!is.null(x_levels)) {
  model_of_level <- sub(" \\| .*", "", x_levels)
  run_lengths <- rle(model_of_level)$lengths
  cum <- cumsum(run_lengths)
  if (length(cum) > 1) boundaries <- cum[-length(cum)] + 0.5
}

heat_plot_core <- ggplot(heat_long, aes(x = ModelCond, y = FeatureName, fill = F1)) +
  geom_tile(color = NA) +
  geom_text(aes(label = ifelse(is.na(F1), "", sprintf("%.2f", F1))), size = 2.4) +
  { if (length(boundaries) > 0) geom_vline(xintercept = boundaries, color = "black", linewidth = 1) } +
  scale_x_discrete(labels = function(l) sub("^.* \\| ", "", l), expand = ggplot2::expansion(mult = c(0,0), add = c(0,0))) +
  scale_fill_gradient(low = "#FFF5CC", high = "#D7301F", na.value = "#F0F0F0", limits = c(0, 1)) +
  labs(x = NULL, y = NULL, title = "Feature-level F1 Score by Model — Synchrony Features") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right", plot.margin = margin(t = 5, r = 5, b = 2, l = 5))

model_label_plot <- NULL
if (!is.null(x_levels) && length(x_levels) > 0) {
  model_of_level <- sub(" \\| .*", "", x_levels)
  rl <- rle(model_of_level)
  centers <- cumsum(rl$lengths) - rl$lengths/2
  model_labels <- model_display[rl$values]
  model_df <- tibble(x = centers, y = -8, label = model_labels)
  model_label_plot <- ggplot(model_df, aes(x = x, y = y, label = label)) +
    geom_text(fontface = "bold", size = 3.6, angle = 90, hjust = 0, vjust = 1) +
    scale_x_continuous(limits = c(0.5, length(x_levels) + 0.5), expand = ggplot2::expansion(mult = c(0,0), add = c(0,0))) +
    scale_y_continuous(limits = c(-9, 0.5), expand = ggplot2::expansion(mult = c(0,0), add = c(0,0))) +
    theme_void() +
    theme(plot.margin = margin(t = 0, r = 5, b = 2, l = 5))
}

heat_plot <- if (!is.null(model_label_plot)) cowplot::plot_grid(heat_plot_core, model_label_plot, ncol = 1, rel_heights = c(1, 0.25), align = 'v', axis = 'lr') else heat_plot_core
ggsave(file.path(fig_dir, "feature_f1_heatmap_synchrony.pdf"), heat_plot, width = 11, height = 8)

message("Saved: ", file.path(fig_dir, "model_performance_bar_synchrony.pdf"))
message("Saved: ", file.path(fig_dir, "feature_f1_heatmap_synchrony.pdf"))

# MAJORITY-CLASS BASELINE F1 COMPUTATION
# Computes F1 score for a naive classifier that always predicts the majority class

# Compute majority-class baseline F1 for binary features
majority_baseline_binary <- function(y_true) {
  y <- as.integer(to_num(y_true))
  y <- y[!is.na(y)]
  y <- ifelse(y != 0L, 1L, 0L)  # Binarise
  if (length(y) == 0) return(NA_real_)
  
  n1 <- sum(y == 1L)
  n0 <- sum(y == 0L)
  majority_class <- ifelse(n1 >= n0, 1L, 0L)
  
  if (majority_class == 0L) {
    # Always predict 0: TP=0, FP=0, FN=n1 -> Precision undefined, Recall=0, F1=0
    return(0)
  } else {
    # Always predict 1: TP=n1, FP=n0, FN=0 -> Precision=n1/(n1+n0), Recall=1
    precision <- n1 / (n1 + n0)
    recall <- 1.0
    f1 <- 2 * precision * recall / (precision + recall)
    return(f1)
  }
}

# Compute baseline for each feature using first available result file
message("\nComputing majority-class baseline F1 for each feature...")

baseline_file <- all_files[1]
baseline_df <- read_csv(baseline_file, show_col_types = FALSE) %>%
  filter(!ritual_number %in% exclude_rituals)

baseline_metrics <- purrr::map_dfr(seq_len(nrow(features_meta)), function(i) {
  feat <- features_meta$feature_variable[i]
  fname <- features_meta$feature_name[i]
  
  if (!(feat %in% names(baseline_df))) return(NULL)
  
  y_true <- baseline_df[[feat]]
  baseline_f1 <- majority_baseline_binary(y_true)
  
  # Compute base rate (proportion of positive class)
  y <- to_num(y_true)
  y <- y[!is.na(y)]
  y <- ifelse(y != 0, 1, 0)
  base_rate <- mean(y == 1)
  
  tibble(
    Feature = feat,
    FeatureName = fname,
    BaseRate = base_rate,
    MajorityBaselineF1 = baseline_f1
  )
})

# Merge with LLM performance to compare
best_llm_per_feature <- per_feature %>%
  group_by(Feature) %>%
  summarise(BestLLM_F1 = max(F1, na.rm = TRUE), .groups = "drop")

baseline_comparison <- baseline_metrics %>%
  left_join(best_llm_per_feature, by = "Feature") %>%
  mutate(
    LLM_vs_Baseline = BestLLM_F1 - MajorityBaselineF1,
    LLM_Beats_Baseline = BestLLM_F1 > MajorityBaselineF1
  ) %>%
  arrange(desc(LLM_vs_Baseline))

# Save baseline comparison
write_csv(baseline_comparison, file.path(fig_dir, "majority_baseline_comparison.csv"))
message("Saved: ", file.path(fig_dir, "majority_baseline_comparison.csv"))

# Summary statistics
n_beats_baseline <- sum(baseline_comparison$LLM_Beats_Baseline, na.rm = TRUE)
n_total <- sum(!is.na(baseline_comparison$LLM_Beats_Baseline))
mean_improvement <- mean(baseline_comparison$LLM_vs_Baseline, na.rm = TRUE)
mean_baseline <- mean(baseline_comparison$MajorityBaselineF1, na.rm = TRUE)
mean_llm <- mean(baseline_comparison$BestLLM_F1, na.rm = TRUE)

summary_baseline <- tibble(
  Metric = c("Features where LLM beats baseline", 
             "Total features", 
             "Proportion beating baseline",
             "Mean majority-class baseline F1",
             "Mean best LLM F1",
             "Mean improvement over baseline"),
  Value = c(n_beats_baseline, 
            n_total, 
            round(n_beats_baseline/n_total, 3),
            round(mean_baseline, 3),
            round(mean_llm, 3),
            round(mean_improvement, 3))
)

write_csv(summary_baseline, file.path(fig_dir, "majority_baseline_summary.csv"))
message("Saved: ", file.path(fig_dir, "majority_baseline_summary.csv"))