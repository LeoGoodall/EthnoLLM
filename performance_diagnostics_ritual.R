suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(lme4)
  library(lmerTest)
  library(performance)
  library(sjPlot)
  library(purrr)
  library(stringr)
  library(koRpus)
  library(koRpus.lang.en)
  library(forcats)
  library(cowplot)
})

# ============================================================================
# CONFIGURATION
# ============================================================================

results_dir <- "all"
features_csv <- "data/features_all.csv"
rituals_csv <- "data/rituals_codes.csv"
exclude_csv <- "data/exclude.csv"
output_dir <- "performance_diagnostics_ritual"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Model and condition setup
model_bases <- c("gptoss120b", "deepseekv31671b", "gpt5nano", "llama33b", "claudesonnet45", "qwen3")
conditions <- c("baseline", "mtp", "ensemble_mtp")

excluded_features <- c("ParticipantPeakDysphoria", "IndividualExegesis", "DissolveUnion", "Disgust")

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

message("Loading data...")

# Load features metadata
features_meta <- read_csv(features_csv, show_col_types = FALSE) %>%
  filter(!(feature_variable %in% excluded_features))

# Determine feature type
is_binary_feature <- function(options_txt) {
  grepl("Present (1)/ Absent (0)", options_txt, fixed = TRUE)
}

features_meta <- features_meta %>%
  mutate(
    feature_type = ifelse(is_binary_feature(feature_options), "binary", "multiclass"),
    feature_description_length = nchar(feature_description)
  )

# Load rituals with metadata
rituals_meta <- read_csv(rituals_csv, show_col_types = FALSE)

# Load excluded rituals
excluded_rituals <- read_csv(exclude_csv, show_col_types = FALSE)$exclude

# Filter rituals
rituals_meta <- rituals_meta %>%
  filter(!(ritual_number %in% excluded_rituals))

# Parse model-condition from filename
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

# Load all results files and reshape to long format
message("Loading and reshaping results files...")

all_results <- list()
result_files <- list.files(results_dir, pattern = "^results_.*\\.csv$", full.names = TRUE)
result_files <- result_files[grepl(paste(model_bases, collapse = "|"), result_files)]

for (f in result_files) {
  info <- parse_model_condition(f)
  model <- info$base
  condition <- info$condition
  
  if (!(model %in% model_bases)) next
  
  df <- read_csv(f, show_col_types = FALSE)
  
  # Round ground truth for specific variables (preserve -1)
  if ("PeakEuphoria" %in% names(df)) {
    df$PeakEuphoria <- ifelse(as.numeric(df$PeakEuphoria) == -1, -1, round(as.numeric(df$PeakEuphoria)))
  }
  if ("PeakDysphoria" %in% names(df)) {
    df$PeakDysphoria <- ifelse(as.numeric(df$PeakDysphoria) == -1, -1, round(as.numeric(df$PeakDysphoria)))
  }
  
  # Reshape to long format
  feature_vars <- features_meta$feature_variable
  
  for (feat in feature_vars) {
    col_true <- feat
    col_pred <- paste0(feat, "_llm")
    
    if (!(col_true %in% names(df)) || !(col_pred %in% names(df))) next
    
    temp <- df %>%
      select(ritual_number, all_of(col_true), all_of(col_pred)) %>%
      mutate(
        feature_variable = feat,
        model = model,
        condition = condition,
        y_true = as.character(.data[[col_true]]),
        y_pred = as.character(.data[[col_pred]])
      ) %>%
      select(ritual_number, feature_variable, model, condition, y_true, y_pred)
    
    all_results[[paste(model, condition, feat, sep = "_")]] <- temp
  }
}

# Combine all results
performance_data <- bind_rows(all_results)

# ============================================================================
# 2. CALCULATE OUTCOME: F1 SCORES PER RITUAL
# ============================================================================

message("Calculating per-ritual F1 scores...")

# First, prepare the data
performance_data <- performance_data %>%
  mutate(
    y_true_num = suppressWarnings(as.numeric(y_true)),
    y_pred_num = suppressWarnings(as.numeric(y_pred)),
    # Skip comparisons where ground truth is -1 or 999 or NA or pred is NA
    skip_comparison = is.na(y_true_num) | is.na(y_pred_num) | 
                      y_true_num == -1 | y_true_num == 999
  ) %>%
  filter(!skip_comparison) %>%
  select(-skip_comparison)

message(sprintf("Total predictions: %d", nrow(performance_data)))

# Merge feature types before calculating F1
performance_data <- performance_data %>%
  left_join(
    features_meta %>% select(feature_variable, feature_type),
    by = "feature_variable"
  )

# Function to calculate F1 for a set of predictions
calculate_f1 <- function(y_true, y_pred, feature_types) {
  # Separate binary and multiclass features
  binary_idx <- feature_types == "binary"
  
  if (sum(binary_idx) > 0) {
    # Binary features: treat as 0/1
    yt_bin <- ifelse(y_true[binary_idx] != 0, 1, 0)
    yp_bin <- ifelse(y_pred[binary_idx] != 0, 1, 0)
    
    tp <- sum(yt_bin == 1 & yp_bin == 1)
    fp <- sum(yt_bin == 0 & yp_bin == 1)
    fn <- sum(yt_bin == 1 & yp_bin == 0)
    tn <- sum(yt_bin == 0 & yp_bin == 0)
    
    # Treat undefined precision/recall as 0 (model failed to predict this class)
    precision <- if ((tp + fp) > 0) tp/(tp + fp) else 0
    recall <- if ((tp + fn) > 0) tp/(tp + fn) else 0
    f1_binary <- if ((precision + recall) > 0) {
      2 * precision * recall/(precision + recall)
    } else {
      0
    }
  } else {
    f1_binary <- NA_real_
  }
  
  if (sum(!binary_idx) > 0) {
    # Multiclass features: macro F1
    yt_mc <- y_true[!binary_idx]
    yp_mc <- y_pred[!binary_idx]
    
    classes <- sort(unique(yt_mc))
    per_class <- sapply(classes, function(cn) {
      tp <- sum(yt_mc == cn & yp_mc == cn)
      fp <- sum(yt_mc != cn & yp_mc == cn)
      fn <- sum(yt_mc == cn & yp_mc != cn)
      # Treat undefined precision/recall as 0 (model failed to predict this class)
      p <- if ((tp + fp) > 0) tp/(tp + fp) else 0
      r <- if ((tp + fn) > 0) tp/(tp + fn) else 0
      if ((p + r) > 0) 2*p*r/(p + r) else 0
    })
    f1_multiclass <- mean(per_class)  # No na.rm needed since all values are now numeric
  } else {
    f1_multiclass <- NA_real_
  }
  
  # Average F1 across binary and multiclass (weighted by count)
  n_binary <- sum(binary_idx)
  n_multiclass <- sum(!binary_idx)
  total_n <- n_binary + n_multiclass
  
  if (total_n == 0) return(NA_real_)
  
  f1_overall <- 0
  if (!is.na(f1_binary)) f1_overall <- f1_overall + (f1_binary * n_binary / total_n)
  if (!is.na(f1_multiclass)) f1_overall <- f1_overall + (f1_multiclass * n_multiclass / total_n)
  
  if (is.na(f1_binary) && is.na(f1_multiclass)) return(NA_real_)
  if (is.na(f1_binary)) return(f1_multiclass)
  if (is.na(f1_multiclass)) return(f1_binary)
  
  return(f1_overall)
}

# Calculate F1 at ritual level (per model-condition combo)
performance_data_ritual <- performance_data %>%
  group_by(ritual_number, model, condition) %>%
  summarise(
    f1_score = calculate_f1(y_true_num, y_pred_num, feature_type),
    n_predictions = n(),
    .groups = "drop"
  ) %>%
  filter(!is.na(f1_score))

message(sprintf("Per-ritual F1 scores calculated: %d observations", nrow(performance_data_ritual)))

# ============================================================================
# 3. MERGE PREDICTORS
# ============================================================================

message("Merging predictors...")

# Add ritual-level predictors to the ritual-level F1 data
performance_data_ritual <- performance_data_ritual %>%
  left_join(
    rituals_meta %>% select(ritual_number, text, Region, Date, character_length),
    by = "ritual_number"
  )

# Clean up Region and Date
performance_data_ritual <- performance_data_ritual %>%
  mutate(
    region = as.character(Region),
    year = suppressWarnings(as.numeric(Date)),
    char_length = as.numeric(character_length)
  ) %>%
  filter(!is.na(char_length) & char_length > 0)

# Also calculate average feature-level characteristics per ritual for the detailed data
# performance_data already has feature_type from earlier merge, so we just need description_length
feature_summary <- performance_data %>%
  left_join(features_meta %>% select(feature_variable, feature_description_length),
            by = "feature_variable") %>%
  group_by(ritual_number) %>%
  summarise(
    avg_feat_desc_length = mean(feature_description_length, na.rm = TRUE),
    prop_binary = mean(feature_type == "binary", na.rm = TRUE),
    n_features = n(),
    .groups = "drop"
  )

performance_data_ritual <- performance_data_ritual %>%
  left_join(feature_summary, by = "ritual_number")

message(sprintf("After merging: %d ritual-level observations", nrow(performance_data_ritual)))

# ============================================================================
# 4. CALCULATE TEXT COMPLEXITY METRICS (Gunning FOG and VOCD-D)
# ============================================================================

message("Calculating text complexity metrics (this may take a while)...")

# Function to calculate text metrics safely
calculate_text_metrics <- function(text) {
  if (is.na(text) || nchar(text) < 10) {
    return(list(fog = NA_real_, mtld = NA_real_))
  }
  
  fog <- tryCatch({
    # Tokenize and calculate readability
    tagged <- koRpus::tokenize(text, lang = "en", format = "obj")
    read_stats <- koRpus::readability(tagged, index = "FOG", quiet = TRUE)
    as.numeric(read_stats@FOG$FOG)
  }, error = function(e) NA_real_)
  
  mtld <- tryCatch({
    # Calculate lexical diversity using MTLD with char=TRUE (more stable)
    tagged <- koRpus::tokenize(text, lang = "en", format = "obj")
    lex_div <- koRpus::lex.div(tagged, measure = "MTLD", char = TRUE, quiet = TRUE)
    as.numeric(lex_div@MTLD$MTLD)
  }, error = function(e) NA_real_)
  
  list(fog = fog, mtld = mtld)
}

# Calculate metrics for unique texts (to avoid redundant computation)
unique_texts <- performance_data_ritual %>%
  distinct(ritual_number, text) %>%
  filter(!is.na(text))

message(sprintf("Computing metrics for %d unique texts...", nrow(unique_texts)))

# Compute metrics with progress
text_metrics <- unique_texts %>%
  mutate(metrics = map(text, calculate_text_metrics)) %>%
  mutate(
    gunning_fog = map_dbl(metrics, "fog"),
    mtld = map_dbl(metrics, "mtld")
  ) %>%
  select(ritual_number, gunning_fog, mtld)

# Merge back
performance_data_ritual <- performance_data_ritual %>%
  left_join(text_metrics, by = "ritual_number")

message("Text metrics calculated.")

# ============================================================================
# 5. Z-SCORE CONTINUOUS PREDICTORS
# =====================, quiet = TRUE=========================================

message("Standardising continuous predictors...")

performance_data_ritual <- performance_data_ritual %>%
  mutate(
    # Log-transform character length first
    log_char_length = log(char_length),
    # Z-score continuous predictors
    z_log_char_length = scale(log_char_length)[,1],
    z_avg_feat_desc_length = scale(avg_feat_desc_length)[,1],
    z_year = scale(year, center = TRUE, scale = TRUE)[,1],
    z_fog = scale(gunning_fog)[,1],
    z_mtld = scale(mtld)[,1]
  ) %>%
  filter(!is.na(z_log_char_length) & !is.na(z_avg_feat_desc_length) & 
         !is.na(z_fog) & !is.na(z_mtld) & !is.na(z_year) &
         !is.na(region) & !is.na(prop_binary) & !is.na(f1_score))

message(sprintf("Final dataset: %d ritual-level observations", nrow(performance_data_ritual)))

# Convert categorical variables to factors and ensure no empty levels
performance_data_ritual <- performance_data_ritual %>%
  mutate(
    model = factor(model, levels = model_bases),
    condition = factor(condition, levels = conditions),
    region = factor(region)
  ) %>%
  # Ensure ritual_number is character first, then factor (avoiding issues with existing factors)
  mutate(
    ritual_number = as.character(ritual_number),
    ritual_number = factor(ritual_number)
  )

# Drop any unused factor levels
performance_data_ritual <- performance_data_ritual %>%
  mutate(across(where(is.factor), droplevels))

# Save prepared dataset
write_csv(performance_data_ritual, file.path(output_dir, "performance_data_prepared.csv"))

# ============================================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================================

message("Generating descriptive statistics...")

desc_stats <- list()

# Overall F1
desc_stats$overall <- performance_data_ritual %>%
  summarise(
    n_observations = n(),
    n_rituals = n_distinct(ritual_number),
    n_models = n_distinct(model),
    n_conditions = n_distinct(condition),
    overall_f1 = mean(f1_score, na.rm = TRUE),
    median_f1 = median(f1_score, na.rm = TRUE),
    sd_f1 = sd(f1_score, na.rm = TRUE)
  )

# By model
desc_stats$by_model <- performance_data_ritual %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean_f1 = mean(f1_score, na.rm = TRUE),
    median_f1 = median(f1_score, na.rm = TRUE),
    sd_f1 = sd(f1_score, na.rm = TRUE),
    .groups = "drop"
  )

# By condition
desc_stats$by_condition <- performance_data_ritual %>%
  group_by(condition) %>%
  summarise(
    n = n(),
    mean_f1 = mean(f1_score, na.rm = TRUE),
    median_f1 = median(f1_score, na.rm = TRUE),
    sd_f1 = sd(f1_score, na.rm = TRUE),
    .groups = "drop"
  )

# By model-condition
desc_stats$by_model_condition <- performance_data_ritual %>%
  group_by(model, condition) %>%
  summarise(
    n = n(),
    mean_f1 = mean(f1_score, na.rm = TRUE),
    median_f1 = median(f1_score, na.rm = TRUE),
    sd_f1 = sd(f1_score, na.rm = TRUE),
    .groups = "drop"
  )

# Continuous predictor summary
desc_stats$continuous_vars <- performance_data_ritual %>%
  summarise(
    char_length_median = median(char_length, na.rm = TRUE),
    char_length_iqr = IQR(char_length, na.rm = TRUE),
    fog_mean = mean(gunning_fog, na.rm = TRUE),
    fog_sd = sd(gunning_fog, na.rm = TRUE),
    mtld_mean = mean(mtld, na.rm = TRUE),
    mtld_sd = sd(mtld, na.rm = TRUE),
    year_median = median(year, na.rm = TRUE),
    year_range = paste(min(year, na.rm = TRUE), "-", max(year, na.rm = TRUE))
  )

# Save descriptive stats
capture.output(desc_stats, file = file.path(output_dir, "descriptive_statistics.txt"))

# ============================================================================
# 7. EXPLORATORY VISUALIZATIONS
# ============================================================================

message("Creating exploratory visualizations...")

# F1 by model and condition
p1 <- ggplot(desc_stats$by_model_condition, aes(x = model, y = mean_f1, fill = condition)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "Mean F1 Score by Model and Condition",
       x = "Model", y = "Mean F1 Score", fill = "Condition") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "f1_by_model_condition.pdf"), p1, width = 10, height = 6)

# F1 distribution
p2 <- ggplot(performance_data_ritual, aes(x = f1_score)) +
  geom_histogram(bins = 50, fill = "#4C72B0", alpha = 0.7) +
  labs(title = "Distribution of Ritual-level F1 Scores",
       x = "F1 Score", y = "Count") +
  theme_bw()

ggsave(file.path(output_dir, "f1_distribution.pdf"), p2, width = 8, height = 5)

# Text length distribution
p3 <- ggplot(performance_data_ritual %>% distinct(ritual_number, char_length), 
             aes(x = char_length)) +
  geom_histogram(bins = 50, fill = "#4C72B0", alpha = 0.7) +
  scale_x_log10(labels = scales::comma) +
  labs(title = "Distribution of Text Length (Characters)",
       x = "Character Length (log scale)", y = "Count") +
  theme_bw()

ggsave(file.path(output_dir, "text_length_distribution.pdf"), p3, width = 8, height = 5)

# Bivariate: F1 vs continuous predictors
# Average across model-condition for cleaner plots
ritual_f1_avg <- performance_data_ritual %>%
  group_by(ritual_number, char_length, gunning_fog, mtld, year) %>%
  summarise(mean_f1 = mean(f1_score, na.rm = TRUE), .groups = "drop")

p4a <- ggplot(ritual_f1_avg, aes(x = char_length, y = mean_f1)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(method = "loess", color = "#C44E52") +
  scale_x_log10(labels = scales::comma) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "Ritual-level F1 vs Text Length",
       x = "Character Length (log scale)", y = "Mean F1 Score") +
  theme_bw()

p4b <- ggplot(ritual_f1_avg, aes(x = gunning_fog, y = mean_f1)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(method = "loess", color = "#C44E52") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "Ritual-level F1 vs Gunning FOG",
       x = "Gunning FOG Index", y = "Mean F1 Score") +
  theme_bw()

p4c <- ggplot(ritual_f1_avg, aes(x = mtld, y = mean_f1)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(method = "loess", color = "#C44E52") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "Ritual-level F1 vs Lexical Diversity (MTLD)",
       x = "MTLD Score", y = "Mean F1 Score") +
  theme_bw()

p4d <- ggplot(ritual_f1_avg, aes(x = year, y = mean_f1)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(method = "loess", color = "#C44E52") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "Ritual-level F1 vs Publication Year",
       x = "Year", y = "Mean F1 Score") +
  theme_bw()

combined_bivariate <- cowplot::plot_grid(p4a, p4b, p4c, p4d, ncol = 2)
ggsave(file.path(output_dir, "bivariate_f1_predictors.pdf"), combined_bivariate, width = 12, height = 10)

# ============================================================================
# 8. NESTED MIXED-EFFECTS MODELS
# ============================================================================

message("Fitting nested mixed-effects models...")
message("This may take several minutes depending on dataset size...")

# Prepare data subset if needed for faster testing (comment out for full analysis)
# performance_data_ritual <- performance_data_ritual %>% sample_frac(0.1)

# M0: Random effects only (baseline)
message("Fitting M0: Random effects only...")
M0 <- lmer(f1_score ~ 1 + (1 | ritual_number),
           data = performance_data_ritual,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 20000)))

# M1: MODEL factors (model + condition + interaction)
message("Fitting M1: MODEL factors...")
M1 <- lmer(f1_score ~ model + condition + model:condition + 
             (1 | ritual_number),
           data = performance_data_ritual,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 20000)))

# M2: MODEL + TEXT factors
message("Fitting M2: MODEL + TEXT factors...")
M2 <- lmer(f1_score ~ model + condition + model:condition +
             z_log_char_length + z_fog + z_mtld + region + z_year +
             (1 | ritual_number),
           data = performance_data_ritual,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 20000)))

# M3: MODEL + TEXT + TASK factors (using aggregated feature characteristics)
message("Fitting M3: MODEL + TEXT + TASK factors...")
M3 <- lmer(f1_score ~ model + condition + model:condition +
             z_log_char_length + z_fog + z_mtld + region + z_year +
             prop_binary + z_avg_feat_desc_length +
             (1 | ritual_number),
           data = performance_data_ritual,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 20000)))

# Save model objects
saveRDS(M0, file.path(output_dir, "model_M0.rds"))
saveRDS(M1, file.path(output_dir, "model_M1.rds"))
saveRDS(M2, file.path(output_dir, "model_M2.rds"))
saveRDS(M3, file.path(output_dir, "model_M3.rds"))

# ============================================================================
# 9. MODEL COMPARISON
# ============================================================================

message("Comparing models...")

# Likelihood ratio tests
model_comparison <- anova(M0, M1, M2, M3)
write.csv(model_comparison, file.path(output_dir, "model_comparison_anova.csv"), row.names = FALSE)

# AIC/BIC comparison
aic_bic <- data.frame(
  Model = c("M0_RandomOnly", "M1_Model", "M2_Model_Text", "M3_Model_Text_Task"),
  AIC = c(AIC(M0), AIC(M1), AIC(M2), AIC(M3)),
  BIC = c(BIC(M0), BIC(M1), BIC(M2), BIC(M3))
)
write.csv(aic_bic, file.path(output_dir, "model_comparison_aic_bic.csv"), row.names = FALSE)

# R-squared (variance explained)
r2_M1 <- r2_nakagawa(M1)
r2_M2 <- r2_nakagawa(M2)
r2_M3 <- r2_nakagawa(M3)

r2_comparison <- data.frame(
  Model = c("M1_Model", "M2_Model_Text", "M3_Model_Text_Task"),
  R2_marginal = c(r2_M1$R2_marginal, r2_M2$R2_marginal, r2_M3$R2_marginal),
  R2_conditional = c(r2_M1$R2_conditional, r2_M2$R2_conditional, r2_M3$R2_conditional)
)
write.csv(r2_comparison, file.path(output_dir, "model_comparison_r2.csv"), row.names = FALSE)

# Variance components
variance_M3 <- as.data.frame(VarCorr(M3))
write.csv(variance_M3, file.path(output_dir, "variance_components_M3.csv"), row.names = FALSE)

# ============================================================================
# 10. MODEL SUMMARIES AND COEFFICIENTS
# ============================================================================

message("Extracting model summaries...")

# Full summary for M3 (final model)
sink(file.path(output_dir, "model_M3_summary.txt"))
print(summary(M3))
sink()

# Extract coefficients with CIs
coefs_M3 <- as.data.frame(summary(M3)$coefficients)
coefs_M3$term <- rownames(coefs_M3)
rownames(coefs_M3) <- NULL

# Add confidence intervals (for lmer, use regular confint on fixed effects)
ci_M3 <- confint(M3, parm = "beta_", method = "Wald")
ci_M3_df <- as.data.frame(ci_M3)
ci_M3_df$term <- rownames(ci_M3_df)
rownames(ci_M3_df) <- NULL
names(ci_M3_df) <- c("CI_lower", "CI_upper", "term")

coefs_M3 <- coefs_M3 %>%
  left_join(ci_M3_df, by = "term")

write.csv(coefs_M3, file.path(output_dir, "coefficients_M3.csv"), row.names = FALSE)

# ============================================================================
# 11. FIGURES
# ============================================================================

message("Creating figures...")

# Figure 1: Model comparison (R² comparison)
fig1_data <- r2_comparison %>%
  pivot_longer(cols = c(R2_marginal, R2_conditional), 
               names_to = "R2_type", values_to = "R2_value") %>%
  mutate(
    Model = factor(Model, levels = c("M1_Model", "M2_Model_Text", "M3_Model_Text_Task")),
    R2_type = factor(R2_type, levels = c("R2_marginal", "R2_conditional"),
                     labels = c("Fixed effects only", "Fixed + Random effects"))
  )

fig1 <- ggplot(fig1_data, aes(x = Model, y = R2_value, fill = R2_type)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  scale_x_discrete(labels = c("MODEL", "MODEL +\nTEXT", "MODEL +\nTEXT +\nTASK")) +
  scale_fill_manual(values = c("#4C72B0", "#C44E52")) +
  labs(title = "Variance Explained by Nested Models",
       subtitle = "Incremental contribution of MODEL, TEXT, and TASK factors",
       x = "Model", y = "R² (Variance Explained)", fill = NULL) +
  theme_bw() +
  theme(legend.position = "bottom",
        axis.text.x = element_text(size = 11))

ggsave(file.path(output_dir, "fig1_model_comparison_r2.pdf"), fig1, width = 8, height = 6)

# Figure 2: Coefficient plot (forest plot) for M3
# Focus on key predictors (exclude interaction terms for clarity)
coefs_plot <- coefs_M3 %>%
  filter(!grepl(":", term)) %>%  # Exclude interactions
  filter(term != "(Intercept)") %>%
  mutate(
    term_clean = case_when(
      grepl("^model", term) ~ gsub("model", "Model: ", term),
      grepl("^condition", term) ~ gsub("condition", "Condition: ", term),
      grepl("^region", term) ~ gsub("region", "Region: ", term),
      term == "z_log_char_length" ~ "Text Length (log)",
      term == "z_fog" ~ "Gunning FOG",
      term == "z_mtld" ~ "Lexical Diversity (MTLD)",
      term == "z_year" ~ "Publication Year",
      term == "z_avg_feat_desc_length" ~ "Avg Feature Description Length",
      term == "prop_binary" ~ "Proportion Binary Features",
      TRUE ~ term
    ),
    significant = `Pr(>|t|)` < 0.05
  ) %>%
  arrange(Estimate)

fig2 <- ggplot(coefs_plot, aes(x = Estimate, y = reorder(term_clean, Estimate), 
                                color = significant)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper), height = 0.3, linewidth = 0.8) +
  geom_point(size = 3) +
  scale_color_manual(values = c("FALSE" = "gray60", "TRUE" = "#C44E52")) +
  labs(title = "Fixed Effect Coefficients (Model M3)",
       subtitle = "Effect on F1 score with 95% confidence intervals",
       x = "Coefficient (change in F1 score)", y = NULL) +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.y = element_text(size = 9))

ggsave(file.path(output_dir, "fig2_coefficient_plot_M3.pdf"), fig2, width = 10, height = 12)

# Figure 3: Variance decomposition (pie chart)
variance_decomp <- variance_M3 %>%
  filter(grp %in% c("ritual_number", "Residual")) %>%
  mutate(
    source = case_when(
      grp == "ritual_number" ~ "Between-ritual variation",
      grp == "Residual" ~ "Within-ritual variation (residual)"
    ),
    variance = vcov
  )

total_var <- sum(variance_decomp$variance)
variance_decomp <- variance_decomp %>%
  mutate(
    proportion = variance / total_var,
    percentage = scales::percent(proportion, accuracy = 0.1)
  )

fig3 <- ggplot(variance_decomp, aes(x = "", y = proportion, fill = source)) +
  geom_col(width = 1) +
  coord_polar("y") +
  scale_fill_manual(values = c("#4C72B0", "#C44E52")) +
  geom_text(aes(label = percentage), position = position_stack(vjust = 0.5), 
            color = "white", fontface = "bold", size = 5) +
  labs(title = "Variance Decomposition (Model M3)",
       fill = "Source of Variation") +
  theme_void() +
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "fig3_variance_decomposition.pdf"), fig3, width = 7, height = 7)

# Figure 4: Model comparison barplot (AIC)
fig4_data <- aic_bic %>%
  mutate(Model = factor(Model, levels = c("M0_RandomOnly", "M1_Model", "M2_Model_Text", "M3_Model_Text_Task")))

fig4 <- ggplot(fig4_data, aes(x = Model, y = AIC)) +
  geom_col(fill = "#4C72B0", width = 0.6) +
  scale_x_discrete(labels = c("M0:\nRandom\nonly", "M1:\nMODEL", 
                               "M2:\nMODEL +\nTEXT", "M3:\nMODEL +\nTEXT +\nTASK")) +
  labs(title = "Model Comparison: AIC",
       subtitle = "Lower AIC indicates better model fit",
       x = "Model", y = "AIC") +
  theme_bw()

ggsave(file.path(output_dir, "fig4_model_comparison_aic.pdf"), fig4, width = 8, height = 6)

message("Figures saved.")

# ============================================================================
# 12. SUMMARY REPORT
# ============================================================================

message("Generating summary report...")

sink(file.path(output_dir, "SUMMARY_REPORT.txt"))

cat("DATASET OVERVIEW\n")
cat(sprintf("Total observations (ritual-model-condition): %d\n", desc_stats$overall$n_observations))
cat(sprintf("Unique rituals: %d\n", desc_stats$overall$n_rituals))
cat(sprintf("Models evaluated: %d\n", desc_stats$overall$n_models))
cat(sprintf("Conditions evaluated: %d\n", desc_stats$overall$n_conditions))
cat(sprintf("Overall mean F1: %.3f\n", desc_stats$overall$overall_f1))
cat(sprintf("Overall median F1: %.3f\n\n", desc_stats$overall$median_f1))

cat("DESCRIPTIVE STATISTICS\n")
cat("F1 Score by Model:\n")
print(desc_stats$by_model)
cat("\nF1 Score by Condition:\n")
print(desc_stats$by_condition)
cat("\n")

cat("MODEL COMPARISON\n")
cat("Nested Model Comparison (Likelihood Ratio Tests):\n")
print(model_comparison)
cat("\n")

cat("AIC/BIC Comparison:\n")
print(aic_bic)
cat("\n")

cat("Variance Explained (R²):\n")
print(r2_comparison)
cat("\n")

cat("VARIANCE DECOMPOSITION (Model M3)\n")
cat("---------------------------------------------------------------------------\n")
print(variance_decomp %>% select(source, variance, percentage))
cat("\n")

cat("KEY FINDINGS\n")
cat("---------------------------------------------------------------------------\n")
cat(sprintf("MODEL factors explain: %.1f%% of variance (marginal R²)\n", 
            r2_M1$R2_marginal * 100))
cat(sprintf("Adding TEXT factors increases to: %.1f%% (gain: %.1f%%)\n", 
            r2_M2$R2_marginal * 100, 
            (r2_M2$R2_marginal - r2_M1$R2_marginal) * 100))
cat(sprintf("Adding TASK factors increases to: %.1f%% (gain: %.1f%%)\n", 
            r2_M3$R2_marginal * 100, 
            (r2_M3$R2_marginal - r2_M2$R2_marginal) * 100))
cat(sprintf("\nTotal variance explained by fixed + random effects: %.1f%%\n", 
            r2_M3$R2_conditional * 100))

cat("Analysis complete. All outputs saved to:", output_dir, "\n")

sink()
