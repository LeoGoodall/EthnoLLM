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

results_dir <- "all"
features_csv <- "data/features_all.csv"
rituals_csv <- "data/rituals_codes.csv"
exclude_csv <- "data/exclude.csv"
output_dir <- "performance_diagnostics_feature"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Model and condition setup
model_bases <- c("gptoss120b", "deepseekv31671b", "gpt5nano", "llama33b", "claudesonnet45", "qwen3")
conditions <- c("baseline", "mtp", "ensemble_mtp")

excluded_features <- c("ParticipantPeakDysphoria", "IndividualExegesis", "DissolveUnion", "Disgust")


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
message("Loading and reshaping results files to prediction level...")

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
prediction_data <- bind_rows(all_results)

message(sprintf("Total predictions loaded: %d", nrow(prediction_data)))


# 2. CALCULATE FEATURE BASE RATES
message("Calculating feature base rates...")

# For each feature, calculate base rate from ground truth across all rituals
feature_base_rates <- rituals_meta %>%
  select(ritual_number, all_of(features_meta$feature_variable)) %>%
  # Convert all feature columns to character first to avoid type conflicts in pivot_longer
  mutate(across(-ritual_number, as.character)) %>%
  pivot_longer(cols = -ritual_number, names_to = "feature_variable", values_to = "y_true") %>%
  mutate(
    y_true_num = suppressWarnings(as.numeric(y_true)),
    # Filter out missing and -1 values
    skip = is.na(y_true_num) | y_true_num == -1 | y_true_num == 999
  ) %>%
  filter(!skip) %>%
  # Join feature type
  left_join(features_meta %>% select(feature_variable, feature_type), by = "feature_variable") %>%
  # For binary: convert to 0/1; for multiclass: convert to presence (1) vs absence (0)
  mutate(
    is_present = case_when(
      feature_type == "binary" ~ ifelse(y_true_num != 0, 1, 0),
      feature_type == "multiclass" ~ ifelse(y_true_num != 0, 1, 0),  # Treat any non-zero as "present"
      TRUE ~ NA_real_
    )
  ) %>%
  filter(!is.na(is_present)) %>%
  group_by(feature_variable) %>%
  summarise(
    base_rate = mean(is_present, na.rm = TRUE),
    n_observations = n(),
    .groups = "drop"
  )

message("Feature base rates calculated.")

# 3. PREPARE PREDICTION-LEVEL DATA WITH OUTCOME
message("Preparing prediction-level dataset...")

# Convert to numeric and filter valid comparisons
prediction_data <- prediction_data %>%
  mutate(
    y_true_num = suppressWarnings(as.numeric(y_true)),
    y_pred_num = suppressWarnings(as.numeric(y_pred)),
    # Skip comparisons where ground truth is -1 or 999 or NA or pred is NA
    skip_comparison = is.na(y_true_num) | is.na(y_pred_num) | 
                      y_true_num == -1 | y_true_num == 999
  ) %>%
  filter(!skip_comparison) %>%
  select(-skip_comparison, -y_true, -y_pred)

message(sprintf("Valid predictions after filtering: %d", nrow(prediction_data)))

# Merge feature metadata
prediction_data <- prediction_data %>%
  left_join(
    features_meta %>% select(feature_variable, feature_type, feature_description_length),
    by = "feature_variable"
  ) %>%
  left_join(feature_base_rates %>% select(feature_variable, base_rate), by = "feature_variable")

# Calculate binary outcome: correct (1) vs incorrect (0)
# For binary features: y_true and y_pred are converted to 0/1
# For multiclass features: exact match
prediction_data <- prediction_data %>%
  mutate(
    # For binary: convert to 0/1
    y_true_binary = case_when(
      feature_type == "binary" ~ ifelse(y_true_num != 0, 1, 0),
      feature_type == "multiclass" ~ y_true_num,
      TRUE ~ NA_real_
    ),
    y_pred_binary = case_when(
      feature_type == "binary" ~ ifelse(y_pred_num != 0, 1, 0),
      feature_type == "multiclass" ~ y_pred_num,
      TRUE ~ NA_real_
    ),
    # Outcome: correct (1) or incorrect (0)
    correct = ifelse(y_true_binary == y_pred_binary, 1, 0)
  )

# Add ritual-level predictors
prediction_data <- prediction_data %>%
  left_join(
    rituals_meta %>% select(ritual_number, text, Region, Date, character_length),
    by = "ritual_number"
  ) %>%
  mutate(
    region = as.character(Region),
    year = suppressWarnings(as.numeric(Date)),
    char_length = as.numeric(character_length)
  ) %>%
  filter(!is.na(char_length) & char_length > 0)

message(sprintf("After merging predictors: %d predictions", nrow(prediction_data)))


# 4. CALCULATE TEXT COMPLEXITY METRICS
message("Calculating text complexity metrics (this may take a while)...")

# Function to calculate text metrics safely
calculate_text_metrics <- function(text) {
  if (is.na(text) || nchar(text) < 10) {
    return(list(fog = NA_real_, mtld = NA_real_))
  }
  
  fog <- tryCatch({
    suppressMessages(suppressWarnings({
      tagged <- koRpus::tokenize(text, lang = "en", format = "obj")
      read_stats <- koRpus::readability(tagged, index = "FOG", quiet = TRUE)
      as.numeric(read_stats@FOG$FOG)
    }))
  }, error = function(e) NA_real_)
  
  mtld <- tryCatch({
    suppressMessages(suppressWarnings({
      tagged <- koRpus::tokenize(text, lang = "en", format = "obj")
      lex_div <- koRpus::lex.div(tagged, measure = "MTLD", char = TRUE, quiet = TRUE)
      as.numeric(lex_div@MTLD$MTLD)
    }))
  }, error = function(e) NA_real_)
  
  list(fog = fog, mtld = mtld)
}

# Calculate metrics for unique texts (to avoid redundant computation)
unique_texts <- prediction_data %>%
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
prediction_data <- prediction_data %>%
  left_join(text_metrics, by = "ritual_number")

message("Text metrics calculated.")

# 5. Z-SCORE CONTINUOUS PREDICTORS
message("Standardizing continuous predictors...")

prediction_data <- prediction_data %>%
  mutate(
    # Log-transform character length first
    log_char_length = log(char_length),
    # Z-score continuous predictors
    z_log_char_length = scale(log_char_length)[,1],
    z_feature_desc_length = scale(feature_description_length)[,1],
    z_feature_base_rate = scale(base_rate)[,1],
    z_year = scale(year, center = TRUE, scale = TRUE)[,1],
    z_fog = scale(gunning_fog)[,1],
    z_mtld = scale(mtld)[,1]
  ) %>%
  filter(!is.na(z_log_char_length) & !is.na(z_feature_desc_length) &
         !is.na(z_feature_base_rate) & !is.na(z_fog) & !is.na(z_mtld) & 
         !is.na(z_year) & !is.na(region) & !is.na(feature_type) & !is.na(correct))

message(sprintf("Final dataset: %d predictions", nrow(prediction_data)))

# Convert categorical variables to factors
prediction_data <- prediction_data %>%
  mutate(
    model = factor(model, levels = model_bases),
    condition = factor(condition, levels = conditions),
    region = factor(region),
    feature_type = factor(feature_type, levels = c("binary", "multiclass")),
    ritual_number = factor(as.character(ritual_number)),
    feature_variable = factor(feature_variable)
  ) %>%
  mutate(across(where(is.factor), droplevels))

# Save prepared dataset
write_csv(prediction_data, file.path(output_dir, "prediction_data_prepared.csv"))
message("Saved prepared prediction-level dataset.")

# 6. STRATIFY DATA: POSITIVE vs NEGATIVE CASES
message("Stratifying data by true class...")

# Positive cases: y_true = 1 (for binary) or any specific class (for multiclass)
# For simplicity, treat as: "feature is present/expressed" vs "feature is absent"
positive_data <- prediction_data %>%
  filter(y_true_binary != 0)  # Present/expressed

negative_data <- prediction_data %>%
  filter(y_true_binary == 0)  # Absent

message(sprintf("Positive cases (true positives/negatives): %d", nrow(positive_data)))
message(sprintf("Negative cases (true negatives/positives): %d", nrow(negative_data)))
message(sprintf("Ratio: %.2f%% positive", 100 * nrow(positive_data) / nrow(prediction_data)))

# 7. DESCRIPTIVE STATISTICS
message("Generating descriptive statistics...")

desc_stats <- list()

# Overall statistics
desc_stats$overall <- prediction_data %>%
  summarise(
    n_predictions = n(),
    n_rituals = n_distinct(ritual_number),
    n_features = n_distinct(feature_variable),
    n_models = n_distinct(model),
    n_conditions = n_distinct(condition),
    overall_accuracy = mean(correct, na.rm = TRUE),
    prop_positive = mean(y_true_binary != 0, na.rm = TRUE)
  )

# By model
desc_stats$by_model <- prediction_data %>%
  group_by(model) %>%
  summarise(
    n = n(),
    accuracy = mean(correct, na.rm = TRUE),
    .groups = "drop"
  )

# By condition
desc_stats$by_condition <- prediction_data %>%
  group_by(condition) %>%
  summarise(
    n = n(),
    accuracy = mean(correct, na.rm = TRUE),
    .groups = "drop"
  )

# By feature type
desc_stats$by_feature_type <- prediction_data %>%
  group_by(feature_type) %>%
  summarise(
    n = n(),
    accuracy = mean(correct, na.rm = TRUE),
    prop_positive = mean(y_true_binary != 0),
    .groups = "drop"
  )

# By stratum
desc_stats$by_stratum <- bind_rows(
  positive_data %>% summarise(stratum = "Positive", n = n(), accuracy = mean(correct)),
  negative_data %>% summarise(stratum = "Negative", n = n(), accuracy = mean(correct))
)

# Feature-level base rates
desc_stats$feature_base_rates <- feature_base_rates %>%
  arrange(base_rate) %>%
  mutate(
    rarity = case_when(
      base_rate < 0.05 ~ "Very rare (<5%)",
      base_rate < 0.20 ~ "Rare (5-20%)",
      base_rate < 0.50 ~ "Moderate (20-50%)",
      TRUE ~ "Common (>50%)"
    )
  )

# Save descriptive stats
capture.output(desc_stats, file = file.path(output_dir, "descriptive_statistics_stratified.txt"))
message("Descriptive statistics saved.")



# 8. FIT NESTED MODELS: POSITIVE STRATUM (Detection)
message("\n==== FITTING MODELS FOR POSITIVE STRATUM (Detection) ====")
message("This may take 15-30 minutes depending on dataset size...")

# Check for existing models (checkpointing)
M0_pos_file <- file.path(output_dir, "model_M0_positive.rds")
M1_pos_file <- file.path(output_dir, "model_M1_positive.rds")
M2_pos_file <- file.path(output_dir, "model_M2_positive.rds")
M3_pos_file <- file.path(output_dir, "model_M3_positive.rds")

if (file.exists(M3_pos_file)) {
  message("Loading existing positive stratum models from checkpoint...")
  M0_pos <- readRDS(M0_pos_file)
  M1_pos <- readRDS(M1_pos_file)
  M2_pos <- readRDS(M2_pos_file)
  M3_pos <- readRDS(M3_pos_file)
  message("Positive stratum models loaded from checkpoint.")
} else {
  # M0: Random effects only
  message("Fitting M0_pos: Random effects only...")
  M0_pos <- glmer(correct ~ 1 + (1 | ritual_number) + (1 | feature_variable),
                  data = positive_data,
                  family = binomial,
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  
  # M1: MODEL factors
  message("Fitting M1_pos: MODEL factors...")
  M1_pos <- glmer(correct ~ model + condition + model:condition + 
                    (1 | ritual_number) + (1 | feature_variable),
                  data = positive_data,
                  family = binomial,
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  
  # M2: MODEL + TEXT factors
  message("Fitting M2_pos: MODEL + TEXT factors...")
  M2_pos <- glmer(correct ~ model + condition + model:condition +
                    z_log_char_length + z_fog + z_mtld + region + z_year +
                    (1 | ritual_number) + (1 | feature_variable),
                  data = positive_data,
                  family = binomial,
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  
  # M3: MODEL + TEXT + TASK factors
  message("Fitting M3_pos: MODEL + TEXT + TASK factors...")
  M3_pos <- glmer(correct ~ model + condition + model:condition +
                    z_log_char_length + z_fog + z_mtld + region + z_year +
                    feature_type + z_feature_desc_length + z_feature_base_rate +
                    (1 | ritual_number) + (1 | feature_variable),
                  data = positive_data,
                  family = binomial,
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  
  message("Positive stratum models fitted successfully.")
  
  # Save model objects
  saveRDS(M0_pos, M0_pos_file)
  saveRDS(M1_pos, M1_pos_file)
  saveRDS(M2_pos, M2_pos_file)
  saveRDS(M3_pos, M3_pos_file)
}

# 9. FIT NESTED MODELS: NEGATIVE STRATUM (Specificity)
message("\n==== FITTING MODELS FOR NEGATIVE STRATUM (Specificity) ====")
message("This may take 30-60 minutes on HPC depending on dataset size...")

# Check for existing models (checkpointing)
M0_neg_file <- file.path(output_dir, "model_M0_negative.rds")
M1_neg_file <- file.path(output_dir, "model_M1_negative.rds")
M2_neg_file <- file.path(output_dir, "model_M2_negative.rds")
M3_neg_file <- file.path(output_dir, "model_M3_negative.rds")

if (file.exists(M3_neg_file)) {
  message("Loading existing negative stratum models from checkpoint...")
  M0_neg <- readRDS(M0_neg_file)
  M1_neg <- readRDS(M1_neg_file)
  M2_neg <- readRDS(M2_neg_file)
  M3_neg <- readRDS(M3_neg_file)
  message("Negative stratum models loaded from checkpoint.")
} else {
  # M0: Random effects only
  message("Fitting M0_neg: Random effects only...")
  M0_neg <- glmer(correct ~ 1 + (1 | ritual_number) + (1 | feature_variable),
                  data = negative_data,
                  family = binomial,
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  saveRDS(M0_neg, M0_neg_file)  # Save immediately after fitting
  
  # M1: MODEL factors
  message("Fitting M1_neg: MODEL factors...")
  M1_neg <- glmer(correct ~ model + condition + model:condition + 
                    (1 | ritual_number) + (1 | feature_variable),
                  data = negative_data,
                  family = binomial,
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  saveRDS(M1_neg, M1_neg_file)  # Save immediately after fitting
  
  # M2: MODEL + TEXT factors
  message("Fitting M2_neg: MODEL + TEXT factors...")
  M2_neg <- glmer(correct ~ model + condition + model:condition +
                    z_log_char_length + z_fog + z_mtld + region + z_year +
                    (1 | ritual_number) + (1 | feature_variable),
                  data = negative_data,
                  family = binomial,
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  saveRDS(M2_neg, M2_neg_file)  # Save immediately after fitting
  
  # M3: MODEL + TEXT + TASK factors
  message("Fitting M3_neg: MODEL + TEXT + TASK factors...")
  M3_neg <- glmer(correct ~ model + condition + model:condition +
                    z_log_char_length + z_fog + z_mtld + region + z_year +
                    feature_type + z_feature_desc_length + z_feature_base_rate +
                    (1 | ritual_number) + (1 | feature_variable),
                  data = negative_data,
                  family = binomial,
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)))
  saveRDS(M3_neg, M3_neg_file)  # Save immediately after fitting
  
  message("Negative stratum models fitted successfully.")
}


# 10. MODEL COMPARISON: POSITIVE STRATUM
message("\n==== COMPARING POSITIVE STRATUM MODELS ====")

# Likelihood ratio tests
model_comparison_pos <- anova(M0_pos, M1_pos, M2_pos, M3_pos)
write.csv(model_comparison_pos, file.path(output_dir, "model_comparison_anova_positive.csv"), row.names = FALSE)

# AIC/BIC comparison
aic_bic_pos <- data.frame(
  Model = c("M0_RandomOnly", "M1_Model", "M2_Model_Text", "M3_Model_Text_Task"),
  AIC = c(AIC(M0_pos), AIC(M1_pos), AIC(M2_pos), AIC(M3_pos)),
  BIC = c(BIC(M0_pos), BIC(M1_pos), BIC(M2_pos), BIC(M3_pos)),
  Stratum = "Positive"
)
write.csv(aic_bic_pos, file.path(output_dir, "model_comparison_aic_bic_positive.csv"), row.names = FALSE)

# R-squared (variance explained)
r2_M1_pos <- r2_nakagawa(M1_pos)
r2_M2_pos <- r2_nakagawa(M2_pos)
r2_M3_pos <- r2_nakagawa(M3_pos)

r2_comparison_pos <- data.frame(
  Model = c("M1_Model", "M2_Model_Text", "M3_Model_Text_Task"),
  R2_marginal = c(r2_M1_pos$R2_marginal, r2_M2_pos$R2_marginal, r2_M3_pos$R2_marginal),
  R2_conditional = c(r2_M1_pos$R2_conditional, r2_M2_pos$R2_conditional, r2_M3_pos$R2_conditional),
  Stratum = "Positive"
)
write.csv(r2_comparison_pos, file.path(output_dir, "model_comparison_r2_positive.csv"), row.names = FALSE)

# Variance components
variance_M3_pos <- as.data.frame(VarCorr(M3_pos))
write.csv(variance_M3_pos, file.path(output_dir, "variance_components_M3_positive.csv"), row.names = FALSE)


# 11. MODEL COMPARISON: NEGATIVE STRATUM
message("\n==== COMPARING NEGATIVE STRATUM MODELS ====")

# Likelihood ratio tests
model_comparison_neg <- anova(M0_neg, M1_neg, M2_neg, M3_neg)
write.csv(model_comparison_neg, file.path(output_dir, "model_comparison_anova_negative.csv"), row.names = FALSE)

# AIC/BIC comparison
aic_bic_neg <- data.frame(
  Model = c("M0_RandomOnly", "M1_Model", "M2_Model_Text", "M3_Model_Text_Task"),
  AIC = c(AIC(M0_neg), AIC(M1_neg), AIC(M2_neg), AIC(M3_neg)),
  BIC = c(BIC(M0_neg), BIC(M1_neg), BIC(M2_neg), BIC(M3_neg)),
  Stratum = "Negative"
)
write.csv(aic_bic_neg, file.path(output_dir, "model_comparison_aic_bic_negative.csv"), row.names = FALSE)

# R-squared (variance explained)
r2_M1_neg <- r2_nakagawa(M1_neg)
r2_M2_neg <- r2_nakagawa(M2_neg)
r2_M3_neg <- r2_nakagawa(M3_neg)

r2_comparison_neg <- data.frame(
  Model = c("M1_Model", "M2_Model_Text", "M3_Model_Text_Task"),
  R2_marginal = c(r2_M1_neg$R2_marginal, r2_M2_neg$R2_marginal, r2_M3_neg$R2_marginal),
  R2_conditional = c(r2_M1_neg$R2_conditional, r2_M2_neg$R2_conditional, r2_M3_neg$R2_conditional),
  Stratum = "Negative"
)
write.csv(r2_comparison_neg, file.path(output_dir, "model_comparison_r2_negative.csv"), row.names = FALSE)

# Variance components
variance_M3_neg <- as.data.frame(VarCorr(M3_neg))
write.csv(variance_M3_neg, file.path(output_dir, "variance_components_M3_negative.csv"), row.names = FALSE)


# 12. EXTRACT COEFFICIENTS WITH CIs
message("Extracting model coefficients...")

# Positive stratum
coefs_M3_pos <- as.data.frame(summary(M3_pos)$coefficients)
coefs_M3_pos$term <- rownames(coefs_M3_pos)
rownames(coefs_M3_pos) <- NULL
coefs_M3_pos$Stratum <- "Positive"

# Add odds ratios
coefs_M3_pos$OR <- exp(coefs_M3_pos$Estimate)

# Confidence intervals
ci_M3_pos <- confint(M3_pos, parm = "beta_", method = "Wald")
ci_M3_pos_df <- as.data.frame(ci_M3_pos)
ci_M3_pos_df$term <- rownames(ci_M3_pos_df)
rownames(ci_M3_pos_df) <- NULL
names(ci_M3_pos_df) <- c("CI_lower", "CI_upper", "term")
ci_M3_pos_df$OR_lower <- exp(ci_M3_pos_df$CI_lower)
ci_M3_pos_df$OR_upper <- exp(ci_M3_pos_df$CI_upper)

coefs_M3_pos <- coefs_M3_pos %>% left_join(ci_M3_pos_df, by = "term")
write.csv(coefs_M3_pos, file.path(output_dir, "coefficients_M3_positive.csv"), row.names = FALSE)

# Negative stratum
coefs_M3_neg <- as.data.frame(summary(M3_neg)$coefficients)
coefs_M3_neg$term <- rownames(coefs_M3_neg)
rownames(coefs_M3_neg) <- NULL
coefs_M3_neg$Stratum <- "Negative"

# Add odds ratios
coefs_M3_neg$OR <- exp(coefs_M3_neg$Estimate)

# Confidence intervals
ci_M3_neg <- confint(M3_neg, parm = "beta_", method = "Wald")
ci_M3_neg_df <- as.data.frame(ci_M3_neg)
ci_M3_neg_df$term <- rownames(ci_M3_neg_df)
rownames(ci_M3_neg_df) <- NULL
names(ci_M3_neg_df) <- c("CI_lower", "CI_upper", "term")
ci_M3_neg_df$OR_lower <- exp(ci_M3_neg_df$CI_lower)
ci_M3_neg_df$OR_upper <- exp(ci_M3_neg_df$CI_upper)

coefs_M3_neg <- coefs_M3_neg %>% left_join(ci_M3_neg_df, by = "term")
write.csv(coefs_M3_neg, file.path(output_dir, "coefficients_M3_negative.csv"), row.names = FALSE)

# Combined coefficients for comparison
coefs_combined <- bind_rows(coefs_M3_pos, coefs_M3_neg)
write.csv(coefs_combined, file.path(output_dir, "coefficients_M3_combined.csv"), row.names = FALSE)


# 13. FIGURES
# Figure 1: Model comparison (R² comparison) - Both strata
r2_comparison_combined <- bind_rows(r2_comparison_pos, r2_comparison_neg)

fig1_data <- r2_comparison_combined %>%
  pivot_longer(cols = c(R2_marginal, R2_conditional), 
               names_to = "R2_type", values_to = "R2_value") %>%
  mutate(
    Model = factor(Model, levels = c("M1_Model", "M2_Model_Text", "M3_Model_Text_Task")),
    R2_type = factor(R2_type, levels = c("R2_marginal", "R2_conditional"),
                     labels = c("Fixed effects only", "Fixed + Random effects")),
    Stratum = factor(Stratum, levels = c("Positive", "Negative"))
  )

fig1 <- ggplot(fig1_data, aes(x = Model, y = R2_value, fill = R2_type)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ Stratum, ncol = 2) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  scale_x_discrete(labels = c("MODEL", "MODEL +\nTEXT", "MODEL +\nTEXT +\nTASK")) +
  scale_fill_manual(values = c("#4C72B0", "#C44E52")) +
  labs(title = "Variance Explained by Nested Models (Stratified Analysis)",
       subtitle = "Positive = Detection of present features | Negative = Correct rejection of absent features",
       x = "Model", y = "R² (Variance Explained)", fill = NULL) +
  theme_bw() +
  theme(legend.position = "bottom",
        axis.text.x = element_text(size = 9))

ggsave(file.path(output_dir, "fig1_model_comparison_r2_stratified.pdf"), fig1, width = 10, height = 6)

# Figure 2: Coefficient comparison (forest plot) - Key predictors only, both strata
coefs_plot <- coefs_combined %>%
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
      term == "z_feature_desc_length" ~ "Feature Description Length",
      term == "z_feature_base_rate" ~ "Feature Base Rate",
      term == "feature_typemulticlass" ~ "Feature Type: Multiclass",
      TRUE ~ term
    ),
    significant = `Pr(>|z|)` < 0.05,
    Stratum = factor(Stratum, levels = c("Positive", "Negative"))
  )

fig2 <- ggplot(coefs_plot, aes(x = Estimate, y = reorder(term_clean, Estimate), 
                                color = Stratum, shape = significant)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper), height = 0.3, linewidth = 0.6, 
                 position = position_dodge(width = 0.5)) +
  geom_point(size = 2.5, position = position_dodge(width = 0.5)) +
  scale_color_manual(values = c("Positive" = "#C44E52", "Negative" = "#4C72B0")) +
  scale_shape_manual(values = c("FALSE" = 1, "TRUE" = 16)) +
  labs(title = "Fixed Effect Coefficients (Model M3) - Stratified Analysis",
       subtitle = "Log-odds (coefficients) with 95% confidence intervals",
       x = "Coefficient (log-odds)", y = NULL,
       color = "Stratum", shape = "p < 0.05") +
  theme_bw() +
  theme(legend.position = "bottom",
        axis.text.y = element_text(size = 8))

ggsave(file.path(output_dir, "fig2_coefficient_comparison_stratified.pdf"), fig2, width = 10, height = 12)

# Figure 3: Odds Ratios for key TASK predictors (base_rate, desc_length, feature_type)
task_predictors <- c("z_feature_base_rate", "z_feature_desc_length", "feature_typemulticlass")

coefs_task <- coefs_combined %>%
  filter(term %in% task_predictors) %>%
  mutate(
    term_clean = case_when(
      term == "z_feature_base_rate" ~ "Feature Base Rate",
      term == "z_feature_desc_length" ~ "Feature Description Length",
      term == "feature_typemulticlass" ~ "Multiclass (vs Binary)",
      TRUE ~ term
    ),
    significant = `Pr(>|z|)` < 0.05,
    Stratum = factor(Stratum, levels = c("Positive", "Negative"))
  )

fig3 <- ggplot(coefs_task, aes(x = OR, y = term_clean, color = Stratum, shape = significant)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_errorbarh(aes(xmin = OR_lower, xmax = OR_upper), height = 0.2, linewidth = 0.8,
                 position = position_dodge(width = 0.6)) +
  geom_point(size = 3.5, position = position_dodge(width = 0.6)) +
  scale_color_manual(values = c("Positive" = "#C44E52", "Negative" = "#4C72B0")) +
  scale_shape_manual(values = c("FALSE" = 1, "TRUE" = 16)) +
  scale_x_continuous(trans = "log10") +
  labs(title = "Task Characteristics: Odds Ratios (Model M3)",
       subtitle = "OR > 1: Increases odds of correct prediction | OR < 1: Decreases odds",
       x = "Odds Ratio (log scale)", y = NULL,
       color = "Stratum", shape = "p < 0.05") +
  theme_bw() +
  theme(legend.position = "bottom",
        axis.text.y = element_text(size = 11, face = "bold"))

ggsave(file.path(output_dir, "fig3_task_predictors_OR.pdf"), fig3, width = 9, height = 5)

# Figure 4: Variance decomposition - Both strata
variance_decomp_pos <- variance_M3_pos %>%
  filter(grp %in% c("ritual_number", "feature_variable", "Residual")) %>%
  mutate(
    source = case_when(
      grp == "ritual_number" ~ "Between-ritual variation",
      grp == "feature_variable" ~ "Between-feature variation",
      grp == "Residual" ~ "Residual"
    ),
    Stratum = "Positive"
  )

variance_decomp_neg <- variance_M3_neg %>%
  filter(grp %in% c("ritual_number", "feature_variable", "Residual")) %>%
  mutate(
    source = case_when(
      grp == "ritual_number" ~ "Between-ritual variation",
      grp == "feature_variable" ~ "Between-feature variation",
      grp == "Residual" ~ "Residual"
    ),
    Stratum = "Negative"
  )

variance_decomp_combined <- bind_rows(variance_decomp_pos, variance_decomp_neg) %>%
  group_by(Stratum) %>%
  mutate(
    proportion = vcov / sum(vcov),
    percentage = scales::percent(proportion, accuracy = 0.1)
  ) %>%
  ungroup()

fig4 <- ggplot(variance_decomp_combined, aes(x = Stratum, y = proportion, fill = source)) +
  geom_col(position = "stack", width = 0.6) +
  geom_text(aes(label = percentage), position = position_stack(vjust = 0.5), 
            color = "white", fontface = "bold", size = 4) +
  scale_fill_manual(values = c("#4C72B0", "#55A868", "#C44E52")) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Variance Decomposition (Model M3) - Stratified Analysis",
       subtitle = "Proportion of variance explained by random effects",
       x = "Stratum", y = "Proportion of Variance", fill = "Source") +
  theme_bw() +
  theme(legend.position = "right")

ggsave(file.path(output_dir, "fig4_variance_decomposition_stratified.pdf"), fig4, width = 8, height = 6)

message("Figures saved.")

# 14. SUMMARY REPORT
message("Generating summary report...")

sink(file.path(output_dir, "SUMMARY_REPORT_STRATIFIED.txt"))

cat("PERFORMANCE DIAGNOSTICS: STRATIFIED PREDICTION-LEVEL ANALYSIS\n")

cat("DATASET OVERVIEW\n")
cat(sprintf("Total predictions: %d\n", desc_stats$overall$n_predictions))
cat(sprintf("Unique rituals: %d\n", desc_stats$overall$n_rituals))
cat(sprintf("Unique features: %d\n", desc_stats$overall$n_features))
cat(sprintf("Models evaluated: %d\n", desc_stats$overall$n_models))
cat(sprintf("Conditions evaluated: %d\n", desc_stats$overall$n_conditions))
cat(sprintf("Overall accuracy: %.3f\n", desc_stats$overall$overall_accuracy))
cat(sprintf("Proportion of positive cases: %.3f\n\n", desc_stats$overall$prop_positive))

cat("STRATIFICATION\n")
print(desc_stats$by_stratum)
cat("\n")

cat("FEATURE BASE RATE DISTRIBUTION\n")
table_rarity <- table(desc_stats$feature_base_rates$rarity)
print(table_rarity)
cat("\n")

cat("MODEL COMPARISON: POSITIVE STRATUM (Detection)\n")
cat("Variance Explained (R²):\n")
print(r2_comparison_pos)
cat("\n")

cat(sprintf("MODEL factors explain: %.1f%% of variance (marginal R²)\n", 
            r2_M1_pos$R2_marginal * 100))
cat(sprintf("Adding TEXT factors increases to: %.1f%% (gain: %.1f%%)\n", 
            r2_M2_pos$R2_marginal * 100, 
            (r2_M2_pos$R2_marginal - r2_M1_pos$R2_marginal) * 100))
cat(sprintf("Adding TASK factors increases to: %.1f%% (gain: %.1f%%)\n", 
            r2_M3_pos$R2_marginal * 100, 
            (r2_M3_pos$R2_marginal - r2_M2_pos$R2_marginal) * 100))
cat(sprintf("Total variance explained (fixed + random): %.1f%%\n\n", 
            r2_M3_pos$R2_conditional * 100))

cat("MODEL COMPARISON: NEGATIVE STRATUM (Specificity)\n")
cat("Variance Explained (R²):\n")
print(r2_comparison_neg)
cat("\n")

cat(sprintf("MODEL factors explain: %.1f%% of variance (marginal R²)\n", 
            r2_M1_neg$R2_marginal * 100))
cat(sprintf("Adding TEXT factors increases to: %.1f%% (gain: %.1f%%)\n", 
            r2_M2_neg$R2_marginal * 100, 
            (r2_M2_neg$R2_marginal - r2_M1_neg$R2_marginal) * 100))
cat(sprintf("Adding TASK factors increases to: %.1f%% (gain: %.1f%%)\n", 
            r2_M3_neg$R2_marginal * 100, 
            (r2_M3_neg$R2_marginal - r2_M2_neg$R2_marginal) * 100))
cat(sprintf("Total variance explained (fixed + random): %.1f%%\n\n", 
            r2_M3_neg$R2_conditional * 100))

cat("KEY FINDINGS\n")

cat(sprintf("   - Positive cases (detection): %.1f%% of data\n", 
            100 * nrow(positive_data) / nrow(prediction_data)))
cat(sprintf("   - Negative cases (specificity): %.1f%% of data\n", 
            100 * nrow(negative_data) / nrow(prediction_data)))
cat(sprintf("   - Accuracy differs: Positive=%.2f, Negative=%.2f\n\n",
            mean(positive_data$correct), mean(negative_data$correct)))

cat("Analysis complete. All outputs saved to:", output_dir, "\n")

sink()

# Full model summaries
sink(file.path(output_dir, "model_M3_positive_summary.txt"))
print(summary(M3_pos))
sink()

sink(file.path(output_dir, "model_M3_negative_summary.txt"))
print(summary(M3_neg))
sink()
