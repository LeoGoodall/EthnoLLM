library(tidyverse)
library(maps)
library(gridExtra)

# Load rituals data with coordinates
rituals <- read_csv("data/rituals_codes.csv")

# Create simple coverage map showing text frequency by location
create_coverage_map <- function() {
  # Clean coordinates
  rituals_clean <- rituals %>%
    filter(!is.na(Latitude), !is.na(Longitude)) %>%
    filter(Latitude >= -90 & Latitude <= 90) %>%
    filter(Longitude >= -180 & Longitude <= 180)
  
  # Count texts per location
  location_counts <- rituals_clean %>%
    group_by(Latitude, Longitude) %>%
    summarise(
      n_texts = n(),
      Culture_Name = first(Culture_Name),
      .groups = "drop"
    )
  
  # Get world map data
  world <- map_data("world")
  
  # Create the map
  p <- ggplot() +
    geom_polygon(
      data = world,
      aes(x = long, y = lat, group = group),
      fill = "#c0c0c0",
      color = "white",
      linewidth = 0.2
    ) +
    geom_point(
      data = location_counts,
      aes(x = Longitude, y = Latitude, size = n_texts),
      color = "#2c3e50",
      alpha = 0.6
    ) +
    scale_size_continuous(
      range = c(1.5, 8),
      name = "Number of\nethnographic texts",
      breaks = c(1, 5, 10, 20, 30)
    ) +
    coord_fixed(ratio = 1.3, xlim = c(-180, 180), ylim = c(-60, 85)) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      axis.line = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank(),
      panel.grid = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(5, 5, 5, 5)
    ) +
    guides(size = guide_legend(nrow = 1))
  
  return(p)
}

# Generate and save coverage map
coverage_map <- create_coverage_map()
ggsave("figures_R/ethnographic_coverage_map.pdf", coverage_map, 
       width = 10, height = 5, device = cairo_pdf)
cat("Saved ethnographic coverage map to figures_R/ethnographic_coverage_map.pdf\n")


# Clean coordinates
rituals_clean <- rituals %>%
  filter(!is.na(Latitude), !is.na(Longitude)) %>%
  filter(Latitude >= -90 & Latitude <= 90) %>%
  filter(Longitude >= -180 & Longitude <= 180)

# Model names and their file names
models <- c(
  "qwen3" = "qwen3",
  "gpt5nano" = "gpt5nano",
  "gptoss120b" = "gptoss120b",
  "deepseekv31671b" = "deepseekv31671b",
  "llama33b" = "llama33b",
  "claudesonnet45" = "claudesonnet45",
  "perplexity" = "perplexity"
)

# Model display names for titles
model_display_names <- c(
  "gpt5nano" = "GPT-5 Nano",
  "gptoss120b" = "GPT-OSS 120B",
  "deepseekv31671b" = "DeepSeek V3.1 671B",
  "llama33b" = "Llama 3.2 Instruct (3B)",
  "claudesonnet45" = "Claude Sonnet 4.5",
  "qwen3" = "Qwen 3 Instruct (4B)",
  "perplexity" = "Perplexity Sonar"
)

to_num <- function(x) suppressWarnings(as.numeric(x))

# Robust binary detector (handles both datasets without backslash escapes)
is_binary_feature2 <- function(options_txt) {
  x <- options_txt
  ok <- !is.na(x) & (
    grepl("Present (1)/ Absent (0)", x, fixed = TRUE) |
    grepl("0[[:space:]]*=[[:space:]]*absent", x, ignore.case = TRUE) |
    grepl("1[[:space:]]*=[[:space:]]*present", x, ignore.case = TRUE)
  )
  ifelse(ok, TRUE, FALSE)
}


# Binary F1 (positive class = 1)
f1_binary <- function(y_true, y_pred) {
  yt <- to_num(y_true); yp <- to_num(y_pred)
  valid <- !is.na(yt) & !is.na(yp)
  if (!any(valid)) return(NA_real_)
  yt <- ifelse(yt[valid] != 0, 1L, 0L)
  yp <- ifelse(yp[valid] != 0, 1L, 0L)
  tp <- sum(yt == 1L & yp == 1L)
  fp <- sum(yt == 0L & yp == 1L)
  fn <- sum(yt == 1L & yp == 0L)
  precision <- if ((tp + fp) > 0) tp/(tp + fp) else NA_real_
  recall <- if ((tp + fn) > 0) tp/(tp + fn) else NA_real_
  if (is.na(precision) || is.na(recall) || (precision + recall) == 0) return(NA_real_)
  2 * precision * recall/(precision + recall)
}

# Multiclass macro-F1 over classes present in ground truth; 999 ignored
f1_multiclass_macro <- function(y_true, y_pred) {
  yt <- to_num(y_true); yp <- to_num(y_pred)
  valid <- !is.na(yt) & !is.na(yp)
  yt <- yt[valid]; yp <- yp[valid]
  if (length(yt) == 0) return(NA_real_)
  classes <- sort(unique(yt))
  if (length(classes) == 0) return(NA_real_)
  per <- vapply(classes, function(cn) {
    tp <- sum(yt == cn & yp == cn)
    fp <- sum(yt != cn & yp == cn)
    fn <- sum(yt == cn & yp != cn)
    p <- if ((tp + fp) > 0) tp/(tp + fp) else NA_real_
    r <- if ((tp + fn) > 0) tp/(tp + fn) else NA_real_
    if (is.na(p) || is.na(r) || (p + r) == 0) return(NA_real_)
    2*p*r/(p + r)
  }, numeric(1))
  mean(per, na.rm = TRUE)
}

# Function to process one model for a given dataset
process_model <- function(model_name, model_file, dataset_name) {
  # Determine results directory and features file
  if (dataset_name == "synchrony") {
    results_dir <- "synchrony"
    features_file <- "data/features_synchrony.csv"
    has_human_suffix <- TRUE
  } else if (dataset_name == "all") {
    results_dir <- "all"
    features_file <- "data/features_all.csv"
    has_human_suffix <- FALSE
  } else {
    stop(paste("Unknown dataset:", dataset_name))
  }
  
  results_file <- paste0(results_dir, "/results_", model_file, "_mtp.csv")
  
  if (!file.exists(results_file)) {
    warning(paste("File not found:", results_file))
    return(NULL)
  }
  
  # Load results and features
  results <- read_csv(results_file, show_col_types = FALSE)
  features <- read_csv(features_file, show_col_types = FALSE)
  # Feature types by feature_variable
  feature_types <- features %>% mutate(
    type = ifelse(is_binary_feature2(feature_options), "binary", "multiclass")
  ) %>% select(feature_variable, type)
  
  # Merge with rituals to get coordinates and culture names
  data <- results %>%
    left_join(rituals_clean %>% 
                select(ritual_number, Latitude, Longitude, Culture_Name),
              by = "ritual_number") %>%
    filter(!is.na(Latitude), !is.na(Longitude))
  
  # Compute per-sample micro-F1 across all features, then average per coordinate
  feature_list <- features$feature_variable
  # helper to compute per-row micro-F1
  micro_f1_row <- function(row) {
    true_labels <- character(0)
    pred_labels <- character(0)
    for (i in seq_along(feature_list)) {
      feat <- feature_list[i]
      ftype <- feature_types$type[match(feat, feature_types$feature_variable)]
      if (has_human_suffix) {
        base <- gsub("_human$", "", feat)
        col_true <- feat
        col_pred <- paste0(base, "_llm")
      } else {
        col_true <- feat
        col_pred <- paste0(feat, "_llm")
      }
      if (!(col_true %in% names(row)) || !(col_pred %in% names(row))) next
      yt <- suppressWarnings(as.numeric(row[[col_true]]))
      yp <- suppressWarnings(as.numeric(row[[col_pred]]))
      if (identical(ftype, "multiclass")) {
        # include all numeric labels (including 999) when present
        if (!is.na(yt)) true_labels <- c(true_labels, paste0(feat, "=", yt))
        if (!is.na(yp)) pred_labels <- c(pred_labels, paste0(feat, "=", yp))
      } else {
        # binary: map non-zero to 1, zero to 0
        if (!is.na(yt) && (ifelse(yt != 0, 1, 0)) == 1) true_labels <- c(true_labels, feat)
        if (!is.na(yp) && (ifelse(yp != 0, 1, 0)) == 1) pred_labels <- c(pred_labels, feat)
      }
    }
    tp <- length(intersect(true_labels, pred_labels))
    fp <- length(setdiff(pred_labels, true_labels))
    fn <- length(setdiff(true_labels, pred_labels))
    denom <- (2 * tp + fp + fn)
    if (denom > 0) (2 * tp) / denom else NA_real_
  }
  # compute per-row F1
  data_with_f1 <- data %>% mutate(F1 = purrr::pmap_dbl(as.list(.), function(...) {
    row <- list(...)
    # convert named list to environment-like list
    micro_f1_row(row)
  }))
  # average per coordinate
  f1_agg <- data_with_f1 %>%
    group_by(Latitude, Longitude, Culture_Name) %>%
    summarise(mean_f1 = mean(F1, na.rm = TRUE), .groups = "drop")
  
  # Count instances per coordinate
  count_agg <- rituals_clean %>%
    group_by(Latitude, Longitude) %>%
    summarise(count = n(), .groups = "drop")
  
  # Merge F1 scores with counts
  final_data <- f1_agg %>%
    left_join(count_agg, by = c("Latitude", "Longitude")) %>%
    mutate(count = ifelse(is.na(count), 1, count))
  
  final_data$model <- model_name
  return(final_data)
}

# Function to create plots for a dataset
create_plots <- function(dataset_name) {
  # Process all models for this dataset
  all_data_list <- list()
  for (i in seq_along(models)) {
    model_name <- names(models)[i]
    model_file <- models[[i]]
    result <- process_model(model_name, model_file, dataset_name)
    if (!is.null(result)) {
      all_data_list[[model_name]] <- result
    }
  }
  all_data <- bind_rows(all_data_list)
  
  # Draw the world map template
  world <- map_data("world")
  
  # Create a plot for each model
  plot_list <- list()
  
  for (model_name in names(models)) {
    model_data <- all_data %>% filter(model == model_name)
    
    if (nrow(model_data) == 0) {
      next
    }
    
    p <- ggplot() +
      geom_polygon(
        data = world,
        aes(x = long, y = lat, group = group),
        fill = "lightgray",
        color = "white",
        linewidth = 0.1
      ) +
      geom_point(
        data = model_data,
        aes(x = Longitude, y = Latitude, size = count, color = mean_f1),
        alpha = 0.6
      ) +
      geom_text(
        data = model_data %>% filter(!is.na(Culture_Name) & Culture_Name != ""),
        aes(x = Longitude, y = Latitude, label = Culture_Name),
        size = 0.8,
        color = "black",
        vjust = 0.5,
        hjust = 0.5,
        angle = 0,
        show.legend = FALSE
      ) +
      scale_size_continuous(range = c(1, 5), name = "Count") +
      scale_color_gradient(low = "red", high = "green", name = "F1 Score", limits = c(0, 1)) +
      coord_fixed(ratio = 1.3, xlim = c(-180, 180), ylim = c(-90, 90)) +
      theme_bw() +
      theme(
        legend.position = "none",
        axis.line = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        plot.title = element_text(size = 10, face = "bold", hjust = 0.5)
      ) +
      ggtitle(model_display_names[model_name])
    
    plot_list[[model_name]] <- p
  }
  
  return(plot_list)
}

# Function to create aggregated plot (average across all models)
create_aggregated_plot <- function(dataset_name) {
  # Process all models for this dataset
  all_data_list <- list()
  for (i in seq_along(models)) {
    model_name <- names(models)[i]
    model_file <- models[[i]]
    result <- process_model(model_name, model_file, dataset_name)
    if (!is.null(result)) {
      all_data_list[[model_name]] <- result
    }
  }
  all_data <- bind_rows(all_data_list)
  
  if (nrow(all_data) == 0) {
    return(NULL)
  }
  
  # Aggregate F1 scores across all models for each coordinate
  agg_data <- all_data %>%
    group_by(Latitude, Longitude) %>%
    summarise(
      mean_f1 = mean(mean_f1, na.rm = TRUE),
      Culture_Name = first(Culture_Name),
      count = first(count),
      .groups = "drop"
    )

# Draw the world map template
world <- map_data("world")

  # Create the aggregated plot
  p <- ggplot() +
    geom_polygon(
      data = world,
      aes(x = long, y = lat, group = group),
      fill = "lightgray",
      color = "white",
      linewidth = 0.1
  ) +
  geom_point(
      data = agg_data,
      aes(x = Longitude, y = Latitude, size = count, color = mean_f1),
      alpha = 0.6
    ) +
    geom_text(
      data = agg_data %>% filter(!is.na(Culture_Name) & Culture_Name != ""),
      aes(x = Longitude, y = Latitude, label = Culture_Name),
      size = 0.8,
      color = "black",
      vjust = 0.5,
      hjust = 0.5,
      angle = 0,
      show.legend = FALSE
    ) +
    scale_size_continuous(range = c(1, 5), name = "Count") +
    scale_color_gradient(low = "red", high = "green", name = "F1 Score", limits = c(0, 1)) +
    coord_fixed(ratio = 1.3, xlim = c(-180, 180), ylim = c(-90, 90)) +
  theme_bw() +
    theme(
      legend.position = "none",
      axis.line = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      plot.title = element_text(size = 10, face = "bold", hjust = 0.5)
    ) +
    ggtitle(paste0(toupper(substr(dataset_name, 1, 1)), substr(dataset_name, 2, nchar(dataset_name))))
  
  return(p)
}

# Generate plots for synchrony dataset and save as PDF
synchrony_plots <- create_plots("synchrony")
if (length(synchrony_plots) > 0) {
  pdf("figures_R/synchrony/map_synchrony.pdf", width = 10, height = 10)
  grid_plot <- do.call(grid.arrange, c(synchrony_plots, ncol = 2))
  print(grid_plot)
  dev.off()
  cat("Saved synchrony map to figures_R/synchrony/map_synchrony.pdf\n")
}

# Generate plots for all dataset and save as PDF
all_plots <- create_plots("all")
if (length(all_plots) > 0) {
  pdf("figures_R/all/map_all.pdf", width = 10, height = 10)
  grid_plot <- do.call(grid.arrange, c(all_plots, ncol = 2))
  print(grid_plot)
  dev.off()
  cat("Saved all map to figures_R/all/map_all.pdf\n")
}

# Generate aggregated combined plots
synchrony_agg <- create_aggregated_plot("synchrony")
all_agg <- create_aggregated_plot("all")

if (!is.null(synchrony_agg) && !is.null(all_agg)) {
  pdf("figures_R/maps_agg_combined.pdf", width = 10, height = 5)
  grid_plot <- grid.arrange(synchrony_agg, all_agg, ncol = 2)
  print(grid_plot)
  dev.off()
  cat("Saved aggregated combined maps to figures_R/maps_agg_combined.pdf\n")
}

# STATISTICAL ANALYSIS: Region vs F1 Scores

# First, sanity check: log all unique Region values
if ("Region" %in% names(rituals)) {
  unique_regions <- unique(rituals$Region[!is.na(rituals$Region)])
  cat("Unique Region values (sanity check):\n")
  for (region in sort(unique_regions)) {
    count <- sum(rituals$Region == region, na.rm = TRUE)
    cat("  - ", region, ": ", count, " rituals\n", sep = "")
  }
  cat("\n")
} else {
  cat("Warning: Region column not found in rituals data!\n")
}

# Function to perform region-based statistical analysis for a model
analyze_model_regions <- function(model_name, model_file, dataset_name) {
  # Process the model
  result <- process_model(model_name, model_file, dataset_name)
  
  if (is.null(result) || nrow(result) == 0) {
    return(NULL)
  }
  
  # Merge with Region data from rituals
  result_with_region <- result %>%
    left_join(rituals %>% 
                select(Latitude, Longitude, Region) %>%
                distinct(),
              by = c("Latitude", "Longitude")) %>%
    filter(!is.na(Region), !is.na(mean_f1))
  
  if (nrow(result_with_region) == 0) {
    cat("No data with Region for", model_name, "in", dataset_name, "\n")
    return(NULL)
  }
  
  # Statistical tests
  # 1. Overall test: ANOVA or Kruskal-Wallis (depending on normality)
  regions <- unique(result_with_region$Region)
  
  if (length(regions) < 2) {
    cat("Not enough regions for", model_name, "in", dataset_name, "- need at least 2\n")
    return(NULL)
  }
  
  # Check normality using Shapiro-Wilk (on residuals or per group)
  # For simplicity, we'll use both ANOVA and Kruskal-Wallis
  
  # Perform ANOVA
  aov_result <- tryCatch({
    aov(mean_f1 ~ Region, data = result_with_region)
  }, error = function(e) NULL)
  
  # Perform Kruskal-Wallis test (non-parametric alternative)
  kw_result <- tryCatch({
    kruskal.test(mean_f1 ~ Region, data = result_with_region)
  }, error = function(e) NULL)
  
  # Summary statistics by region
  region_stats <- result_with_region %>%
    group_by(Region) %>%
    summarise(
      n = n(),
      mean_f1 = mean(mean_f1, na.rm = TRUE),
      sd_f1 = sd(mean_f1, na.rm = TRUE),
      median_f1 = median(mean_f1, na.rm = TRUE),
      min_f1 = min(mean_f1, na.rm = TRUE),
      max_f1 = max(mean_f1, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(mean_f1))
  
  # Pairwise comparisons (t-tests with Bonferroni correction)
  pairwise_results <- data.frame()
  
  if (length(regions) > 1) {
    region_pairs <- combn(regions, 2, simplify = FALSE)
    
    for (pair in region_pairs) {
      region1 <- pair[1]
      region2 <- pair[2]
      
      data1 <- result_with_region$mean_f1[result_with_region$Region == region1]
      data2 <- result_with_region$mean_f1[result_with_region$Region == region2]
      
      if (length(data1) > 0 && length(data2) > 0) {
        # Perform t-test
        ttest <- tryCatch({
          t.test(data1, data2)
        }, error = function(e) NULL)
        
        if (!is.null(ttest)) {
          pairwise_results <- rbind(pairwise_results, data.frame(
            region1 = region1,
            region2 = region2,
            mean_diff = ttest$estimate[1] - ttest$estimate[2],
            p_value = ttest$p.value,
            stringsAsFactors = FALSE
          ))
        }
      }
    }
    
    # Apply Bonferroni correction
    if (nrow(pairwise_results) > 0) {
      n_comparisons <- nrow(pairwise_results)
      pairwise_results$p_value_corrected <- pmin(pairwise_results$p_value * n_comparisons, 1)
      pairwise_results$significant <- pairwise_results$p_value_corrected < 0.05
    }
  }
  
  # Compile results
  results_list <- list(
    model = model_name,
    dataset = dataset_name,
    n_regions = length(regions),
    regions = paste(regions, collapse = ", "),
    overall_mean_f1 = mean(result_with_region$mean_f1, na.rm = TRUE),
    region_stats = region_stats,
    anova_p = if (!is.null(aov_result)) summary(aov_result)[[1]][["Pr(>F)"]][1] else NA,
    kruskal_wallis_p = if (!is.null(kw_result)) kw_result$p.value else NA,
    kruskal_wallis_chi2 = if (!is.null(kw_result)) kw_result$statistic else NA,
    pairwise = pairwise_results
  )
  
  return(results_list)
}

# Perform analysis for all models and datasets
all_results <- list()

for (dataset_name in c("synchrony", "all")) {
  cat("\n--- Analysis for", dataset_name, "dataset ---\n\n")
  
  for (i in seq_along(models)) {
    model_name <- names(models)[i]
    model_file <- models[[i]]
    
    cat("Analyzing", model_name, "...\n")
    result <- analyze_model_regions(model_name, model_file, dataset_name)
    
    if (!is.null(result)) {
      all_results[[paste(dataset_name, model_name, sep = "_")]] <- result
      
      # Print summary
      cat("Regions:", result$regions, "\n")
      cat("Overall mean F1:", round(result$overall_mean_f1, 3), "\n")
      cat("ANOVA p-value:", if (!is.na(result$anova_p)) round(result$anova_p, 4) else "N/A", "\n")
      cat("Kruskal-Wallis p-value:", if (!is.na(result$kruskal_wallis_p)) round(result$kruskal_wallis_p, 4) else "N/A", "\n")
      
      if (nrow(result$region_stats) > 0) {
        cat("\n  Region Statistics:\n")
        print(result$region_stats)
      }
      
      if (nrow(result$pairwise) > 0) {
        cat("\n  Significant pairwise differences (Bonferroni corrected):\n")
        sig_pairs <- result$pairwise[result$pairwise$significant, ]
        if (nrow(sig_pairs) > 0) {
          print(sig_pairs[, c("region1", "region2", "mean_diff", "p_value_corrected")])
        } else {
          cat("    None\n")
        }
      }
      cat("\n")
    }
  }
}

# Save results to CSV files
if (length(all_results) > 0) {
  # Create summary table
  summary_rows <- list()
  region_stat_rows <- list()
  pairwise_rows <- list()
  
  for (result_key in names(all_results)) {
    result <- all_results[[result_key]]
    
    # Summary row
    summary_rows[[result_key]] <- data.frame(
      dataset = result$dataset,
      model = result$model,
      n_regions = result$n_regions,
      overall_mean_f1 = result$overall_mean_f1,
      anova_p = result$anova_p,
      kruskal_wallis_p = result$kruskal_wallis_p,
      kruskal_wallis_chi2 = result$kruskal_wallis_chi2,
      stringsAsFactors = FALSE
    )
    
    # Region statistics
    if (nrow(result$region_stats) > 0) {
      region_stat_rows[[result_key]] <- result$region_stats %>%
        mutate(dataset = result$dataset, model = result$model, .before = 1)
    }
    
    # Pairwise comparisons
    if (nrow(result$pairwise) > 0) {
      pairwise_rows[[result_key]] <- result$pairwise %>%
        mutate(dataset = result$dataset, model = result$model, .before = 1)
    }
  }
  
  # Write summary CSV
  summary_df <- bind_rows(summary_rows)
  write_csv(summary_df, "figures_R/region_analysis_summary.csv")
  cat("Saved region analysis summary to figures_R/region_analysis_summary.csv\n")
  
  # Write region statistics CSV
  if (length(region_stat_rows) > 0) {
    region_stat_df <- bind_rows(region_stat_rows)
    write_csv(region_stat_df, "figures_R/region_analysis_by_region.csv")
    cat("Saved region statistics to figures_R/region_analysis_by_region.csv\n")
  }
  
  # Write pairwise comparisons CSV
  if (length(pairwise_rows) > 0) {
    pairwise_df <- bind_rows(pairwise_rows)
    write_csv(pairwise_df, "figures_R/region_analysis_pairwise.csv")
    cat("Saved pairwise comparisons to figures_R/region_analysis_pairwise.csv\n")
  }
  
}
