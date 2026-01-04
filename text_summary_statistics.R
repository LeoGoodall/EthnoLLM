suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

# ============================================================================
# TEXT SUMMARY STATISTICS
# Count of ethnographic texts and text length summary statistics
# ============================================================================

# Load data
rituals <- read_csv("data/rituals_codes.csv", show_col_types = FALSE)
excluded <- read_csv("data/exclude.csv", show_col_types = FALSE)$exclude

# Filter out excluded rituals
rituals_included <- rituals %>%

  filter(!(ritual_number %in% excluded))

# Count
n_total <- nrow(rituals)
n_excluded <- length(excluded)
n_included <- nrow(rituals_included)
n_cultures <- n_distinct(rituals_included$Culture_Name)

# Text length summary statistics (characters)
char_lengths <- rituals_included$character_length

mean_char <- mean(char_lengths, na.rm = TRUE)
sd_char <- sd(char_lengths, na.rm = TRUE)
median_char <- median(char_lengths, na.rm = TRUE)
min_char <- min(char_lengths, na.rm = TRUE)
max_char <- max(char_lengths, na.rm = TRUE)
q1_char <- quantile(char_lengths, 0.25, na.rm = TRUE)
q3_char <- quantile(char_lengths, 0.75, na.rm = TRUE)

# Word counts (actual count from text)
library(stringr)
word_counts <- sapply(rituals_included$text, function(txt) {
  length(str_split(txt, "\\s+")[[1]])
})

mean_words <- mean(word_counts, na.rm = TRUE)
sd_words <- sd(word_counts, na.rm = TRUE)
median_words <- median(word_counts, na.rm = TRUE)
min_words <- min(word_counts, na.rm = TRUE)
max_words <- max(word_counts, na.rm = TRUE)

# Save summary to CSV
summary_df <- tibble(
  Metric = c("Total rituals", "Excluded rituals", "Included rituals", "Unique cultures",
             "Mean word count", "SD word count", "Median word count", 
             "Min word count", "Max word count",
             "Mean character length", "SD character length", 
             "Median character length", "Min character length", "Max character length"),
  Value = c(n_total, n_excluded, n_included, n_cultures,
            round(mean_words, 1), round(sd_words, 1), round(median_words, 1),
            min_words, max_words,
            round(mean_char, 1), round(sd_char, 1),
            round(median_char, 1), min_char, max_char)
)

write_csv(summary_df, "figures_R/text_summary_statistics.csv")
cat("Saved: figures_R/text_summary_statistics.csv\n")
