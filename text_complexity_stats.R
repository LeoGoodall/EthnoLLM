suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

# Load the prepared data from performance diagnostics (already has FOG and MTLD computed)
data_file <- "performance_diagnostics_ritual/performance_data_prepared.csv"

df <- read_csv(data_file, show_col_types = FALSE)

# Get unique rituals (text-level stats)
unique_texts <- df %>%
  distinct(ritual_number, gunning_fog, mtld, char_length)

# Gunning Fog statistics
fog_stats <- unique_texts %>%
  filter(!is.na(gunning_fog)) %>%
  summarise(
    n = n(),
    mean = mean(gunning_fog),
    sd = sd(gunning_fog),
    median = median(gunning_fog),
    min = min(gunning_fog),
    max = max(gunning_fog),
    q25 = quantile(gunning_fog, 0.25),
    q75 = quantile(gunning_fog, 0.75)
  )

# MTLD statistics
mtld_stats <- unique_texts %>%
  filter(!is.na(mtld)) %>%
  summarise(
    n = n(),
    mean = mean(mtld),
    sd = sd(mtld),
    median = median(mtld),
    min = min(mtld),
    max = max(mtld),
    q25 = quantile(mtld, 0.25),
    q75 = quantile(mtld, 0.75)
  )

# Save to CSV
output_df <- tibble(
  Metric = c("Gunning Fog Mean", "Gunning Fog SD", "Gunning Fog Median", 
             "Gunning Fog Min", "Gunning Fog Max",
             "MTLD Mean", "MTLD SD", "MTLD Median", "MTLD Min", "MTLD Max"),
  Value = c(fog_stats$mean, fog_stats$sd, fog_stats$median, 
            fog_stats$min, fog_stats$max,
            mtld_stats$mean, mtld_stats$sd, mtld_stats$median, 
            mtld_stats$min, mtld_stats$max)
)

write_csv(output_df, "figures_R/text_complexity_statistics.csv")
message("Saved: figures_R/text_complexity_statistics.csv")

