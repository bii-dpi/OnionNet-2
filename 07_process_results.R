library(glue)
library(dplyr)

results_summary <- NULL
for (direction in c("btd")) {
  for (decoy_style in c("ZS")) {
    path <- glue("results/{decoy_style}_{direction}.csv")
    results <- read.csv(text = readLines(path, warn = F))
    curr_summary <- results %>%
      group_by(epoch) %>%
      summarize(LogAUC = mean(LogAUC)) %>%
      mutate(direction = direction, decoy_style = decoy_style) %>%
      select(direction, decoy_style, LogAUC)
    results_summary <- rbind(results_summary, curr_summary) 
  }
}

write.csv(results_summary, "../IntegratedShallow/results/external_val_gnn.csv",
          row.names = F)


write_test_indiv <- function(decoy_style) {
  path <- glue("results/{decoy_style}_complete.csv")
  results <- read.csv(text = readLines(path, warn = F)) %>%
    filter(epoch == 49) %>%
    select(-epoch) %>%
    group_by(pdb_id) %>%
    summarize(LogAUC = mean(LogAUC))
  
  write.csv(results,
            glue("results/test_indiv_{decoy_style}_{decoy_style}.csv"),
            row.names = F)
}


for (decoy_style in c("ZS", "DS")) {
  write_test_indiv(decoy_style)
}
