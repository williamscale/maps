library(tidyverse)
library(reticulate)

# source_python('D:/maps/Running/pickle_reader.py')
# gpx.data <- read_pickle_file('D:/maps/Running/Data/kg_06142024_gpx.pkl')

pd <- import('pandas')
gpx.data <- pd$read_pickle('D:/maps/Running/Data/kg_06142024_gpx.pkl')
fit.data <- pd$read_pickle('D:/maps/Running/Data/kg_06142024_fit.pkl')

fit.data <- fit.data %>% mutate(Activity = paste0('fit_', Activity))

data <- bind_rows(gpx.data, fit.data)

rm(pd, fit.data, gpx.data)
gc()

data %>% distinct(Type)

running <- data %>%
  filter(Type == 'running')
cycling <- data %>%
  filter(Type == 'cycling')

saveRDS(running, file = 'kg_running_06142024.RDS')
saveRDS(cycling, file = 'kg_cycling_06142024.RDS')
