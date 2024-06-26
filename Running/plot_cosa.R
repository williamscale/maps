library(tidyverse)
library(sf)
library(nngeo)

data.raw <- readRDS('D:/maps/Running/Data/kg_running_06142024.RDS')
# colSums(is.na(data.raw))
# data <- data.raw %>% na.omit()
# data <- data.raw
sf_use_s2(FALSE)

bexar.bounds <- st_read('D:/maps/Running/bexar/Bexar_County_Boundary.shp') %>%
  st_transform(4326)
cosa.bounds <- st_read('D:/maps/Running/cosa/CosaBoundary.shp') %>%
  st_transform(4326) %>%
  st_remove_holes()
cosa.streets <- st_read('D:/maps/Running/cosa/Streets.shp') %>%
  st_transform(4326)

strava.all <- data %>%
  st_as_sf(coords = c('Longitude', 'Latitude'),
           crs = st_crs(bexar.bounds))

strava.bexar <- st_filter(strava.all, bexar.bounds)
strava.cosa <- st_filter(strava.all, cosa.bounds)

streets.keep <- c('1604', '410', 'IH 35', 'IH 10', 'IH 37')
streets.keep.df <- cosa.streets %>%
  filter(grepl(paste(streets.keep, collapse = '|'), MSAG_NAME),
         !grepl('access', MSAG_NAME, ignore.case = TRUE),
         !grepl('at', MSAG_NAME, ignore.case = TRUE))

n <- 40
cosa.grid <- cosa.bounds %>% 
  st_make_grid(n = c(n, n))

cosa.grid.map <- st_intersection(cosa.bounds, cosa.grid) %>%
  st_as_sf() %>%
  mutate(grid_id = 1:n())

pings.binned <- cosa.grid.map %>% 
  st_join(strava.cosa) %>%
  count(grid_id) %>%
  mutate(n = na_if(n, 1))

quants <- quantile(pings.binned$n,
                   probs = seq(0, 1, 0.1),
                   na.rm = TRUE)

pings.binned <- pings.binned %>%
  mutate(n.quant = as.integer(cut(n, breaks = quants, labels = seq(1, 100, 10))))

ggplot() +
  geom_sf(data = pings.binned,
          aes(fill = n.quant),
          color = 'lightgrey',
          size = 0.1) +
  geom_sf(data = cosa.bounds,
          fill = NA,
          color = '#444444') +
  geom_sf(data = streets.keep.df,
          size = 0.7,
          color = '#F6CF65') +
  # color = 'turq/uoise4') +
  scale_fill_viridis_c(option = 'plasma',
                       na.value = 'white') +
  coord_sf() +
  theme_void() +
  theme(panel.background = element_rect(fill = 'slategrey',
                                        size = 5,
                                        color = '#444444'),
        legend.title = element_blank(),
        legend.position = c(0.84, 0.94),
        legend.direction = 'horizontal',
        legend.text = element_blank()) +
  guides(fill = guide_colorbar(ticks = FALSE)) +
  annotate('text', x = -98.26, y = 29.705, label = 'Frequency\u2192',
           color = 'black', size = 5, family = 'mono') +
  annotate('text', x = -98.65, y = 29.12, label = 'San Antonio Running',
           color = 'black', size = 8, family = 'mono', fontface = 'bold')
