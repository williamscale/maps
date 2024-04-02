library(dplyr)
library(ggplot2)
library(sf)
library(ggthemes)
library(showtext)

# neighborhoods <- read_sf(dsn = 'D:/mpls_map/Minneapolis_Neighborhoods')
streets <- read_sf(dsn = 'D:/sa_map/Streets')
# water <- read_sf(dsn = 'D:/mpls_map/Water-shp')
parks <- read_sf(dsn = 'D:/sa_map/Park_Boundaries')

# ggplot() +
#   geom_sf(data = streets) +
#   geom_sf(data = water,
#           fill = 'lightblue3',
#           color = 'lightblue3') +
#   geom_sf(data = parks,
#           fill = 'darkgreen',
#           color = 'darkgreen') +
#   annotate(geom = 'text',
#            x = -93.245,
#            y = 45.045,
#            label = 'MPLS',
#            color = 'black',
#            family = 'mono',
#            size = 12) +
#   theme_map() +
#   theme(plot.background = element_rect(fill = 'ivory'))

font_add_google('Fredericka the Great', 'fredericka the great')
# font_add_google('Oooh Baby', 'ooh baby')
# font_add_google('Charmonman', 'charmonman')
font_add_google('Satisfy', 'satisfy')
showtext_auto()


# png('D:/mpls_map/mpls.png', width = 8, height = 10, units = 'in', res = 300)
ggplot() +
  geom_sf(data = streets) +
  # geom_sf(data = water,
  #         fill = 'steelblue',
  #         color = 'steelblue') +
  geom_sf(data = parks,
          fill = 'darkgreen',
          color = 'darkgreen') +
  # geom_segment(aes(x = -93.335,
  #                  y = 44.883,
  #                  xend = -93.278,
  #                  yend = 44.883),
  #              linetype = 'dotted',
  #              color = 'steelblue') +
  # geom_segment(aes(x = -93.2518,
  #                  y = 44.883,
  #                  xend = -93.194,
  #                  yend = 44.883),
  #              linetype = 'dotted',
  #              color = 'steelblue') +
  # geom_segment(aes(x = -93.335,
  #                  y = 44.865,
  #                  xend = -93.194,
  #                  yend = 44.865),
  #              linetype = 'dotted',
  #              color = 'steelblue') +
  # annotate(geom = 'text',
  #          # x = -93.285,
  #          x = -93.265,
  #          y = 44.883,
  #          label = 'Visit your',
  #          color = 'darkgreen',
  #          family = 'satisfy',
  #          size = 72,
  #          hjust = 0.5) +
  # annotate(geom = 'text',
  #          x = -93.265,
  #          y = 44.875,
  #          label = 'San Antonio Parks',
  #          color = 'black',
  #          family = 'fredericka the great',
  #          size = 164,
  #          hjust = 0.5) +
  theme_map() +
  theme(plot.background = element_rect(fill = 'ivory'))
ggsave('D:/sa_map/sa0.png',
       sa.plot,
       height = 20,
       width = 16,
       dpi = 600,
       units = 'in', bg = 'transparent')
# dev.off()

# ggplot() +
#   geom_sf(data = neighborhoods,
#           aes(fill = BDNAME),
#           show.legend = FALSE) +
#   # geom_sf(data = water,
#   #         fill = 'lightblue3',
#   #         color = 'lightblue3') +
#   # geom_sf_label(data = neighborhoods,
#   #               aes(label = BDNAME)) +
#   geom_sf_text(data = neighborhoods,
#                 aes(label = BDNAME)) +
#   theme_map() +
#   theme(plot.background = element_rect(fill = 'ivory'))
