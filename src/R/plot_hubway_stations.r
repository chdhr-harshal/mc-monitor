#! /usr/local/bin/RScript

library(data.table)
library(ggplot2)
library(ggthemes)
library(ggmap)

# Set working directories
setwd("~/bu/Desktop/MCMonitor")
figures.dir = "~/bu/Desktop/MCMonitor/Plots/"

# Read data
nodes.dd = data.table(read.delim("./Plots_data/hubway_plot_nodes_5.csv.gz",
                          sep=","))
edges.dd = data.table(read.delim("./Plots_data/hubway_plot_edges.csv.gz",
                          sep=","))

# Plot important stations onto map
# p1 = qmplot(lng, lat, data=nodes.dd, maptype = "toner-lite", color = I("red"), size=I(1), xlab="", ylab="", zoom=14, extent="panel")
# p1 = p1 + theme_bw()
# p1 = p1 + theme(strip.background = element_blank(),
#                 axis.text.x = element_blank(),
#                 axis.text.y = element_blank(),
#                 axis.ticks.x = element_blank(),
#                 axis.ticks.y = element_blank(),
#                 plot.margin = unit(c(0.5,0.5,0,0), "cm"))
# p1
# ggsave(paste0(figures.dir, "hubway_stations.pdf"), w=8, h=4, units="cm")

p = get_map(location = c(-71.13, 42.340, -71.05, 42.365), source = "google", zoom = 14, maptype="toner-lite")
p = ggmap(p)
p = p + geom_point(data=nodes.dd, aes(x=lng, y=lat), color="red", size=I(1))
p = p + theme_bw()
p = p + theme(strip.background = element_blank(),
              axis.title.x = element_blank(),
              axis.title.y = element_blank(),
              axis.text.x = element_blank(),
              axis.text.y = element_blank(),
              axis.ticks.x = element_blank(),
              axis.ticks.y = element_blank(),
              plot.margin = unit(c(0.5,0.5,0,0), "cm"))
p
ggsave(paste0(figures.dir, "hubway_stations_5.pdf"), w=8, h=4, units="cm")


# Plot important paths onto map

p2 = qmplot(lng, lat, data=edges.dd, maptype = "toner-lite", color=node_type, size=I(1), xlab="", ylab="")
p2 = p2 + geom_path(data=edges.dd, aes(x=lng, y=lat, group=edge_id), color="black", size=I(0.5))
p2 = p2 + labs(color="Node Type")
p2 = p2 + theme_bw()
p2 = p2 + theme(strip.background = element_blank(),
                axis.text.x = element_blank(),
                axis.text.y = element_blank(),
                legend.position = c(0.10,0.15),
                legend.text = element_text(size=3),
                legend.background = element_blank(),
                legend.key = element_blank(),
                legend.key.size = unit(0.32, "cm"),
                legend.title = element_blank())
p2
ggsave(paste0(figures.dir, "hubway_paths.pdf"), w=9, h=4.5, units="cm")
