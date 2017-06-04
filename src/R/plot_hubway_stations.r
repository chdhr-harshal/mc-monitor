#! /usr/local/bin/RScript

library(data.table)
library(ggplot2)
library(ggthemes)
library(ggmap)

# Set working directories
setwd("~/Projects/markov_traffic/")
figures.dir = "~/Projects/markov_traffic/Plots/"

# Read data
nodes.dd = data.table(read.delim("./Plots_data/hubway_plot_nodes.csv.gz",
                          sep=","))
edges.dd = data.table(read.delim("./Plots_data/hubway_plot_edges.csv.gz",
                          sep=","))

# Plot important stations onto map
p1 = qmplot(lng, lat, data=nodes.dd, maptype = "toner-lite", color = I("red"), size=I(2), xlab="Longitude", ylab="Latitude")
p1 = p1 + theme_bw()
p1 = p1 + theme(strip.background = element_blank())
p1
ggsave(paste0(figures.dir, "hubway_stations.pdf"), w=12, h=4)

# Plot important paths onto map

p2 = qmplot(lng, lat, data=edges.dd, maptype = "toner-lite", color=node_type, size=I(2), legend = "top", xlab="Longitude", ylab="Latitude")
p2 = p2 + geom_path(data=edges.dd, aes(x=lng, y=lat, group=edge_id), color="black", size=I(0.5))
p2 = p2 + labs(color="Node Type")
p2 = p2 + theme_bw()
p2
ggsave(paste0(figures.dir, "hubway_paths.pdf"), w=12, h=4)
