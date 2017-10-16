#! /usr/local/bin/RScript

library(data.table)
library(ggplot2)
library(ggthemes)

# Set working directories
setwd("~/bu/Desktop/MCMonitor")
figures.dir = "~/bu/Desktop/MCMonitor/Plots/"

# Read data
dd = data.table(read.delim("./Plots_data/hubway_edges_evolution.csv.gz",
                           sep=","))
dd[, item_distribution := factor(item_distribution, levels=c("ego", "direct", "uniform", "inverse"),
                                 labels=c("Ego", "Direct", "Uniform", "Inverse"))]
dd[, method_name := factor(method_name)]
levels(dd$method_name)=c("Edge-Betweenness", "NumItems", "Probability", "Random", "EdgeGreedy")
dd = dd[method_name != "Greedy-Heuristic" & method_name != "Random"]

max_k = max(dd$k) + 10

# All plots together in a horizontal row
p = ggplot(data=dd, aes(x=k, y=objective_values))
p = p + stat_summary(aes(y=objective_value, color=method_name, group=method_name), geom="line", fun.y=mean)
p = p + scale_x_continuous(name=expression(k), breaks=seq(0,max_k,50))
p = p + scale_y_continuous(name=expression(F[ET](D)))
p = p + theme_bw()
p = p + theme(strip.background = element_blank(),
              legend.title = element_blank(),
              legend.text = element_text(size=6),
              legend.position = c(0.7,0.75),
              axis.text.y = element_text(size=5),
              axis.text.x = element_text(size=5),
              axis.title = element_text(size=7))
p
ggsave(paste0(figures.dir, "hubway_edges.pdf"), w=8, h=8, units ="cm")