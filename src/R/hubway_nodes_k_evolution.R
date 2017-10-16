#! /usr/local/bin/RScript

library(data.table)
library(ggplot2)
library(ggthemes)
library(gridExtra)

# Set working directories
setwd("~/bu/Desktop/MCMonitor")
figures.dir = "~/bu/Desktop/MCMonitor/Plots/"

# Read data
dd = data.table(read.delim("./Plots_data/hubway_nodes_evolution.csv.gz",
                           sep=","))
dd[, method_name := factor(method_name)]
levels(dd$method_name)=c("Node-Betweenness", "Closeness", "In-Degree", "In-Probabibility", "Node-NumItems", "Random", "NodeGreedy")
dd[, item_distribution := factor(item_distribution, levels=c("ego", "direct", "uniform", "inverse"),
                                 labels=c("Ego", "Direct", "Uniform", "Inverse"))]
dd = dd[method_name != "Random"]

max_k = max(dd$k) + 10

p1 = ggplot(data=dd, aes(x=k, y=objective_values))
p1 = p1 + stat_summary(aes(y=objective_value, color=method_name, group=method_name), geom="line", fun.y=mean)
p1 = p1 + scale_x_continuous(name=expression(k), breaks=seq(0,max_k,15))
p1 = p1 + scale_y_continuous(name=expression(F[NI](S)))
p1 = p1 + theme_bw()
p1 = p1 + theme(strip.background = element_blank(),
              legend.background = element_rect(fill = "transparent",colour= NA),
              legend.title = element_blank(),
              legend.text = element_text(size=5),
              legend.spacing.y = unit(1, units = "mm"),
              legend.key=element_blank(),
              legend.key.size = unit(3, units="mm"),
              legend.position = c(0.7,0.65),
              legend.box.background = element_blank(),
              axis.text.y = element_text(size=5),
              axis.text.x = element_text(size=5),
              axis.title = element_text(size=7))
p1
ggsave(paste0(figures.dir, "hubway_nodes.pdf"), w=5, h=5, units ="cm")

# Edges plot
dd = data.table(read.delim("./Plots_data/hubway_edges_evolution.csv.gz",
                           sep=","))
dd[, item_distribution := factor(item_distribution, levels=c("ego", "direct", "uniform", "inverse"),
                                 labels=c("Ego", "Direct", "Uniform", "Inverse"))]
dd[, method_name := factor(method_name)]
levels(dd$method_name)=c("Edge-Betweenness", "Edge-NumItems", "Probability", "Random", "EdgeGreedy")
dd = dd[method_name != "Greedy-Heuristic" & method_name != "Random"]

max_k = max(dd$k) + 10

# All plots together in a horizontal row
p2 = ggplot(data=dd, aes(x=k, y=objective_values))
p2 = p2 + stat_summary(aes(y=objective_value, color=method_name, group=method_name), geom="line", fun.y=mean)
p2 = p2 + scale_x_continuous(name=expression(k), breaks=seq(0,max_k,50))
p2 = p2 + scale_y_continuous(name=expression(F[ET](D)))
p2 = p2 + theme_bw()
p2 = p2 + theme(strip.background = element_blank(),
              legend.background = element_rect(fill = "transparent",colour= NA),
              legend.title = element_blank(),
              legend.text = element_text(size=5),
              legend.spacing.y = unit(1, units="mm"),
              legend.key=element_blank(),
              legend.key.size = unit(3, units="mm"),
              legend.position = c(0.7,0.75),
              legend.box.background = element_blank(),
              axis.text.y = element_text(size=5),
              axis.text.x = element_text(size=5),
              axis.title = element_text(size=7))
p2
ggsave(paste0(figures.dir, "hubway_edges.pdf"), w=5, h=5, units ="cm")

# Combine 2 plots
pdf(paste0(figures.dir, "hubway_nodes_edges.pdf"), width = 6, height=2)
grid.arrange(p1, p2, ncol=2)
dev.off()
