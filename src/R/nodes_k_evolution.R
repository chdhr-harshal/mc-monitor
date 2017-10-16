#! /usr/local/bin/RScript

library(data.table)
library(ggplot2)
library(ggthemes)

# Set working directories
setwd("~/bu/Desktop/MCMonitor")
figures.dir = "~/bu/Desktop/MCMonitor/Plots/"

# Read data
dd = data.table(read.delim("./Plots_data/geo_nodes_k_objective_evolution.csv.gz",
                           sep=","))
dd[, method_name := factor(method_name)]
levels(dd$method_name)=c("Node-Betweenness", "Closeness", "In-Degree", "In-Probabibility", "Node-NumItems","PageRank", "Random", "NodeGreedy")
dd[, item_distribution := factor(item_distribution, levels=c("ego", "direct", "uniform", "inverse"),
                                 labels=c("Ego", "Direct", "Uniform", "Inverse"))]
dd = dd[method_name != "Random" & method_name != "PageRank" ]

max_k = max(dd$k) + 10


# All plots together in a horizontal row
p = ggplot(data=dd, aes(x=k, y=objective_values))
p = p + stat_summary(aes(y=objective_value, color=method_name, group=method_name), geom="line", fun.y=mean)
p = p + facet_wrap(~item_distribution, scales="free", nrow=1)
p = p + scale_x_continuous(name=expression(k), breaks=seq(0,max_k,10))
p = p + scale_y_continuous(name=expression(F[NI](S)))
p = p + theme_bw()
p = p + theme(strip.background = element_blank(),
              legend.margin = margin(-10,0,0,0),
              legend.title = element_blank(),
              legend.text = element_text(size=6),
              legend.position = "bottom",
              legend.direction = "horizontal",
              axis.text.y = element_text(size=5),
              axis.text.x = element_text(size=5),
              axis.title = element_text(size=7))
p = p + guides(colour = guide_legend(nrow = 1))
p
ggsave(paste0(figures.dir, "geo_nodes.pdf"), w=16, h=4, units ="cm")

# Direct distribution
p = ggplot(data=dd[item_distribution=="direct"], aes(x=k, y=objective_value))
p = p + stat_summary(aes(y=objective_value, color=method_name, group=method_name), geom="line", fun.y=mean)
p = p + scale_x_continuous(name=expression(k), breaks=seq(0,max_k,10))
p = p + scale_y_continuous(name=expression(F[NI](S)))
p = p + theme_bw()
p = p + theme(strip.background = element_blank(),
              legend.title = element_blank(),
              axis.text.y = element_text())
p
ggsave(paste0(figures.dir, "geo_direct_nodes.pdf"), units ="cm")

# Ego distribution
p = ggplot(data=dd[item_distribution=="ego"], aes(x=k, y=objective_value))
p = p + stat_summary(aes(y=objective_value, color=method_name, group=method_name), geom="line", fun.y=mean)
p = p + scale_x_continuous(name=expression(k), breaks=seq(0,max_k,10))
p = p + scale_y_continuous(name=expression(F[NI](S)))
p = p + theme_bw()
p = p + theme(strip.background = element_blank(),
              legend.title = element_blank(),
              legend.text = element_text(size=6),
              axis.text.y = element_text(size=5),
              axis.text.x = element_text(size=5),
              axis.title = element_text(size=7))
p
ggsave(paste0(figures.dir, "ba_ego_nodes.pdf"), h=8, w=10,units ="cm")

# Inverse distribution
p = ggplot(data=dd[item_distribution=="inverse"], aes(x=k, y=objective_value))
p = p + stat_summary(aes(y=objective_value, color=method_name, group=method_name), geom="line", fun.y=mean)
p = p + scale_x_continuous(name="k", breaks=seq(0,max_k,10))
p = p + scale_y_continuous(name="Node Monitoring Objective")
p = p + theme_bw()
p = p + theme(strip.background = element_blank())
p
ggsave(paste0(figures.dir, "geo_inverse_nodes.pdf"), units ="cm")

# Uniform distribution
p = ggplot(data=dd[item_distribution=="uniform"], aes(x=k, y=objective_value))
p = p + stat_summary(aes(y=objective_value, color=method_name, group=method_name), geom="line", fun.y=mean)
p = p + scale_x_continuous(name="k", breaks=seq(0,max_k,10))
p = p + scale_y_continuous(name="Node Monitoring Objective")
p = p + theme_bw()
p = p + theme(strip.background = element_blank())
p
ggsave(paste0(figures.dir, "geo_uniform_nodes.pdf"), units ="cm")