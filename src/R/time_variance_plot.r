#! /usr/local/bin/RScript

library(data.table)
library(ggplot2)
library(ggthemes)

# Set working directories
setwd("~/bu/Desktop/MCMonitor")
figures.dir = "~/bu/Desktop/MCMonitor/Plots/"

# Read data
dd = data.table(read.delim("./Plots_data/time_evolution.csv.gz",
                           sep=","))
dd[, item_distribution := as.character(item_distribution)]
dd[, graph := as.character(graph)]

max_time = max(dd$t)+1

# Time evolution plot
p = ggplot(data=dd, aes(x=t, y=objective_value))
p = p + stat_summary(aes(y=objective_value, color=item_distribution, group=item_distribution), geom="line", fun.y=mean)
p = p + scale_x_continuous(name="Time", breaks=seq(0,max_time,1))
p = p + scale_y_continuous(name="Baseline Objective")
p = p + facet_grid(graph~item_distribution)
p = p + theme_bw()
p = p + theme(strip.background = element_blank())
p
ggsave(paste0(figures.dir, "time_evolution.pdf"), w=16, h=16)
