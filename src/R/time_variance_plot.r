#! /usr/local/bin/RScript

library(data.table)
library(ggplot2)
library(ggthemes)

# Set working directories
setwd("~/Projects/markov_traffic/")
figures.dir = "~/Projects/markov_traffic/Plots/"

# Read data
dd = data.table(read.delim("./Plots_data/baseline_objective_evolution.csv.gz",
                           sep=","))
dd[, object_distribution := as.character(object_distribution)]
dd[, problem := as.character(problem)]

# Node objective function
node.dd = dd[problem == "node"]
edge.dd = dd[problem == "edge"]
max_time = max(dd$time)

# Node baseline objective plot
p1 = ggplot(data=node.dd, aes(x=time, y=objective_value))
p1 = p1 + geom_line(aes(group=object_distribution))
p1 = p1 + geom_point()
p1 = p1 + scale_x_continuous(name="Time", breaks=seq(0,max_time,5))
p1 = p1 + scale_y_continuous(name="Nodes Monitoring Baseline Objective")
p1 = p1 + facet_wrap(~object_distribution, nrow=1)
p1 = p1 + theme_bw()
p1 = p1 + theme(strip.background = element_blank())
p1
ggsave(paste0(figures.dir, "nodes_baseline_evolution.pdf"), w=12, h=4)

# Edge baseline objective plot
p2 = ggplot(data=edge.dd, aes(x=time, y=objective_value))
p2 = p2 + geom_line(aes(group=object_distribution))
p2 = p2 + geom_point()
p2 = p2 + scale_x_continuous(name="Time", breaks=seq(0,max_time,1))
p2 = p2 + scale_y_continuous(name="Edge Monitoring Baseline Objective")
p2 = p2 + facet_wrap(~object_distribution, nrow=1)
p2 = p2 + theme_bw()
p2 = p2 + theme(strip.background = element_blank())
p2
ggsave(paste0(figures.dir, "edges_baseline_evolution.pdf"), w=12, h=4)
