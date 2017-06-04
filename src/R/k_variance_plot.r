#! /usr/local/bin/RScript

library(data.table)
library(ggplot2)
library(ggthemes)

# Set working directories
setwd("~/Projects/markov_traffic/")
figures.dir = "~/Projects/markov_traffic/Plots/"

# Read data
dd = data.table(read.delim("./Plots_data/k_objective_evolution_hubway.csv.gz",
                           sep=","))
dd[, object_distribution := as.character(object_distribution)]
dd[, method := as.character(method)]
dd[, problem := as.character(problem)]


# Node k evolution plot for time=0
node.dd = dd[problem == "node"]
node.dd = node.dd[time == 0]
max_k = max(node.dd$k)
p1 = ggplot(data=node.dd, aes(x=k, y=objective_value))
p1 = p1 + geom_line(aes(color=method, group=method))
# p1 = p1 + geom_point()
p1 = p1 + scale_x_continuous(name="k", breaks=seq(0,max_k,10))
p1 = p1 + scale_y_continuous(name="Node Monitoring Objective")
p1 = p1 + facet_wrap(~object_distribution, nrow=1)
p1 = p1 + theme_bw()
p1 = p1 + theme(strip.background = element_blank())
p1
ggsave(paste0(figures.dir, "hubway_nodes_k_evolution.pdf"), w=12, h=4)

# Edge k evolution plot for time=0
edge.dd = dd[problem == "edge"]
edge.dd = edge.dd[time == 0]
max_k = max(edge.dd$k)
p2 = ggplot(data=edge.dd, aes(x=k, y=objective_value))
p2 = p2 + geom_line(aes(color=method, group=method))
# p2 = p2 + geom_point()
p2 = p2 + scale_x_continuous(name="k", breaks=seq(0,max_k,10))
p2 = p2 + scale_y_continuous(name="Edge Monitoring Objective")
p2 = p2 + facet_wrap(~object_distribution, nrow=1)
p2 = p2 + theme_bw()
p2 = p2 + theme(strip.background = element_blank())
p2
ggsave(paste0(figures.dir, "hubway_edges_k_evolution.pdf"), w=12, h=4)
