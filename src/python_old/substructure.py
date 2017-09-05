#!/usr/local/bin/python

"""
Extends the MarkovChain class from simulation.py to verify optimal substructure
property
"""

from simulation import *

class Substructure(MarkovChain):
    def __init__(self, num_nodes, num_objects, time_left,
            transition_matrix, initial_object_distribution, removal_node_id):
        super(Substructure, self).__init__(num_nodes, num_objects, time_left,
                transition_matrix, initial_object_distribution)
        self.removal_node_id = removal_node_id
        self.remove_incoming_objects(self.removal_node_id)
        self.remove_incoming_edge_weights(self.removal_node_id)

    def remove_incoming_edge_weights(self, removal_node_id):
        """
        Removes all incoming edges to a node
        and balances the outgoing edge weights of other nodes
        to maintain the stochasticity
        """

        self.updated_transition_matrix = self.transition_matrix.copy()
        # Set the column for the node to be removed to 0
        # i.e., set all incoming probabilities to 0
        for row in xrange(self.num_nodes):
            self.transition_matrix[row][removal_node_id] = 0

        # Divide each row by its sum to maintain stochasticity
        self.transition_matrix = self.transition_matrix/self.transition_matrix.sum(axis=1)[:,None]
        self.G = nx.from_numpy_matrix(self.transition_matrix, create_using=nx.DiGraph())

    def remove_incoming_objects(self, removal_node_id):
        """
        Remove the expected number of objects that would cross
        the incoming edges into the removal_node_id
        """
        incoming_edges = filter(lambda(x,y):y==removal_node_id, self.G.edges())
        parent_nodes = [x for (x,y) in incoming_edges]

        # Remove expected number of incoming objects into removal_node_id
        # from the parent nodes
        for node_id in parent_nodes:
            current_objects = self.nodes_list[node_id].num_objects
            edge_weight = self.G.get_edge_data(node_id, removal_node_id)['weight']
            self.nodes_list[node_id].remove_objects(current_objects*edge_weight)
            self.num_objects -= current_objects*edge_weight
            self.current_object_distribution[node_id] = self.nodes_list[node_id].num_objects

if __name__ == "__main__":
    mc = MarkovChain(6, 600, 1000, None, None)
    print "Created markov chain"

    for i in reversed(xrange(mc.time_left)):
        print "Time left: {}\n".format(mc.time_left)

        greedy_nodes = mc.get_greedy_nodes(2)
        value = mc.get_greedy_nodes_algorithm_value(2)
        print "Objective value f_G({}) = {}".format(greedy_nodes, value)

        greedy_node = mc.get_greedy_nodes(1)
        print "First greedy node in G is {}".format(greedy_node)
        print "\n"
        print "Updating G to G1 by removing node {}".format(greedy_node)
        sb = Substructure(mc.num_nodes,
                        mc.num_objects,
                        mc.time_left,
                        mc.transition_matrix,
                        mc.current_object_distribution.values(),
                        greedy_node[0])
        print "\n"
        greedy_node = sb.get_greedy_nodes(1)
        print "First greedy node in G1 is {}".format(greedy_node)

        value = sb.get_greedy_nodes_algorithm_value(1)
        print "Objective function f_G1({}) = {}".format(greedy_node, value)

        if greedy_nodes[1] != greedy_node[0]:
            print "\n******************\nHYPOTHESIS WRONG\n******************"
            raw_input()

        print "\n===================================================================="
        mc.simulate_time_step()

# if __name__ == "__main__":
#     mc = MarkovChain(10, 100, 100, None, None)
#     print "Created markov chain"
#     k = 5
#
#     for i in reversed(xrange(mc.time_left)):
#         for k_prime in range(1,k):
#             brute_nodes_k_1 = mc.get_brute_force_nodes_sets(k_prime-1)
#             brute_nodes_k = mc.get_brute_force_nodes_sets(k_prime)
#
#             flag = False
#             for min_set in brute_nodes_k:
#                 flag = set(brute_nodes_k_1[0])<set(min_set)
#                 if flag:
#                     break
#             if not flag:
#                 print "\n******************\nHYPOTHESIS WRONG\n******************"
#                 print "Optimal nodes size  {} = {}".format(k_prime -1, brute_nodes_k_1)
#                 print "Optimal nodes size {} = {}".format(k_prime, brute_nodes_k)
#                 raw_input()
#         print mc
#         mc.simulate_time_step()
