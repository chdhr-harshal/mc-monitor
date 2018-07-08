# Markov Chain Monitoring

<strong>Authors:</strong>
<ul>
<li> <a href="http://cs-people.bu.edu/harshal">Harshal A. Chaudhari</a> (Correspoding author) </li>
<li> <a href="https://michalis.co/">Michael Mathioudakis</a> </li>
<li> <a href="https://www.cs.bu.edu/~evimaria/">Evimaria Terzi</a> </li>
</ul>

In networking applications, one often wishes to obtain estimates about the number of objects at different parts of the network (e.g., the number of cars at an intersection of a road network or the number of packets expected to reach a node in a computer network) by monitoring the traffic in a small number of network nodes or edges. We formalize this task by defining the Markov Chain Monitoring problem. Given an initial distribution of items over the nodes of a Markov chain, we wish to estimate the distribution of items at subsequent times. We do this by asking a limited number of queries that retrieve, for example, how many items transitioned to a specific node or over a specific edge at a particular time. We consider different types of queries, each defining a different variant of the Markov Chain Monitoring. For each variant, we design efficient algorithms for choosing the queries that make our estimates as accurate as possible. In our experiments with synthetic and real datasets we demonstrate the efficiency and the efficacy of our algorithms in a variety of settings

<strong>Paper: </strong><a href="https://epubs.siam.org/doi/abs/10.1137/1.9781611975321.50">https://epubs.siam.org/doi/abs/10.1137/1.9781611975321.50</a>

<strong>Github:</strong> <a href="https://github.com/chdhr-harshal/mc-monitor">https://github.com/chdhr-harshal/mc-monitor</a>
