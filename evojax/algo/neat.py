import jax
import jax.numpy as jnp
import json
from evojax.algo.ind import Ind
from evojax.algo.utils import *
from evojax.algo.nsga_sort import nsga_sort
from evojax.algo.base import NEAlgorithm


class NEAT(NEAlgorithm):
    """NEAT main class. Evolves population given fitness values of individuals."""

    def __init__(self, hyp):
        """Intialize NEAT algorithm with hyperparameters
        Args:
          hyp - (dict) - algorithm hyperparameters

        Attributes:
          p       - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
          pop     - (Ind)      - Current population
          species - (Species)  - Current species
          innov   - (jnp.array) - innovation record
                    [5 X nUniqueGenes]
                    [0,:] == Innovation Number
                    [1,:] == Source
                    [2,:] == Destination
                    [3,:] == New Node?
                    [4,:] == Generation evolved
          gen     - (int)      - Current generation
        """
        self.p = hyp
        self.pop = []
        self.species = []
        self.innov = []
        self.gen = 0
        self.pop_size = hyp["popSize"]
        self.ann_nInput = hyp["ann_nInput"]
        self.ann_nOutput = hyp["ann_nOutput"]
        self._best_wMat = None
        self._best_aVec = None
        self.key = jax.random.PRNGKey(0)

    """ Subfunctions """
    from evojax.algo._variation import evolvePop, recombine
    from evojax.algo._speciate import (
        Species,
        speciate,
        compatDist,
        assignSpecies,
        assignOffspring,
    )

    def ask(self):
        """Returns newly evolved population"""
        if len(self.pop) == 0:
            self.initPop()  # Initialize population
        else:
            self.probMoo()  # Rank population according to objectives
            self.speciate()  # Divide population into species
            self.key, subkey = jax.random.split(self.key)
            self.key = self.evolvePop(subkey)  # Create child population

        # Find the maximum dimensions for wMat and aVec
        max_nodes = max(len(ind.wMat) for ind in self.pop)

        # Pre-allocate arrays for padded wMat and aVec
        padded_wMat = jnp.zeros((len(self.pop), max_nodes, max_nodes))
        padded_aVec = jnp.zeros((len(self.pop), max_nodes))

        # Pad wMat and aVec for each individual using vectorized operations
        for i, ind in enumerate(self.pop):
            padded_wMat = padded_wMat.at[i, : len(ind.wMat), : len(ind.wMat)].set(
                ind.wMat
            )
            padded_aVec = padded_aVec.at[i, : len(ind.aVec)].set(ind.aVec)

        return (
            jnp.array([len(ind.wMat) for ind in self.pop]),
            padded_wMat,
            padded_aVec,
        )

    def tell(self, fitness):
        """Assigns fitness to current population

        Args:
          fitness - (jnp.array) - fitness value of each individual
                   [nInd X 1]

        """
        # Vectorized assignment of fitness
        for ind, fit in zip(self.pop, fitness):
            ind.fitness = fit

        # Update best_params using jax operations
        best_index = jnp.argmax(fitness)
        self._best_wMat = self.pop[best_index].wMat
        self._best_aVec = self.pop[best_index].aVec

    def initPop(self):
        """Initialize population with a list of random individuals"""
        ##  Create base individual
        p = self.p  # readability

        # - Create Nodes -
        nodeId = jnp.arange(0, p["ann_nInput"] + p["ann_nOutput"] + 1)
        node = jnp.empty((3, len(nodeId)))
        node = node.at[0, :].set(nodeId)

        # Node types: [1:input, 2:hidden, 3:bias, 4:output]
        node = node.at[1, 0].set(4)  # Bias
        node = node.at[1, 1 : p["ann_nInput"] + 1].set(1)  # Input Nodes
        node = node.at[1, (p["ann_nInput"] + 1) :].set(2)  # Output Nodes

        # Node Activations
        node = node.at[2, :].set(p["ann_initAct"])

        # - Create Conns -
        nConn = (p["ann_nInput"] + 1) * p["ann_nOutput"]
        ins = jnp.arange(0, p["ann_nInput"] + 1)  # Input and Bias Ids
        outs = (p["ann_nInput"] + 1) + jnp.arange(0, p["ann_nOutput"])  # Output Ids

        conn = jnp.empty((5, nConn))
        conn = conn.at[0, :].set(jnp.arange(0, nConn))  # Connection Id
        conn = conn.at[1, :].set(jnp.tile(ins, len(outs)))  # Source Nodes
        conn = conn.at[2, :].set(jnp.repeat(outs, len(ins)))  # Destination Nodes
        conn = conn.at[3, :].set(jnp.nan)  # Weight Values
        conn = conn.at[4, :].set(1)  # Enabled?
        # Create population of individuals with varied weights
        pop = []
        for i in range(p["popSize"]):
            self.key, subkey = jax.random.split(self.key)
            newInd = Ind(conn, node)
            newInd.conn = newInd.conn.at[3, :].set(
                (2 * (jax.random.uniform(subkey, (nConn,)) - 0.5)) * p["ann_absWCap"]
            )
            self.key, subkey = jax.random.split(self.key)
            newInd.conn = newInd.conn.at[4, :].set(
                jax.random.uniform(subkey, (nConn,)) < p["prob_initEnable"]
            )
            newInd.express()
            newInd.birth = 0
            pop.append(newInd)

        # - Create Innovation Record -
        innov = jnp.zeros((5, nConn))
        innov = innov.at[0:3, :].set(pop[0].conn[0:3, :])
        innov = innov.at[3, :].set(-1)

        self.pop = pop
        self.innov = innov

    def probMoo(self):
        """Rank population according to Pareto dominance."""
        # Compile objectives
        meanFit = jnp.array([ind.fitness for ind in self.pop])
        nConns = jnp.array([ind.nConn for ind in self.pop])
        nConns = jnp.where(
            nConns == 0, 1, nConns
        )  # No connections is pareto optimal but boring...
        objVals = jnp.column_stack((meanFit, 1 / nConns))  # Maximize

        # Alternate between two objectives and single objective
        self.key, subkey = jax.random.split(self.key)
        if self.p["alg_probMoo"] < jax.random.uniform(subkey):
            rank = nsga_sort(objVals[:, [0, 1]])
        else:
            rank = rankArray(-objVals[:, 0])

        # Assign ranks
        for i in range(len(self.pop)):
            self.pop[i].rank = rank[i]

    @property
    def best_params(self) -> jnp.ndarray:
        return self._best_wMat, self._best_aVec

    @best_params.setter
    def best_params(self, wMat, aVec) -> None:
        self._best_wMat = wMat
        self._best_aVec = aVec
