import jax.numpy as jnp
import jax.random as random
from evojax.policy.ann import getLayer, getNodeOrder
from jax import tree_util


class Ind:
    """Individual class: genes, network, and fitness"""

    def __init__(self, conn, node):
        """Intialize individual with given genes
        Args:
          conn - [5 X nUniqueGenes]
                 [0,:] == Innovation Number
                 [1,:] == Source
                 [2,:] == Destination
                 [3,:] == Weight
                 [4,:] == Enabled?
          node - [3 X nUniqueGenes]
                 [0,:] == Node Id
                 [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                 [2,:] == Activation function (as int)

        Attributes:
          node    - (jnp.array) - node genes (see args)
          conn    - (jnp.array) - conn genes (see args)
          nInput  - (int)       - number of inputs
          nOutput - (int)       - number of outputs
          wMat    - (jnp.array) - weight matrix, one row and column for each node
                    [N X N]     - rows: connection from; cols: connection to
          wVec    - (jnp.array) - wMat as a flattened vector
                    [N**2 X 1]
          aVec    - (jnp.array) - activation function of each node (as int)
                    [N X 1]
          nConn   - (int)       - number of connections
          fitness - (float)     - fitness averaged over all trials (higher better)
          rank    - (int)       - rank in population (lower better)
          birth   - (int)       - generation born
          species - (int)       - ID of species
        """
        self.node = jnp.array(node)
        self.conn = jnp.array(conn)
        self.nInput = jnp.sum(node[1, :] == 1)
        self.nOutput = jnp.sum(node[1, :] == 2)
        self.wMat = jnp.array([])
        self.wVec = jnp.array([])
        self.aVec = jnp.array([])
        self.nConn = 0
        self.fitness = 0.0
        self.rank = 0
        self.birth = 0
        self.species = 0

    def nConns(self):
        """Returns number of active connections"""
        return int(jnp.sum(self.conn[4, :]))

    def express(self):
        """Converts genes to weight matrix and activation vector"""
        order, wMat = getNodeOrder(self.node, self.conn)
        if order is not False:
            self.wMat = wMat
            self.aVec = self.node[2, order]

            wVec = self.wMat.flatten()
            wVec = jnp.where(jnp.isnan(wVec), 0, wVec)
            self.wVec = wVec
            self.nConn = jnp.sum(wVec != 0)
            return True
        else:
            return False

    def createChild(self, p, innov, key, gen=0, mate=None):
        """Create new individual with this individual as a parent

          Args:
            p      - (dict)      - algorithm hyperparameters (see p/hypkey.txt)
            innov  - (jnp.array) - innovation record
               [5 X nUniqueGenes]
               [0,:] == Innovation Number
               [1,:] == Source
               [2,:] == Destination
               [3,:] == New Node?
               [4,:] == Generation evolved
            gen    - (int)       - (optional) generation (for innovation recording)
            mate   - (Ind)       - (optional) second for individual for crossover


        Returns:
            child  - (Ind)       - newly created individual
            innov  - (jnp.array) - updated innovation record

        """
        if mate is not None:
            child, key = self.crossover(mate, key)
        else:
            child = Ind(self.conn, self.node)

        child, innov, key = child.mutate(p, key, innov, gen)
        return child, innov, key

    # -- Canonical NEAT recombination operators ------------------------------ -- #

    def crossover(self, mate, key):
        """Combine genes of two individuals to produce new individual

          Procedure:
          ) Inherit all nodes and connections from most fit parent
          ) Identify matching connection genes in parentA and parentB
          ) Replace weights with parentB weights with some probability

          Args:
            parentA  - (Ind) - Fittest parent
              .conns - (jnp.array) - connection genes
                       [5 X nUniqueGenes]
                       [0,:] == Innovation Number (unique Id)
                       [1,:] == Source Node Id
                       [2,:] == Destination Node Id
                       [3,:] == Weight Value
                       [4,:] == Enabled?
            parentB - (Ind) - Less fit parent

        Returns:
            child   - (Ind) - newly created individual

        """
        parentA = self
        parentB = mate

        # Inherit all nodes and connections from most fit parent
        child = Ind(parentA.conn, parentA.node)

        # Identify matching connection genes in ParentA and ParentB
        aConn = parentA.conn[0, :]
        bConn = parentB.conn[0, :]
        matching, IA, IB = jnp.intersect1d(aConn, bConn, return_indices=True)

        # Replace weights with parentB weights with some probability
        bProb = 0.5
        key, subkey = random.split(key)
        bGenes = random.uniform(subkey, shape=(1, len(matching))) < bProb
        child.conn = child.conn.at[3, IA[bGenes[0]]].set(parentB.conn[3, IB[bGenes[0]]])

        return child, key

    def mutate(self, p, key, innov=None, gen=None):
        """Randomly alter topology and weights of individual

        Args:
          p        - (dict)      - algorithm hyperparameters (see p/hypkey.txt)
          child    - (Ind) - individual to be mutated
            .conns - (jnp.array) - connection genes
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number (unique Id)
                     [1,:] == Source Node Id
                     [2,:] == Destination Node Id
                     [3,:] == Weight Value
                     [4,:] == Enabled?
            .nodes - (jnp.array) - node genes
                     [3 X nUniqueGenes]
                     [0,:] == Node Id
                     [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                     [2,:] == Activation function (as int)
          innov    - (jnp.array) - innovation record
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number
                     [1,:] == Source
                     [2,:] == Destination
                     [3,:] == New Node?
                     [4,:] == Generation evolved

        Returns:
            child   - (Ind)       - newly created individual
            innov   - (jnp.array) - innovation record

        """
        # Readability
        nConn = self.conn.shape[1]
        connG = self.conn
        nodeG = self.node

        # - Re-enable connections
        disabled = jnp.where(connG[4, :] == 0)[0]
        if len(disabled) > 0:
            key, subkey = random.split(key)
            reenabled = (
                random.uniform(subkey, shape=(len(disabled),)) < p["prob_enable"]
            )
            connG = connG.at[4, disabled].set(reenabled)

        # - Weight mutation
        key, subkey = random.split(key)
        mutatedWeights = random.uniform(subkey, shape=(1, nConn)) < p["prob_mutConn"]
        key, subkey = random.split(key)
        weightChange = (
            mutatedWeights * random.normal(subkey, shape=(1, nConn)) * p["ann_mutSigma"]
        )
        connG = connG.at[3, :].add(weightChange[0])

        # Clamp weight strength
        connG = connG.at[3, :].set(
            jnp.clip(connG[3, :], -p["ann_absWCap"], p["ann_absWCap"])
        )

        key, subkey = random.split(key)
        r = random.uniform(subkey)
        if (r < p["prob_addNode"]) and jnp.any(connG[4, :] == 1):
            # print("mutAddNode")
            connG, nodeG, innov, key = self.mutAddNode(connG, nodeG, innov, gen, p, key)

        key, subkey = random.split(key)
        if random.uniform(subkey) < p["prob_addConn"]:
            # print("mutAddConn")
            connG, innov, key = self.mutAddConn(connG, nodeG, innov, gen, p, key)

        child = Ind(connG, nodeG)
        child.birth = gen

        return child, innov, key

    def mutAddNode(self, connG, nodeG, innov, gen, p, key):
        """Add new node to genome

        Args:
          connG    - (jnp.array) - connection genes
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number (unique Id)
                     [1,:] == Source Node Id
                     [2,:] == Destination Node Id
                     [3,:] == Weight Value
                     [4,:] == Enabled?
          nodeG    - (jnp.array) - node genes
                     [3 X nUniqueGenes]
                     [0,:] == Node Id
                     [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                     [2,:] == Activation function (as int)
          innov    - (jnp.array) - innovation record
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number
                     [1,:] == Source
                     [2,:] == Destination
                     [3,:] == New Node?
                     [4,:] == Generation evolved
          gen      - (int) - current generation
          p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


        Returns:
          connG    - (jnp.array) - updated connection genes
          nodeG    - (jnp.array) - updated node genes
          innov    - (jnp.array) - updated innovation record

        """
        if innov is None:
            newNodeId = int(jnp.max(nodeG[0, :]) + 1)
            newConnId = connG[0, -1] + 1
        else:
            newNodeId = int(jnp.max(innov[2, :]) + 1)
            newConnId = innov[0, -1] + 1

        # Choose connection to split
        connActive = jnp.where(connG[4, :] == 1)[0]
        if len(connActive) < 1:
            return connG, nodeG, innov  # No active connections, nothing to split
        key, subkey = random.split(key)
        connSplit = connActive[
            random.randint(subkey, shape=(), minval=0, maxval=len(connActive))
        ]

        # Create new node
        key, subkey = random.split(key)
        newActivation = p["ann_actRange"][
            random.randint(subkey, shape=(), minval=0, maxval=len(p["ann_actRange"]))
        ]
        newNode = jnp.array([[newNodeId, 3, newActivation]]).T

        # Add connections to and from new node
        connTo = connG[:, connSplit].copy()
        connTo = connTo.at[0].set(newConnId)
        connTo = connTo.at[2].set(newNodeId)
        connTo = connTo.at[3].set(1)  # weight set to 1

        connFrom = connG[:, connSplit].copy()
        connFrom = connFrom.at[0].set(newConnId + 1)
        connFrom = connFrom.at[1].set(newNodeId)
        connFrom = connFrom.at[3].set(
            connG[3, connSplit]
        )  # weight set to previous weight value

        newConns = jnp.vstack((connTo, connFrom)).T

        # Disable original connection
        connG = connG.at[4, connSplit].set(0)

        # Record innovations
        if innov is not None:
            newInnov = jnp.empty((5, 2))
            newInnov = newInnov.at[:, 0].set(jnp.hstack((connTo[0:3], newNodeId, gen)))
            newInnov = newInnov.at[:, 1].set(jnp.hstack((connFrom[0:3], -1, gen)))
            innov = jnp.hstack((innov, newInnov))

        # Add new structures to genome
        nodeG = jnp.hstack((nodeG, newNode))
        connG = jnp.hstack((connG, newConns))

        return connG, nodeG, innov, key

    def mutAddConn(self, connG, nodeG, innov, gen, p, key):
        """Add new connection to genome.
        To avoid creating recurrent connections all nodes are first sorted into
        layers, connections are then only created from nodes to nodes of the same or
        later layers.


        Todo: check for preexisting innovations to avoid duplicates in same gen

        Args:
          connG    - (jnp.array) - connection genes
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number (unique Id)
                     [1,:] == Source Node Id
                     [2,:] == Destination Node Id
                     [3,:] == Weight Value
                     [4,:] == Enabled?
          nodeG    - (jnp.array) - node genes
                     [3 X nUniqueGenes]
                     [0,:] == Node Id
                     [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                     [2,:] == Activation function (as int)
          innov    - (jnp.array) - innovation record
                     [5 X nUniqueGenes]
                     [0,:] == Innovation Number
                     [1,:] == Source
                     [2,:] == Destination
                     [3,:] == New Node?
                     [4,:] == Generation evolved
          gen      - (int)      - current generation
          p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


        Returns:
          connG    - (jnp.array) - updated connection genes
          innov    - (jnp.array) - updated innovation record

        """
        if innov is None:
            newConnId = connG[0, -1] + 1
        else:
            newConnId = innov[0, -1] + 1

        nIns = jnp.sum(nodeG[1, :] == 1) + jnp.sum(
            nodeG[1, :] == 4
        )  # 12 inputs + 1 bias
        nOuts = jnp.sum(nodeG[1, :] == 2)  # 3 outputs
        order, wMat = getNodeOrder(nodeG, connG)  # Topological Sort of Network

        hMat = wMat[nIns:-nOuts, nIns:-nOuts]
        if len(hMat) > 0:
            hLay = getLayer(hMat) + 1
        else:
            hLay = jnp.array([])

        # To avoid recurrent connections nodes are sorted into layers, and connections are only allowed from lower to higher layers
        if len(hLay) > 0:
            lastLayer = jnp.max(hLay) + 1
        else:
            lastLayer = 1
        L = jnp.concatenate([jnp.zeros(nIns), hLay, jnp.full((nOuts), lastLayer)])
        nodeKey = jnp.column_stack((nodeG[0, order], L))  # Assign Layers

        key, subkey = random.split(key)
        sources = random.permutation(subkey, len(nodeKey))
        for src in sources:
            srcLayer = nodeKey[src, 1]
            dest = jnp.where(nodeKey[:, 1] > srcLayer)[0]

            # Finding already existing connections:
            #   ) take all connection genes with this source (connG[1,:])
            #   ) take the destination of those genes (connG[2,:])
            #   ) convert to nodeKey index (Gotta be a better jax way...)
            srcIndx = jnp.where(connG[1, :] == nodeKey[src, 0])[0]
            exist = connG[2, srcIndx]
            existKey = []
            for iExist in exist:
                existKey.append(jnp.where(nodeKey[:, 0] == iExist)[0])
            dest = jnp.setdiff1d(
                dest, jnp.array(existKey)
            )  # Remove existing connections

            # Add a random valid connection
            key, subkey = random.split(key)
            dest = random.permutation(subkey, dest)
            if len(dest) > 0:  # (there is a valid connection)
                connNew = jnp.empty((5, 1))
                connNew = connNew.at[0].set(newConnId)
                connNew = connNew.at[1].set(nodeKey[src, 0])
                connNew = connNew.at[2].set(nodeKey[dest[0], 0])
                key, subkey = random.split(key)
                connNew = connNew.at[3].set(
                    (random.uniform(subkey) - 0.5) * 2 * p["ann_absWCap"]
                )
                connNew = connNew.at[4].set(1)
                connG = jnp.hstack((connG, connNew))

                # Record innovation
                if innov is not None:
                    newInnov = jnp.hstack((connNew[0:3].flatten(), -1, gen))
                    innov = jnp.hstack((innov, newInnov[:, None]))
                break

        return connG, innov, key


def flatten_ind(obj):
    children = (
        obj.conn,
        obj.node,
        obj.nInput,
        obj.nOutput,
        obj.wMat,
        obj.wVec,
        obj.aVec,
        obj.nConn,
        obj.fitness,
        obj.rank,
        obj.birth,
        obj.species,
    )
    aux_data = None
    return children, aux_data


def unflatten_ind(aux_data, children):
    n = Ind(children[0], children[1])
    n.nInput = children[2]
    n.nOutput = children[3]
    n.wMat = children[4]
    n.wVec = children[5]
    n.aVec = children[6]
    n.nConn = children[7]
    n.fitness = children[8]
    n.rank = children[9]
    n.birth = children[10]
    n.species = children[11]

    return n


tree_util.register_pytree_node(Ind, flatten_ind, unflatten_ind)
