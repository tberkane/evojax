# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Sequence
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger


# -- ANN Ordering -------------------------------------------------------- -- #


def getNodeOrder(nodeG, connG):
    """Builds connection matrix from genome through topological sorting.

    Args:
      nodeG - (jnp.ndarray) - node genes
              [3 X nUniqueGenes]
              [0,:] == Node Id
              [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
              [2,:] == Activation function (as int)

      connG - (jnp.ndarray) - connection genes
              [5 X nUniqueGenes]
              [0,:] == Innovation Number (unique Id)
              [1,:] == Source Node Id
              [2,:] == Destination Node Id
              [3,:] == Weight Value
              [4,:] == Enabled?

    Returns:
      Q    - (jnp.ndarray) - sorted node order as indices
      wMat - (jnp.ndarray) - ordered weight matrix
             [N X N]

      OR

      False, False      - if cycle is found

    Todo:
      * setdiff1d is slow, as all numbers are positive ints is there a
        better way to do with indexing tricks (as in quickINTersect)?
    """
    conn = jnp.copy(connG)
    node = jnp.copy(nodeG)
    nIns = jnp.sum(node[1, :] == 1) + jnp.sum(node[1, :] == 4)
    nOuts = jnp.sum(node[1, :] == 2)

    # Create connection and initial weight matrices
    conn = conn.at[3, conn[4, :] == 0].set(jnp.nan)  # disabled but still connected
    src = conn[1, :].astype(jnp.int32)
    dest = conn[2, :].astype(jnp.int32)

    lookup = node[0, :].astype(jnp.int32)
    for i in range(len(lookup)):  # Can we vectorize this?
        src = jnp.where(src == lookup[i], i, src)
        dest = jnp.where(dest == lookup[i], i, dest)

    wMat = jnp.zeros((jnp.shape(node)[1], jnp.shape(node)[1]))
    wMat = wMat.at[src, dest].set(conn[3, :])
    connMat = wMat[nIns + nOuts :, nIns + nOuts :]
    connMat = jnp.where(connMat != 0, 1, connMat)

    # Topological Sort of Hidden Nodes
    edge_in = jnp.sum(connMat, axis=0)
    Q = jnp.where(edge_in == 0)[0]  # Start with nodes with no incoming connections
    for i in range(len(connMat)):
        if (len(Q) == 0) or (i >= len(Q)):
            Q = jnp.array([])
            return False, False  # Cycle found, can't sort
        edge_out = connMat[Q[i], :]
        edge_in = edge_in - edge_out  # Remove nodes' conns from total
        nextNodes = jnp.setdiff1d(jnp.where(edge_in == 0)[0], Q)
        Q = jnp.concatenate((Q, nextNodes))

        if jnp.sum(edge_in) == 0:
            break

    # Add In and outs back and reorder wMat according to sort
    Q = Q + nIns + nOuts
    Q = jnp.concatenate((lookup[:nIns], Q, lookup[nIns : nIns + nOuts]))
    wMat = wMat[jnp.ix_(Q, Q)]

    return Q, wMat


def getLayer(wMat):
    """Get layer of each node in weight matrix
    Traverse wMat by row, collecting layer of all nodes that connect to you (X).
    Your layer is max(X)+1. Input and output nodes are ignored and assigned layer
    0 and max(X)+1 at the end.

    Args:
      wMat  - (jnp.ndarray) - ordered weight matrix
             [N X N]

    Returns:
      layer - (jnp.ndarray) - layer # of each node

    Todo:
      * With very large networks this might be a performance sink -- especially,
      given that this happen in the serial part of the algorithm. There is
      probably a more clever way to do this given the adjacency matrix.
    """
    wMat = jnp.where(jnp.isnan(wMat), 0, wMat)
    wMat = jnp.where(wMat != 0, 1, wMat)
    nNode = jnp.shape(wMat)[0]
    layer = jnp.zeros((nNode))

    def loop_body(carry):
        layer, prevOrder = carry
        srcLayer = jnp.max(layer[:, None] * wMat, axis=0) + 1
        return (srcLayer, layer)

    def cond_fn(carry):
        layer, prevOrder = carry
        return jnp.any(prevOrder != layer)

    layer, _ = jax.lax.while_loop(cond_fn, loop_body, (layer, jnp.zeros_like(layer)))
    return layer - 1


def act(weights, aVec, inPattern, nNodes):
    nNodes_temp = weights.shape[0]

    def activation_step(i, nodeAct):
        rawAct = jnp.dot(nodeAct, weights[:, i])
        return nodeAct.at[i].set(applyAct(aVec[i], rawAct))

    nodeAct = jnp.zeros((nNodes_temp,))
    nodeAct = jax.lax.dynamic_update_slice(nodeAct, inPattern, (1,))

    nodeAct = jax.lax.fori_loop(12 + 1, nNodes_temp, activation_step, nodeAct)

    # return nodeAct[-3 - nNodes : -nNodes]
    return jax.lax.dynamic_slice(nodeAct, (-3 - nNodes,), (3,))


def applyAct(actId, x):
    return jax.lax.switch(
        actId.astype(int),
        [
            lambda: x,  # Linear
            lambda: jnp.where(x > 0, 1.0, 0.0),  # Unsigned Step Function
            lambda: jnp.sin(jnp.pi * x),  # Sin
            lambda: jnp.exp(-jnp.square(x) / 2.0),  # Gaussian
            lambda: jnp.tanh(x),  # Hyperbolic Tangent
            lambda: (jnp.tanh(x / 2.0) + 1.0) / 2.0,  # Sigmoid
            lambda: -x,  # Inverse
            lambda: jnp.abs(x),  # Absolute Value
            lambda: jnp.maximum(0, x),  # Relu
            lambda: jnp.cos(jnp.pi * x),  # Cosine
            lambda: jnp.square(x),  # Squared
        ],
    )


class ANNPolicy(PolicyNetwork):

    def __init__(self, logger: logging.Logger = None):
        if logger is None:
            self._logger = create_logger(name="ANNPolicy")
        else:
            self._logger = logger

    def get_actions(
        self,
        t_states: TaskState,
        nNodes: jnp.ndarray,
        weights: jnp.ndarray,
        aVec: jnp.ndarray,
        p_states: PolicyState,
    ) -> Tuple[jnp.ndarray, PolicyState]:
        outputs = []
        if len(nNodes.shape) == 0:
            truncated_weights = weights
            truncated_aVec = aVec
            output = act(truncated_weights, truncated_aVec, t_states.obs[0], nNodes)
            output = nn.softmax(output, axis=-1)
            outputs.append(output)
            outputs = jnp.stack(outputs, axis=0)
        elif len(nNodes.shape) == 1:
            for i in range(nNodes.shape[0]):
                truncated_weights = weights[i]
                truncated_aVec = aVec[i]
                output = act(
                    truncated_weights, truncated_aVec, t_states.obs[i], nNodes[i]
                )
                output = nn.softmax(output, axis=-1)
                outputs.append(output)
            outputs = jnp.stack(outputs, axis=0)
        else:
            for i in range(nNodes.shape[0]):
                outputs_i = []
                for j in range(nNodes.shape[1]):
                    truncated_weights = weights[i, j]
                    truncated_aVec = aVec[i, j]
                    output = act(
                        truncated_weights, truncated_aVec, t_states.obs[j], nNodes[i, j]
                    )
                    output = nn.softmax(output, axis=-1)
                    outputs_i.append(output)
                outputs.append(jnp.stack(outputs_i, axis=0))
            outputs = jnp.stack(outputs, axis=0)
        return outputs, p_states
