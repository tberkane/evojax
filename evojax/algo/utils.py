import jax.numpy as jnp
import jax.random as random


def roulette(pArr, key):
    """Returns random index, with each choices chance weighted
    Args:
        pArr    - (jnp.array) - vector containing weighting of each choice
                  [N X 1]
        key     - (PRNGKey)   - random number generator key

    Returns:
        choice  - (int)       - chosen index
    """
    spin = random.uniform(key) * jnp.sum(pArr)
    cumsum = jnp.cumsum(pArr)
    choice = jnp.argmax(cumsum > spin)
    return choice


def listXor(b, c):
    """Returns elements in lists b and c they don't share"""
    return jnp.setxor1d(jnp.array(b), jnp.array(c))


def rankArray(X):
    """Returns ranking of a list, with ties resolved by first-found first-order
    NOTE: Sorts descending to follow jax conventions
    """
    tmp = jnp.argsort(-X)
    rank = jnp.zeros_like(tmp)
    rank = rank.at[tmp].set(jnp.arange(len(X)))
    return rank


def tiedRank(X):
    """Returns ranking of a list, with ties receiving an averaged rank"""
    sorter = jnp.argsort(-X)
    ranks = jnp.empty_like(sorter)
    ranks = ranks.at[sorter].set(jnp.arange(len(X)))

    mask = jnp.concatenate(([True], X[sorter][1:] != X[sorter][:-1], [True]))
    groups = jnp.arange(len(mask))[mask]
    ranks = ranks.at[sorter].set((groups[:-1] + groups[1:] - 1.0) / 2.0)

    return ranks


def bestIntSplit(ratio, total):
    """Divides a total into integer shares that best reflects ratio
    Args:
        share      - [1 X N ] - Percentage in each pile
        total      - [int   ] - Integer total to split

    Returns:
        intSplit   - [1 x N ] - Number in each pile
    """
    ratio = jnp.array(ratio)
    ratio = ratio / jnp.sum(ratio)

    floatSplit = ratio * total
    intSplit = jnp.floor(floatSplit)
    remainder = int(total - jnp.sum(intSplit))

    deserving = jnp.argsort(-(floatSplit - intSplit))

    intSplit = intSplit.at[deserving[:remainder]].add(1)
    return intSplit


def quickINTersect(A, B):
    """Faster set intersect: only valid for vectors of positive integers.
    (useful for matching indices)
    """
    if (len(A) == 0) or (len(B) == 0):
        return jnp.array([]), jnp.array([])

    P = jnp.zeros((1 + max(jnp.max(A), jnp.max(B))), dtype=bool)
    P = P.at[A].set(True)
    IB = P[B]
    P = P.at[A].set(False)
    P = P.at[B].set(True)
    IA = P[A]

    return IA, IB
