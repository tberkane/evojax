import jax.numpy as jnp
import warnings


def nsga_sort(objVals, returnFronts=False):
    """Returns ranking of objective values based on non-dominated sorting.
    Optionally returns fronts (useful for visualization).

    NOTE: Assumes maximization of objective function

    Args:
      objVals - (jnp.array) - Objective values of each individual
                [nInds X nObjectives]

    Returns:
      rank    - (jnp.array) - Rank in population of each individual
              int([nIndividuals X 1])
      front   - (jnp.array) - Pareto front of each individual
              int([nIndividuals X 1])

    Todo:
      * Extend to N objectives
    """
    fronts = getFronts(objVals)

    # Rank each individual in each front by crowding distance
    for f in range(len(fronts)):
        x1 = objVals[fronts[f], 0]
        x2 = objVals[fronts[f], 1]
        crowdDist = getCrowdingDist(x1) + getCrowdingDist(x2)
        frontRank = jnp.argsort(-crowdDist)
        fronts[f] = [fronts[f][i] for i in frontRank]

    # Convert to ranking
    tmp = jnp.array([ind for front in fronts for ind in front])
    rank = jnp.empty_like(tmp)
    rank = rank.at[tmp].set(jnp.arange(len(tmp)))

    if returnFronts is True:
        return rank, fronts
    else:
        return rank


def getFronts(objVals):
    """Fast non-dominated sort.

    Args:
      objVals - (jnp.array) - Objective values of each individual
                [nInds X nObjectives]

    Returns:
      front   - [list of lists] - One list for each front:
                                  list of indices of individuals in front

    Todo:
      * Extend to N objectives

    [adapted from: https://github.com/haris989/NSGA-II]
    """

    values1 = objVals[:, 0]
    values2 = objVals[:, 1]

    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = jnp.zeros(len(values1), dtype=jnp.int32)
    rank = jnp.zeros(len(values1), dtype=jnp.int32)
    # Get domination relations
    for p in range(0, len(values1)):
        S[p] = []
        for q in range(0, len(values1)):
            if (
                (values1[p] > values1[q] and values2[p] > values2[q])
                or (values1[p] >= values1[q] and values2[p] > values2[q])
                or (values1[p] > values1[q] and values2[p] >= values2[q])
            ):
                if q not in S[p]:
                    S[p].append(q)
            elif (
                (values1[q] > values1[p] and values2[q] > values2[p])
                or (values1[q] >= values1[p] and values2[q] > values2[p])
                or (values1[q] > values1[p] and values2[q] >= values2[p])
            ):
                n = n.at[p].add(1)
        if n[p] == 0:
            rank = rank.at[p].set(0)
            if p not in front[0]:
                front[0].append(p)

    # Assign fronts
    i = 0
    while front[i] != []:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n = n.at[q].add(-1)
                if n[q] == 0:
                    rank = rank.at[q].set(i + 1)
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)
    del front[len(front) - 1]
    return front


def getCrowdingDist(objVector):
    """Returns crowding distance of a vector of values, used once on each front.

    Note: Crowding distance of individuals at each end of front is infinite, as they don't have a neighbor.

    Args:
      objVector - (jnp.array) - Objective values of each individual
                  [nInds X nObjectives]

    Returns:
      dist      - (jnp.array) - Crowding distance of each individual
                  [nIndividuals X 1]
    """
    # Order by objective value
    key = jnp.argsort(objVector)
    sortedObj = objVector[key]

    # Distance from values on either side
    shiftVec = jnp.concatenate(
        [jnp.array([jnp.inf]), sortedObj, jnp.array([jnp.inf])]
    )  # Edges have infinite distance
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # inf on purpose
    prevDist = jnp.abs(sortedObj - shiftVec[:-2])
    nextDist = jnp.abs(sortedObj - shiftVec[2:])
    crowd = prevDist + nextDist
    if (sortedObj[-1] - sortedObj[0]) > 0:
        crowd *= abs((1 / (sortedObj[-1] - sortedObj[0])))  # Normalize by fitness range

    # Restore original order
    dist = jnp.empty(len(key))
    dist = dist.at[key].set(crowd)

    return dist
