import jax.numpy as jnp
import jax.random as random
import jax

from evojax.algo.utils import *
from evojax.algo.hyp import hyp


def evolvePop(species, innov, gen, key):
    newPop = []
    for i in range(len(species)):
        children, innov, key = recombine(species[i], innov, gen, key)
        newPop.append(children)
    pop = [ind for species in newPop for ind in species]

    return pop, innov, key


def recombine(species, innov, gen, key):
    """Creates next generation of child solutions from a species

    Procedure:
      ) Sort all individuals by rank
      ) Eliminate lower percentage of individuals from breeding pool
      ) Pass upper percentage of individuals to child population unchanged
      ) Select parents by tournament selection
      ) Produce new population through crossover and mutation

    Args:
        species - (Species) -
          .members    - [Ind] - parent population
          .nOffspring - (int) - number of children to produce
        innov   - (jnp.array)  - innovation record
                  [5 X nUniqueGenes]
                  [0,:] == Innovation Number
                  [1,:] == Source
                  [2,:] == Destination
                  [3,:] == New Node?
                  [4,:] == Generation evolved
        gen     - (int) - current generation
        key     - (jax.random.PRNGKey) - random number generator
    Returns:
        children - [Ind]      - newly created population
        innov   - (jnp.array)  - updated innovation record

    """
    p = hyp
    nOffspring = int(species.nOffspring)
    pop = species.members
    children = []

    # Sort by rank
    pop = sorted(pop, key=lambda x: x.rank)

    # Cull  - eliminate worst individuals from breeding pool
    numberToCull = int(jnp.floor(p["select_cullRatio"] * len(pop)))
    if numberToCull > 0:
        pop = pop[:-numberToCull]

    # Elitism - keep best individuals unchanged
    nElites = int(jnp.floor(len(pop) * p["select_eliteRatio"]))
    children.extend(pop[:nElites])
    nOffspring -= nElites

    # Get parent pairs via tournament selection
    # -- As individuals are sorted by fitness, index comparison is
    # enough. In the case of ties the first individual wins
    key, subkey = random.split(key)
    parentA = random.randint(subkey, (nOffspring, p["select_tournSize"]), 0, len(pop))
    key, subkey = random.split(key)
    parentB = random.randint(subkey, (nOffspring, p["select_tournSize"]), 0, len(pop))
    parents = jnp.vstack((jnp.min(parentA, axis=1), jnp.min(parentB, axis=1)))
    parents = jnp.sort(parents, axis=0)  # Higher fitness parent first

    # Breed child population
    for i in range(nOffspring):
        key, subkey = random.split(key)
        if random.uniform(subkey) > p["prob_crossover"]:
            # Mutation only: take only highest fit parent
            child, innov, key = pop[parents[0, i]].createChild(p, innov, key, gen=gen)
        else:
            # Crossover
            child, innov, key = pop[parents[0, i]].createChild(
                p, innov, key, gen=gen, mate=pop[parents[1, i]]
            )

        child.express()
        children.append(child)

    return children, innov, key
