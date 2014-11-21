# Author(s): TJ Lane (tjlane@stanford.edu) and Christian Schwantes 
#            (schwancr@stanford.edu)
# Contributors: Vince Voelz, Kyle Beauchamp, Robert McGibbon
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.
"""
Functions for computing forward committors for an MSM. The forward 
committor is defined for a set of sources and sink states, and for
each state, the forward committor is the probability that a walker
starting at that state will visit the sink state before the source
state.

These are the canonical references for TPT. Note that TPT is really a
specialization of ideas very framiliar to the mathematical study of Markov
chains, and there are many books, manuscripts in the mathematical literature
that cover the same concepts.

References
----------
.. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory
       for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
       (2009).
.. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
       pathways in network models of coarse-grained protein dynamics. J. 
       Chem. Phys. 130, 205102 (2009).
"""
from __future__ import print_function, division, absolute_import
import numpy as np

import itertools
import copy

__all__ = ['committors', 'conditional_committors']

def committors(sources, sinks, msm):
    """
    Get the forward committors of the reaction sources -> sinks.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    msm : mixtape.MarkovStateModel
        MSM fit to the data.

    Returns
    -------
    forward_committors : np.ndarray
        The forward committors for the reaction sources -> sinks

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219 
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    sources = np.array(sources, dtype=int).reshape((-1, 1))
    sinks = np.array(sinks, dtype=int).reshape((-1, 1))

    n_states = msm.n_states_
    tprob = msm.transmat_

    # construct the committor problem
    lhs = np.eye(n_states) - tprob

    for a in sources:
        lhs[a, :] = 0.0  # np.zeros(n)
        lhs[:, a] = 0.0
        lhs[a, a] = 1.0

    for b in sinks:
        lhs[b, :] = 0.0  # np.zeros(n)
        lhs[:, b] = 0.0
        lhs[b, b] = 1.0

    ident_sinks = np.zeros(n_states)
    ident_sinks[sinks] = 1.0

    rhs = np.dot(tprob, ident_sinks)
    rhs[sources] = 0.0
    rhs[sinks] = 1.0

    forward_committors = np.linalg.solve(lhs, rhs)

    return forward_committors


def conditional_committors(source, sink, waypoint, msm):
    """
    Computes the conditional committors q^{ABC^+} which are is the probability
    of starting in one state and visiting state B before A while also visiting 
    state C at some point.

    Note that in the notation of Dickson et. al. this computes h_c(A,B), with
        sources   = A
        sinks     = B
        waypoint  = C

    Parameters
    ----------
    waypoint : int
        The index of the intermediate state
    source : int
        The index of the source state
    sink : int
        The index of the sink state
    msm : mixtape.MarkovStateModel
        MSM to analyze.

    Returns
    -------
    cond_committors : np.ndarray
        Conditional committors, i.e. the probability of visiting
        a waypoint when on a path between source and sink.

    See Also
    --------
    hubs.calculate_fraction_visits : function
        Calculate the fraction of visits to a waypoint from a given
        source to a sink.
    hubs.calculate_hub_score : function
        Compute the 'hub score', the weighted fraction of visits for an
        entire network.
    hubs.calculate_all_hub_scores : function
        Wrapper to compute all the hub scores in a network.

    Notes
    -----
    Employs dense linear algebra,
      memory use scales as N^2
      cycle use scales as N^3

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput.,
          Article ASAP DOI: 10.1021/ct300537s
    """

    tprob = msm.transmat_
    n_states = msm.n_states_

    # typecheck
    for data in [source, sink, waypoint]:
        if not isinstance(data, int):
            raise ValueError("source, sink, and waypoint must be integers.")
            
    if (source == waypoint) or (sink == waypoint) or (sink == source):
        raise ValueError('source, sink, waypoint must all be disjoint!')

    # hmmm this is awkward...
    forward_committors = committors([source], [sink], msm)

    # permute the transition matrix into cannonical form - send waypoint the the
    # last row, and source + sink to the end after that
    Bsink_indices = [source, sink, waypoint]
    perm = np.array([i for i in xrange(n_states) if not i in Bsink_indices], dtype=int)
    perm = np.concatenate([perm, Bsink_indices])
    permuted_tprob = tprob[perm, :][:, perm]

    # extract P, R
    n = n_states - len(Bsink_indices)
    P = permuted_tprob[:n, :n]
    R = permuted_tprob[:n, n:]

    # calculate the conditional committors ( B = N*R ), B[i,j] is the prob
    # state i ends in j, where j runs over the source + sink + waypoint
    # (waypoint is position -1)
    B = np.dot(np.linalg.inv(np.eye(n) - P), R)

    # add probs for the sinks, waypoint / b[i] is P( i --> {C & not A, B} )
    b = np.append(B[:, -1].flatten(), [0.0] * (len(Bsink_indices) - 1) + [1.0])
    cond_committors = b * forward_committors[waypoint]

    # get the original order
    cond_committors = cond_committors[np.argsort(perm)] 

    return cond_committors
