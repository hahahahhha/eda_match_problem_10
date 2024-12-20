from networkx import graph_edit_distance
import networkx as nx
from torch.utils.data import DataLoader
import os
import dgl
import torch as th

import re
# 该版本的utils不区分drain和source

# cython: profile=True

import networkx as nx
import time
from dataclasses import dataclass
from functools import lru_cache as cache

def my_graph_edit_distance(G1, G2, node_match, edge_match, timeout, task_name):

    bestcost = []
    start = time.time()
    i = 0
    for _, _, cost in my_optimize_edit_paths(
            G1,
            G2,
            node_match,
            edge_match,
            timeout=timeout,
    ):
        # assert bestcost is None or cost < bestcost
        i += 1
        bestcost.append(cost)
        print(f"Name {task_name}, Cost: {cost}, Iter: {i}, Time: {time.time() - start:.2f}/{timeout}")

    print(f"Finished Name {task_name}, Cost: {bestcost[-1]}, Iter: {i}, Time: {time.time() - start:.2f}/{timeout}")
    return bestcost[-1], bestcost


def my_optimize_edit_paths(
        G1,
        G2,
        node_match=None,
        edge_match=None,
        node_subst_cost=None,
        node_del_cost=None,
        node_ins_cost=None,
        edge_subst_cost=None,
        edge_del_cost=None,
        edge_ins_cost=None,
        upper_bound=None,
        strictly_decreasing=True,
        roots=None,
        timeout=None,
):
    """GED (graph edit distance) calculation: advanced interface.

    Graph edit path is a sequence of node and edge edit operations
    transforming graph G1 to graph isomorphic to G2.  Edit operations
    include substitutions, deletions, and insertions.

    Graph edit distance is defined as minimum cost of edit path.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be of the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2
        should be considered equal during matching.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute
        dictionaries for n1 and n2 as inputs.

        Ignored if node_subst_cost is specified.  If neither
        node_match nor node_subst_cost are specified then node
        attributes are not considered.

    edge_match : callable
        A function that returns True if the edge attribute dictionaries
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during matching.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute
        dictionaries of the edges under consideration.

        Ignored if edge_subst_cost is specified.  If neither
        edge_match nor edge_subst_cost are specified then edge
        attributes are not considered.

    node_subst_cost, node_del_cost, node_ins_cost : callable
        Functions that return the costs of node substitution, node
        deletion, and node insertion, respectively.

        The functions will be called like

           node_subst_cost(G1.nodes[n1], G2.nodes[n2]),
           node_del_cost(G1.nodes[n1]),
           node_ins_cost(G2.nodes[n2]).

        That is, the functions will receive the node attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function node_subst_cost overrides node_match if specified.
        If neither node_match nor node_subst_cost are specified then
        default node substitution cost of 0 is used (node attributes
        are not considered during matching).

        If node_del_cost is not specified then default node deletion
        cost of 1 is used.  If node_ins_cost is not specified then
        default node insertion cost of 1 is used.

    edge_subst_cost, edge_del_cost, edge_ins_cost : callable
        Functions that return the costs of edge substitution, edge
        deletion, and edge insertion, respectively.

        The functions will be called like

           edge_subst_cost(G1[u1][v1], G2[u2][v2]),
           edge_del_cost(G1[u1][v1]),
           edge_ins_cost(G2[u2][v2]).

        That is, the functions will receive the edge attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function edge_subst_cost overrides edge_match if specified.
        If neither edge_match nor edge_subst_cost are specified then
        default edge substitution cost of 0 is used (edge attributes
        are not considered during matching).

        If edge_del_cost is not specified then default edge deletion
        cost of 1 is used.  If edge_ins_cost is not specified then
        default edge insertion cost of 1 is used.

    upper_bound : numeric
        Maximum edit distance to consider.

    strictly_decreasing : bool
        If True, return consecutive approximations of strictly
        decreasing cost.  Otherwise, return all edit paths of cost
        less than or equal to the previous minimum cost.

    roots : 2-tuple
        Tuple where first element is a node in G1 and the second
        is a node in G2.
        These nodes are forced to be matched in the comparison to
        allow comparison between rooted graphs.

    timeout : numeric
        Maximum number of seconds to execute.
        After timeout is met, the current best GED is returned.

    Returns
    -------
    Generator of tuples (node_edit_path, edge_edit_path, cost)
        node_edit_path : list of tuples (u, v)
        edge_edit_path : list of tuples ((u1, v1), (u2, v2))
        cost : numeric

    See Also
    --------
    graph_edit_distance, optimize_graph_edit_distance, optimal_edit_paths

    References
    ----------
    .. [1] Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, Patrick
       Martineau. An Exact Graph Edit Distance Algorithm for Solving
       Pattern Recognition Problems. 4th International Conference on
       Pattern Recognition Applications and Methods 2015, Jan 2015,
       Lisbon, Portugal. 2015,
       <10.5220/0005209202710278>. <hal-01168816>
       https://hal.archives-ouvertes.fr/hal-01168816

    """
    # TODO: support DiGraph

    import numpy as np
    import scipy as sp

    @dataclass
    class CostMatrix:
        C: ...
        lsa_row_ind: ...
        lsa_col_ind: ...
        ls: ...


    def make_CostMatrix(C, m, n):
        # assert(C.shape == (m + n, m + n))
        lsa_row_ind, lsa_col_ind = sp.optimize.linear_sum_assignment(C)

        # Fixup dummy assignments:
        # each substitution i<->j should have dummy assignment m+j<->n+i
        # NOTE: fast reduce of Cv relies on it
        # Create masks for substitution and dummy indices
        is_subst = (lsa_row_ind < m) & (lsa_col_ind < n)
        is_dummy = (lsa_row_ind >= m) & (lsa_col_ind >= n)

        # Map dummy assignments to the correct indices
        lsa_row_ind[is_dummy] = lsa_col_ind[is_subst] + m
        lsa_col_ind[is_dummy] = lsa_row_ind[is_subst] + n

        return CostMatrix(
            C, lsa_row_ind, lsa_col_ind, C[lsa_row_ind, lsa_col_ind].sum()
        )


    def extract_C(C, i, j, m, n):
        # assert(C.shape == (m + n, m + n))
        row_ind = [k in i or k - m in j for k in range(m + n)]
        col_ind = [k in j or k - n in i for k in range(m + n)]
        return C[row_ind, :][:, col_ind]


    def reduce_C(C, i, j, m, n):
        # assert(C.shape == (m + n, m + n))

        col_ind, row_ind = cached_ind(i, j, m, n)
        return C[row_ind, :][:, col_ind]

    @cache
    def cached_ind(i, j, m, n):
        row_ind = [k not in i and k - m not in j for k in range(m + n)]
        col_ind = [k not in j and k - n not in i for k in range(m + n)]
        return col_ind, row_ind

    def reduce_ind(ind, i):
        # assert set(ind) == set(range(len(ind)))
        rind = ind[[k not in i for k in ind]]
        for k in set(i):
            rind[rind >= k] -= 1
        return rind

    def match_edges(u, v, pending_g, pending_h, Ce, matched_uv=None):
        """
        Parameters:
            u, v: matched vertices, u=None or v=None for
               deletion/insertion
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_uv: partial vertex edit path
                list of tuples (u, v) of previously matched vertex
                    mappings u<->v, u=None or v=None for
                    deletion/insertion

        Returns:
            list of (i, j): indices of edge mappings g<->h
            localCe: local CostMatrix of edge mappings
                (basically submatrix of Ce at cross of rows i, cols j)
        """
        M = len(pending_g)
        N = len(pending_h)

        matched_set_p_p = {(p, p) for p, q in matched_uv}
        matched_set_q_q = {(q, q) for p, q in matched_uv}
        matched_set_p_u = {(p, u) for p, q in matched_uv}
        matched_set_u_p = {(u, p) for p, q in matched_uv}
        matched_set_q_v = {(q, v) for p, q in matched_uv}
        matched_set_v_q = {(v, q) for p, q in matched_uv}
        matched_set_u_u = {(u, u)}
        matched_set_v_v = {(v, v)}
        # matched_set_p_q = {(p, q) for p, q in matched_uv}
        # matched_set_q_p = {(q, p) for p, q in matched_uv}
        matched_set_p_u_q_v = {(p, u, q, v) for p, q in matched_uv}

        # assert Ce.C.shape == (M + N, M + N)

        # only attempt to match edges after one node match has been made
        # this will stop self-edges on the first node being automatically deleted
        # even when a substitution is the better option
        if matched_uv is None or len(matched_uv) == 0:
            g_ind = []
            h_ind = []
        else:

            matched_set_g = matched_set_p_u | matched_set_u_p | matched_set_p_p | matched_set_u_u
            matched_set_h = matched_set_q_v | matched_set_v_q | matched_set_q_q | matched_set_v_v

            g_ind = [
                i
                for i in range(M)
                if pending_g[i][:2] in matched_set_g
            ]
            h_ind = [
                j
                for j in range(N)
                if pending_h[j][:2] in matched_set_h
            ]
        m = len(g_ind)
        n = len(h_ind)

        if m or n:
            C = extract_C(Ce.C, g_ind, h_ind, M, N)
            # assert C.shape == (m + n, m + n)

            # Forbid structurally invalid matches
            # NOTE: inf remembered from Ce construction
            for k, i in enumerate(g_ind):
                g = pending_g[i][:2]
                for l, j in enumerate(h_ind):
                    h = pending_h[j][:2]
                    if nx.is_directed(G1) or nx.is_directed(G2):
                        if any(
                                g == (p, u) and h == (q, v) or g == (u, p) and h == (v, q)
                                for p, q in matched_uv
                        ):
                            continue
                    else:
                        g1, g2 = g
                        h1, h2 = h
                        any11 = (g1, g2, h1, h2) in matched_set_p_u_q_v
                        any12 = (g1, g2, h2, h1) in matched_set_p_u_q_v
                        any13 = (g2, g1, h1, h2) in matched_set_p_u_q_v
                        any14 = (g2, g1, h2, h1) in matched_set_p_u_q_v

                        any1 = any11 or any12 or any13 or any14
                        # any2 = any(
                        #         g in ((p, u), (u, p)) and h in ((q, v), (v, q))
                        #         for p, q in matched_uv
                        # )
                        # assert any1 == any2

                        if any1:
                            continue

                    any1 = (g in matched_set_u_u | matched_set_p_p)
                    # any2 = (g == (u, u) or any(g == (p, p) for p, q in matched_uv))
                    # assert any1 == any2
                    if any1:
                        continue

                    any1 = (h in matched_set_v_v | matched_set_q_q)
                    # any2 = (h == (v, v) or any(h == (q, q) for p, q in matched_uv))
                    # assert any1 == any2
                    if any1:
                        continue
                    C[k, l] = inf

            localCe = make_CostMatrix(C, m, n)
            ij = [
                (
                    g_ind[k] if k < m else M + h_ind[l],
                    h_ind[l] if l < n else N + g_ind[k],
                )
                for k, l in zip(localCe.lsa_row_ind, localCe.lsa_col_ind)
                if k < m or l < n
            ]

        else:
            ij = []
            localCe = CostMatrix(np.empty((0, 0)), [], [], 0)

        return ij, localCe

    def reduce_Ce(Ce, ij, m, n):
        if len(ij):
            i, j = zip(*ij)
            m_i = m - sum(1 for t in i if t < m)
            n_j = n - sum(1 for t in j if t < n)
            return make_CostMatrix(reduce_C(Ce.C, i, j, m, n), m_i, n_j)
        return Ce

    def get_edit_ops(
            matched_uv, pending_u, pending_v, Cv, pending_g, pending_h, Ce, matched_cost
    ):
        """
        Parameters:
            matched_uv: partial vertex edit path
                list of tuples (u, v) of vertex mappings u<->v,
                u=None or v=None for deletion/insertion
            pending_u, pending_v: lists of vertices not yet mapped
            Cv: CostMatrix of pending vertex mappings
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_cost: cost of partial edit path

        Returns:
            sequence of
                (i, j): indices of vertex mapping u<->v
                Cv_ij: reduced CostMatrix of pending vertex mappings
                    (basically Cv with row i, col j removed)
                list of (x, y): indices of edge mappings g<->h
                Ce_xy: reduced CostMatrix of pending edge mappings
                    (basically Ce with rows x, cols y removed)
                cost: total cost of edit operation
            NOTE: most promising ops first
        """
        m = len(pending_u)
        n = len(pending_v)
        # assert Cv.C.shape == (m + n, m + n)

        # 1) a vertex mapping from optimal linear sum assignment
        i, j = min(
            (k, l) for k, l in zip(Cv.lsa_row_ind, Cv.lsa_col_ind) if k < m or l < n
        )
        xy, localCe = match_edges(
            pending_u[i] if i < m else None,
            pending_v[j] if j < n else None,
            pending_g,
            pending_h,
            Ce,
            matched_uv,
        )
        Ce_xy = reduce_Ce(Ce, xy, len(pending_g), len(pending_h))
        # assert Ce.ls <= localCe.ls + Ce_xy.ls
        if prune(matched_cost + Cv.ls + localCe.ls + Ce_xy.ls):
            pass
        else:
            # get reduced Cv efficiently
            Cv_ij = CostMatrix(
                reduce_C(Cv.C, (i,), (j,), m, n),
                reduce_ind(Cv.lsa_row_ind, (i, m + j)),
                reduce_ind(Cv.lsa_col_ind, (j, n + i)),
                Cv.ls - Cv.C[i, j],
            )
            yield (i, j), Cv_ij, xy, Ce_xy, Cv.C[i, j] + localCe.ls

        # 2) other candidates, sorted by lower-bound cost estimate
        other = []
        fixed_i, fixed_j = i, j
        if m <= n:
            candidates = (
                (t, fixed_j)
                for t in range(m + n)
                if t != fixed_i and (t < m or t == m + fixed_j)
            )
        else:
            candidates = (
                (fixed_i, t)
                for t in range(m + n)
                if t != fixed_j and (t < n or t == n + fixed_i)
            )
        for i, j in candidates:
            if prune(matched_cost + Cv.C[i, j] + Ce.ls):
                continue
            Cv_ij = make_CostMatrix(
                reduce_C(Cv.C, (i,), (j,), m, n),
                m - 1 if i < m else m,
                n - 1 if j < n else n,
            )
            # assert Cv.ls <= Cv.C[i, j] + Cv_ij.ls
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + Ce.ls):
                continue
            xy, localCe = match_edges(
                pending_u[i] if i < m else None,
                pending_v[j] if j < n else None,
                pending_g,
                pending_h,
                Ce,
                matched_uv,
            )
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + localCe.ls):
                continue
            Ce_xy = reduce_Ce(Ce, xy, len(pending_g), len(pending_h))
            # assert Ce.ls <= localCe.ls + Ce_xy.ls
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + localCe.ls + Ce_xy.ls):
                continue
            other.append(((i, j), Cv_ij, xy, Ce_xy, Cv.C[i, j] + localCe.ls))

        yield from sorted(other, key=lambda t: t[4] + t[1].ls + t[3].ls)

    def get_edit_paths(
            matched_uv,
            pending_u,
            pending_v,
            Cv,
            matched_gh,
            pending_g,
            pending_h,
            Ce,
            matched_cost,
    ):
        """
        Parameters:
            matched_uv: partial vertex edit path
                list of tuples (u, v) of vertex mappings u<->v,
                u=None or v=None for deletion/insertion
            pending_u, pending_v: lists of vertices not yet mapped
            Cv: CostMatrix of pending vertex mappings
            matched_gh: partial edge edit path
                list of tuples (g, h) of edge mappings g<->h,
                g=None or h=None for deletion/insertion
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_cost: cost of partial edit path

        Returns:
            sequence of (vertex_path, edge_path, cost)
                vertex_path: complete vertex edit path
                    list of tuples (u, v) of vertex mappings u<->v,
                    u=None or v=None for deletion/insertion
                edge_path: complete edge edit path
                    list of tuples (g, h) of edge mappings g<->h,
                    g=None or h=None for deletion/insertion
                cost: total cost of edit path
            NOTE: path costs are non-increasing
        """
        # debug_print('matched-uv:', matched_uv)
        # debug_print('matched-gh:', matched_gh)
        # debug_print('matched-cost:', matched_cost)
        # debug_print('pending-u:', pending_u)
        # debug_print('pending-v:', pending_v)
        # debug_print(Cv.C)
        # assert list(sorted(G1.nodes)) == list(sorted(list(u for u, v in matched_uv if u is not None) + pending_u))
        # assert list(sorted(G2.nodes)) == list(sorted(list(v for u, v in matched_uv if v is not None) + pending_v))
        # debug_print('pending-g:', pending_g)
        # debug_print('pending-h:', pending_h)
        # debug_print(Ce.C)
        # assert list(sorted(G1.edges)) == list(sorted(list(g for g, h in matched_gh if g is not None) + pending_g))
        # assert list(sorted(G2.edges)) == list(sorted(list(h for g, h in matched_gh if h is not None) + pending_h))
        # debug_print()

        if prune(matched_cost + Cv.ls + Ce.ls):
            return

        if not max(len(pending_u), len(pending_v)):
            # assert not len(pending_g)
            # assert not len(pending_h)
            # path completed!
            # assert matched_cost <= maxcost_value
            nonlocal maxcost_value
            maxcost_value = min(maxcost_value, matched_cost)
            yield matched_uv, matched_gh, matched_cost

        else:
            edit_ops = get_edit_ops(
                matched_uv,
                pending_u,
                pending_v,
                Cv,
                pending_g,
                pending_h,
                Ce,
                matched_cost,
            )
            for ij, Cv_ij, xy, Ce_xy, edit_cost in edit_ops:
                i, j = ij
                # assert Cv.C[i, j] + sum(Ce.C[t] for t in xy) == edit_cost
                if prune(matched_cost + edit_cost + Cv_ij.ls + Ce_xy.ls):
                    continue

                # dive deeper
                u = pending_u.pop(i) if i < len(pending_u) else None
                v = pending_v.pop(j) if j < len(pending_v) else None
                matched_uv.append((u, v))
                for x, y in xy:
                    len_g = len(pending_g)
                    len_h = len(pending_h)
                    matched_gh.append(
                        (
                            pending_g[x] if x < len_g else None,
                            pending_h[y] if y < len_h else None,
                        )
                    )
                sortedx = sorted(x for x, y in xy)
                sortedy = sorted(y for x, y in xy)
                G = [
                    (pending_g.pop(x) if x < len(pending_g) else None)
                    for x in reversed(sortedx)
                ]
                H = [
                    (pending_h.pop(y) if y < len(pending_h) else None)
                    for y in reversed(sortedy)
                ]

                yield from get_edit_paths(
                    matched_uv,
                    pending_u,
                    pending_v,
                    Cv_ij,
                    matched_gh,
                    pending_g,
                    pending_h,
                    Ce_xy,
                    matched_cost + edit_cost,
                )

                # backtrack
                if u is not None:
                    pending_u.insert(i, u)
                if v is not None:
                    pending_v.insert(j, v)
                matched_uv.pop()
                for x, g in zip(sortedx, reversed(G)):
                    if g is not None:
                        pending_g.insert(x, g)
                for y, h in zip(sortedy, reversed(H)):
                    if h is not None:
                        pending_h.insert(y, h)
                for _ in xy:
                    matched_gh.pop()

    # Initialization

    pending_u = list(G1.nodes)
    pending_v = list(G2.nodes)

    initial_cost = 0
    if roots:
        root_u, root_v = roots
        if root_u not in pending_u or root_v not in pending_v:
            raise nx.NodeNotFound("Root node not in graph.")

        # remove roots from pending
        pending_u.remove(root_u)
        pending_v.remove(root_v)

    # cost matrix of vertex mappings
    m = len(pending_u)
    n = len(pending_v)
    C = np.zeros((m + n, m + n))
    if node_subst_cost:
        C[0:m, 0:n] = np.array(
            [
                node_subst_cost(G1.nodes[u], G2.nodes[v])
                for u in pending_u
                for v in pending_v
            ]
        ).reshape(m, n)
        if roots:
            initial_cost = node_subst_cost(G1.nodes[root_u], G2.nodes[root_v])
    elif node_match:
        C[0:m, 0:n] = np.array(
            [
                1 - int(node_match(G1.nodes[u], G2.nodes[v]))
                for u in pending_u
                for v in pending_v
            ]
        ).reshape(m, n)
        if roots:
            initial_cost = 1 - node_match(G1.nodes[root_u], G2.nodes[root_v])
    else:
        # all zeroes
        pass
    # assert not min(m, n) or C[0:m, 0:n].min() >= 0
    if node_del_cost:
        del_costs = [node_del_cost(G1.nodes[u]) for u in pending_u]
    else:
        del_costs = [1] * len(pending_u)
    # assert not m or min(del_costs) >= 0
    if node_ins_cost:
        ins_costs = [node_ins_cost(G2.nodes[v]) for v in pending_v]
    else:
        ins_costs = [1] * len(pending_v)
    # assert not n or min(ins_costs) >= 0
    inf = C[0:m, 0:n].sum() + sum(del_costs) + sum(ins_costs) + 1
    C[0:m, n: n + m] = np.array(
        [del_costs[i] if i == j else inf for i in range(m) for j in range(m)]
    ).reshape(m, m)
    C[m: m + n, 0:n] = np.array(
        [ins_costs[i] if i == j else inf for i in range(n) for j in range(n)]
    ).reshape(n, n)
    Cv = make_CostMatrix(C, m, n)
    # debug_print(f"Cv: {m} x {n}")
    # debug_print(Cv.C)

    pending_g = list(G1.edges)
    pending_h = list(G2.edges)

    # cost matrix of edge mappings
    m = len(pending_g)
    n = len(pending_h)
    C = np.zeros((m + n, m + n))
    if edge_subst_cost:
        C[0:m, 0:n] = np.array(
            [
                edge_subst_cost(G1.edges[g], G2.edges[h])
                for g in pending_g
                for h in pending_h
            ]
        ).reshape(m, n)
    elif edge_match:
        C[0:m, 0:n] = np.array(
            [
                1 - int(edge_match(G1.edges[g], G2.edges[h]))
                for g in pending_g
                for h in pending_h
            ]
        ).reshape(m, n)
    else:
        # all zeroes
        pass
    # assert not min(m, n) or C[0:m, 0:n].min() >= 0
    if edge_del_cost:
        del_costs = [edge_del_cost(G1.edges[g]) for g in pending_g]
    else:
        del_costs = [1] * len(pending_g)
    # assert not m or min(del_costs) >= 0
    if edge_ins_cost:
        ins_costs = [edge_ins_cost(G2.edges[h]) for h in pending_h]
    else:
        ins_costs = [1] * len(pending_h)
    # assert not n or min(ins_costs) >= 0
    inf = C[0:m, 0:n].sum() + sum(del_costs) + sum(ins_costs) + 1
    C[0:m, n: n + m] = np.array(
        [del_costs[i] if i == j else inf for i in range(m) for j in range(m)]
    ).reshape(m, m)
    C[m: m + n, 0:n] = np.array(
        [ins_costs[i] if i == j else inf for i in range(n) for j in range(n)]
    ).reshape(n, n)
    Ce = make_CostMatrix(C, m, n)
    # debug_print(f'Ce: {m} x {n}')
    # debug_print(Ce.C)
    # debug_print()

    maxcost_value = Cv.C.sum() + Ce.C.sum() + 1

    if timeout is not None:
        if timeout <= 0:
            raise nx.NetworkXError("Timeout value must be greater than 0")
        start = time.perf_counter()

    def prune(cost):
        if timeout is not None:
            if time.perf_counter() - start > timeout:
                return True
        if upper_bound is not None:
            if cost > upper_bound:
                return True
        if cost > maxcost_value:
            return True
        if strictly_decreasing and cost >= maxcost_value:
            return True
        return False

    # Now go!

    done_uv = [] if roots is None else [roots]

    for vertex_path, edge_path, cost in get_edit_paths(
            done_uv, pending_u, pending_v, Cv, [], pending_g, pending_h, Ce, initial_cost
    ):
        # assert sorted(G1.nodes) == sorted(u for u, v in vertex_path if u is not None)
        # assert sorted(G2.nodes) == sorted(v for u, v in vertex_path if v is not None)
        # assert sorted(G1.edges) == sorted(g for g, h in edge_path if g is not None)
        # assert sorted(G2.edges) == sorted(h for g, h in edge_path if h is not None)
        # print(vertex_path, edge_path, cost, file = sys.stderr)
        # assert cost == maxcost_value
        yield list(vertex_path), list(edge_path), float(cost)


class UnsupportedInstanceError(Exception):
    pass
class HeteroGraph:
    """
        输入: sp_netlist 或 json 格式的电路数据，解析其中的组件信息，并完成3种等价表示的补全。
        主要方法: 
            - generate_all_from_spectre_netlist: 输入 spectre netlist, 输出json和graph
            - generate_all_from_json: 输入 json, 输出spectre netlist和graph
    
    """
    def __init__(self):
        # 初始化命令
        self.num_pmos = 0
        self.edge_dp2n = []
        self.edge_gp2n = []
        self.edge_sp2n = []
        self.edge_bp2n = []
        self.pmos_name = {}
        self.pmos_feature = []

        self.num_nmos = 0
        self.edge_dn2n = []
        self.edge_gn2n = []
        self.edge_sn2n = []
        self.edge_bn2n = []
        self.nmos_name = {}
        self.nmos_feature = []

        self.num_pnp = 0
        self.edge_bpnp2n = []
        self.edge_epnp2n = []
        self.edge_cpnp2n = []
        self.pnp_name = {}
        self.pnp_feature = []

        self.num_npn = 0
        self.edge_bnpn2n = []
        self.edge_enpn2n = []
        self.edge_cnpn2n = []
        self.npn_name = {}
        self.npn_feature = []

        self.num_r = 0
        self.edge_r2n = []
        self.r_name = {}
        self.r_feature = []

        self.num_c = 0
        self.edge_c2n = []
        self.c_name = {}
        self.c_feature = []

        self.num_l = 0
        self.edge_l2n = []
        self.l_name = {}
        self.l_feature = []

        self.num_s = 0
        self.edge_s2n = []
        self.s_name = {}
        self.s_feature = []

        self.num_diode = 0
        self.edge_diode_p2n = []
        self.edge_diode_n2n = []
        self.diode_name = {}
        self.diode_feature = []
        
        self.num_isource = 0
        self.edge_iin2n = []
        self.edge_iout2n = []
        self.isource_name = {}
        self.isource_feature = []

        self.num_vsource = 0
        self.edge_vp2n = []
        self.edge_vn2n = []
        self.vsource_name = {}
        self.vsource_feature = []

        self.num_diso = 0
        self.edge_diso_vp2n = []
        self.edge_diso_vn2n = []
        self.edge_diso_vout2n = []
        self.diso_name = {}
        self.diso_feature = []

        self.num_dido = 0
        self.edge_dido_vp2n = []
        self.edge_dido_vn2n = []
        self.edge_dido_voutp2n = []
        self.edge_dido_voutn2n = []
        self.dido_name = {}
        self.dido_feature = []

        self.num_siso = 0
        self.edge_siso_vin2n = []
        self.edge_siso_vout2n = []
        self.siso_name = {}
        self.siso_feature = []

        self.num_net = 0
        self.net_name = {}

        self.type_mapping = {
            'PMOS': 'pmos',
            'NMOS': 'nmos',
            'NPN': 'npn',
            'PNP': 'pnp',
            'Res': 'resistor',
            'Cap': 'capacitor',
            'Ind': 'inductor',
            'Diode': 'diode',
            'Switch': 'switch',
            'Current': 'isource',
            'Voltage': 'vsource',
            'Diso_amp': 'diffamp',
            'Siso_amp': 'amp',
            'Dido_amp': 'dido'
        }

        self.json = {}
        self.json_netlist = []
        self.label = ''
        self.sp_netlist = ''
        self.het_graph = None 
        self.skip_json = False


    def set_label(self,label):
        self.label = label

    def set_sp_netlist(self,sp_netlist):
        self.sp_netlist = sp_netlist

    def set_json(self,json):
        self.json = json
        self.json_netlist = json['ckt_netlist']
        self.label = json['ckt_type']

    def extract_col(self,matrix):
        # 提取第一列和第二列
        src = [row[0] for row in matrix]
        dst = [row[1] for row in matrix]
        return src, dst

    def extract_content(self,text):
        # 定义正则表达式模式
        pattern = re.compile(r'[A-Z][a-zA-Z0-9]* \(')
        
        # 找到所有匹配的行
        matches = list(pattern.finditer(text))
        
        if not matches:
            return "没有找到符合格式的内容"
        
        # 获取第一个和最后一个匹配的位置
        first_pos = matches[0].start()
        last_pos = matches[-1].start()
        
        # 找到最后一个匹配行的结束位置
        last_line_end = text.find('\n', last_pos)
        if last_line_end == -1:
            last_line_end = len(text)
        
        # 提取第一个和最后一个匹配之间的内容
        extracted_content = text[first_pos:last_line_end]

        return extracted_content 

    def get_component_info(self,line):
        match_1 = re.match(r'(\w+)\s*\(([^)]+)\)', line)
        if not match_1:
                return False
        # 提取第一个单词
        name = match_1.group(1)

        # 提取括号中的内容，并分割成列表
        ports = match_1.group(2).split()
        
        # 查找第一个捕获组
        match_2 = re.search(r'\)\s+(\w+)', line)
        type = match_2.group(1)

        # 从第一个捕获组开始的子字符串
        start_index = match_2.end(1)
        substring = line[start_index:]

        # 正则表达式匹配 <变量名>=<值>
        pattern = re.compile(r'(\w+)=([^,\s]+)')
        matches = pattern.findall(substring)

        # 将匹配结果保存到字典中
        values = {var: val for var, val in matches}
        
        return name, ports, type, values

    def reset_globals(self):
        self.num_pmos = 0
        self.edge_dp2n = []
        self.edge_gp2n = []
        self.edge_sp2n = []
        self.edge_bp2n = []
        self.pmos_name = {}
        self.pmos_feature = []

        self.num_nmos = 0
        self.edge_dn2n = []
        self.edge_gn2n = []
        self.edge_sn2n = []
        self.edge_bn2n = []
        self.nmos_name = {}
        self.nmos_feature = []

        self.num_pnp = 0
        self.edge_bpnp2n = []
        self.edge_epnp2n = []
        self.edge_cpnp2n = []
        self.pnp_name = {}
        self.pnp_feature = []

        self.num_npn = 0
        self.edge_bnpn2n = []
        self.edge_enpn2n = []
        self.edge_cnpn2n = []
        self.npn_name = {}
        self.npn_feature = []

        self.num_r = 0
        self.edge_r2n = []
        self.r_name = {}
        self.r_feature = []

        self.num_c = 0
        self.edge_c2n = []
        self.c_name = {}
        self.c_feature = []

        self.num_l = 0
        self.edge_l2n = []
        self.l_name = {}
        self.l_feature = []

        self.num_s = 0
        self.edge_s2n = []
        self.s_name = {}
        self.s_feature = []

        self.num_diode = 0
        self.edge_diode_p2n = []
        self.edge_diode_n2n = []
        self.diode_name = {}
        self.diode_feature = []
        
        self.num_isource = 0
        self.edge_iin2n = []
        self.edge_iout2n = []
        self.isource_name = {}
        self.isource_feature = []

        self.num_vsource = 0
        self.edge_vp2n = []
        self.edge_vn2n = []
        self.vsource_name = {}
        self.vsource_feature = []

        self.num_diso = 0
        self.edge_diso_vp2n = []
        self.edge_diso_vn2n = []
        self.edge_diso_vout2n = []
        self.diso_name = {}
        self.diso_feature = []

        self.num_dido = 0
        self.edge_dido_vp2n = []
        self.edge_dido_vn2n = []
        self.edge_dido_voutp2n = []
        self.edge_dido_voutn2n = []
        self.dido_name = {}
        self.dido_feature = []

        self.num_siso = 0
        self.edge_siso_vin2n = []
        self.edge_siso_vout2n = []
        self.siso_name = {}
        self.siso_feature = []

        self.num_net = 0
        self.net_name = {}

        
    def get_key_from_value(self,dict,value):
        reversed_dict = {v: k for k, v in dict.items()}
        key = reversed_dict.get(value)
        return key



    def get_net_index(self,ports,is_3=False):
        net_index = []
        if is_3:
            for net in ports[:3]:
                if net not in self.net_name:
                    self.net_name[net] = self.num_net
                    self.num_net = self.num_net + 1
                net_index.append(self.net_name[net])
        else:
            for net in ports:
                if net not in self.net_name:
                    self.net_name[net] = self.num_net
                    self.num_net = self.num_net + 1
                net_index.append(self.net_name[net])
        return net_index


    def create_pmos(self,component_info):

        name, ports, type, values = component_info
        self.pmos_name[name] = self.num_pmos
        self.num_pmos = self.num_pmos + 1
        net_index = self.get_net_index(ports,True)
        self.edge_dp2n.append([self.pmos_name[name], net_index[0]])
        self.edge_gp2n.append([self.pmos_name[name], net_index[1]])
        self.edge_dp2n.append([self.pmos_name[name], net_index[2]])
        #self.edge_sp2n.append([self.pmos_name[name], net_index[2]])
        #self.edge_bp2n.append([self.pmos_name[name], net_index[3]])

        if self.skip_json:
            return
        port_0 = self.get_key_from_value(self.net_name,net_index[0])
        port_1 = self.get_key_from_value(self.net_name,net_index[1])
        port_2 = self.get_key_from_value(self.net_name,net_index[2])
        #port_3 = self.get_key_from_value(self.net_name,net_index[3])
        
        component_dict = {
            'component_type': 'PMOS',
            'port_connection': {'Drain': port_0, 'Gate': port_1, 'Source':port_2, },}
        self.json_netlist.append(component_dict)
        return
    def create_pmos4(self,component_info):
        name, ports, type, values = component_info
        self.pmos_name[name] = self.num_pmos
        self.num_pmos = self.num_pmos + 1
        net_index = self.get_net_index(ports)
        self.edge_dp2n.append([self.pmos_name[name], net_index[0]])
        self.edge_gp2n.append([self.pmos_name[name], net_index[1]])
        self.edge_dp2n.append([self.pmos_name[name], net_index[2]])
        #self.edge_sp2n.append([self.pmos_name[name], net_index[2]])
        self.edge_bp2n.append([self.pmos_name[name], net_index[3]])

        
        if self.skip_json:
            return
        port_0 = self.get_key_from_value(self.net_name,net_index[0])
        port_1 = self.get_key_from_value(self.net_name,net_index[1])
        port_2 = self.get_key_from_value(self.net_name,net_index[2])
        port_3 = self.get_key_from_value(self.net_name,net_index[3])
        
        component_dict = {
            'component_type': 'PMOS',
            'port_connection': {'Drain': port_0, 'Gate': port_1, 'Source':port_2, 'Body':port_3},}
        self.json_netlist.append(component_dict)
        
        return
    def create_nmos(self, component_info):
        name, ports, type, values = component_info
        self.nmos_name[name] = self.num_nmos
        self.num_nmos += 1
        net_index = self.get_net_index(ports,True)
        self.edge_dn2n.append([self.nmos_name[name], net_index[0]])
        self.edge_gn2n.append([self.nmos_name[name], net_index[1]])
        self.edge_dn2n.append([self.nmos_name[name], net_index[2]])
        #self.edge_sn2n.append([self.nmos_name[name], net_index[2]])
        #self.edge_bn2n.append([self.nmos_name[name], net_index[3]])
        
        if self.skip_json:
            return
        
        port_0 = self.get_key_from_value(self.net_name, net_index[0])
        port_1 = self.get_key_from_value(self.net_name, net_index[1])
        port_2 = self.get_key_from_value(self.net_name, net_index[2])
        #port_3 = self.get_key_from_value(self.net_name, net_index[3])
        
        component_dict = {
            'component_type': 'NMOS',
            'port_connection': {'Drain': port_0, 'Gate': port_1, 'Source': port_2,},}
        
        self.json_netlist.append(component_dict)

        return
    
    def create_nmos4(self, component_info):
        name, ports, type, values = component_info
        self.nmos_name[name] = self.num_nmos
        self.num_nmos += 1
        net_index = self.get_net_index(ports)
        self.edge_dn2n.append([self.nmos_name[name], net_index[0]])
        self.edge_gn2n.append([self.nmos_name[name], net_index[1]])
        self.edge_sn2n.append([self.nmos_name[name], net_index[2]])
         #self.edge_sn2n.append([self.nmos_name[name], net_index[2]])
        self.edge_bn2n.append([self.nmos_name[name], net_index[3]])
        
        if self.skip_json:
            return
        
        port_0 = self.get_key_from_value(self.net_name, net_index[0])
        port_1 = self.get_key_from_value(self.net_name, net_index[1])
        port_2 = self.get_key_from_value(self.net_name, net_index[2])
        port_3 = self.get_key_from_value(self.net_name, net_index[3])
        
        component_dict = {
            'component_type': 'NMOS',
            'port_connection': {'Drain': port_0, 'Gate': port_1, 'Source': port_2, 'Body': port_3},}
        
        self.json_netlist.append(component_dict)

        return
    def create_pnp(self, component_info):
        name, ports, type, values = component_info
        self.pnp_name[name] = self.num_pnp
        self.num_pnp += 1
        net_index = self.get_net_index(ports,is_3=True)
        self.edge_cpnp2n.append([self.pnp_name[name], net_index[0]])
        self.edge_bpnp2n.append([self.pnp_name[name], net_index[1]])
        self.edge_epnp2n.append([self.pnp_name[name], net_index[2]])
        
        if self.skip_json:
            return
        port_collector = self.get_key_from_value(self.net_name, net_index[0])
        port_base = self.get_key_from_value(self.net_name, net_index[1])
        port_emitter = self.get_key_from_value(self.net_name, net_index[2])

        component_dict = {
            'component_type': 'PNP',
            'port_connection': {'Collector': port_collector, 'Base': port_base, 'Emitter': port_emitter},}
        
        self.json_netlist.append(component_dict)

        return
    def create_npn(self, component_info):
        name, ports, type, values = component_info
        self.npn_name[name] = self.num_npn
        self.num_npn += 1
        net_index = self.get_net_index(ports,is_3=True)
        self.edge_cnpn2n.append([self.npn_name[name], net_index[0]])
        self.edge_bnpn2n.append([self.npn_name[name], net_index[1]])
        self.edge_enpn2n.append([self.npn_name[name], net_index[2]])
        
        if self.skip_json:
            return
        port_collector = self.get_key_from_value(self.net_name, net_index[0])
        port_base = self.get_key_from_value(self.net_name, net_index[1])
        port_emitter = self.get_key_from_value(self.net_name, net_index[2])

        component_dict = {
            'component_type': 'NPN',
            'port_connection': {'Collector': port_collector, 'Base': port_base, 'Emitter': port_emitter},}
        
        self.json_netlist.append(component_dict)

        return
    def create_resistor(self, component_info):
        name, ports, type, values = component_info
        self.r_name[name] = self.num_r
        self.num_r += 1
        net_index = self.get_net_index(ports)
        self.edge_r2n.append([self.r_name[name], net_index[0]])
        self.edge_r2n.append([self.r_name[name], net_index[1]])
        
        if self.skip_json:
            return
        port_pos = self.get_key_from_value(self.net_name, net_index[0])
        port_neg = self.get_key_from_value(self.net_name, net_index[1])

        component_dict = {
            'component_type': 'Res',
            'port_connection': {'Pos': port_pos, 'Neg': port_neg},}
        
        self.json_netlist.append(component_dict)

        return
    def create_capacitor(self, component_info):
        name, ports, type, values = component_info
        self.c_name[name] = self.num_c
        self.num_c += 1
        net_index = self.get_net_index(ports)
        self.edge_c2n.append([self.c_name[name], net_index[0]])
        self.edge_c2n.append([self.c_name[name], net_index[1]])
        
        if self.skip_json:
            return
        port_pos = self.get_key_from_value(self.net_name, net_index[0])
        port_neg = self.get_key_from_value(self.net_name, net_index[1])

        component_dict = {
            'component_type': 'Cap',
            'port_connection': {'Pos': port_pos, 'Neg': port_neg},}
        
        self.json_netlist.append(component_dict)

        return
    def create_inductor(self, component_info):
        name, ports, type, values = component_info
        self.l_name[name] = self.num_l
        self.num_l += 1
        net_index = self.get_net_index(ports)
        self.edge_l2n.append([self.l_name[name], net_index[0]])
        self.edge_l2n.append([self.l_name[name], net_index[1]])
        
        if self.skip_json:
            return
        port_pos = self.get_key_from_value(self.net_name, net_index[0])
        port_neg = self.get_key_from_value(self.net_name, net_index[1])

        component_dict = {
            'component_type': 'Ind',
            'port_connection': {'Pos': port_pos, 'Neg': port_neg},}
        
        self.json_netlist.append(component_dict)

        return
    def create_switch(self, component_info):
        name, ports, type, values = component_info
        self.s_name[name] = self.num_s
        self.num_s += 1
        net_index = self.get_net_index(ports)
        self.edge_s2n.append([self.s_name[name], net_index[0]])
        self.edge_s2n.append([self.s_name[name], net_index[1]])
        
        if self.skip_json:
            return
        port_pos = self.get_key_from_value(self.net_name, net_index[0])
        port_neg = self.get_key_from_value(self.net_name, net_index[1])

        component_dict = {
            'component_type': 'Switch',
            'port_connection': {'Pos': port_pos, 'Neg': port_neg},}
        
        self.json_netlist.append(component_dict)

        return

    def create_diode(self, component_info):
        name, ports, type, values = component_info
        self.diode_name[name] = self.num_diode
        self.num_diode += 1
        net_index = self.get_net_index(ports)
        self.edge_diode_p2n.append([self.diode_name[name], net_index[0]])
        self.edge_diode_n2n.append([self.diode_name[name], net_index[1]])
        
        if self.skip_json:
            return
        port_in = self.get_key_from_value(self.net_name, net_index[0])
        port_out = self.get_key_from_value(self.net_name, net_index[1])

        component_dict = {
            'component_type': 'Diode',
            'port_connection': {'In': port_in, 'Out': port_out},}
        
        self.json_netlist.append(component_dict)

        return

    def create_isource(self, component_info):
        name, ports, type, values = component_info
        self.isource_name[name] = self.num_isource
        self.num_isource += 1
        net_index = self.get_net_index(ports)
        self.edge_iin2n.append([self.isource_name[name], net_index[0]])
        self.edge_iout2n.append([self.isource_name[name], net_index[1]])
        
        if self.skip_json:
            return
        port_in = self.get_key_from_value(self.net_name, net_index[0])
        port_out = self.get_key_from_value(self.net_name, net_index[1])

        component_dict = {
            'component_type': 'Current',
            'port_connection': {'In': port_in, 'Out': port_out},}
        
        self.json_netlist.append(component_dict)

        return

    def create_vsource(self, component_info):
        name, ports, type, values = component_info
        self.vsource_name[name] = self.num_vsource
        self.num_vsource += 1
        net_index = self.get_net_index(ports)
        self.edge_vp2n.append([self.vsource_name[name], net_index[0]])
        self.edge_vn2n.append([self.vsource_name[name], net_index[1]])
        
        if self.skip_json:
            return
        port_positive = self.get_key_from_value(self.net_name, net_index[0])
        port_negative = self.get_key_from_value(self.net_name, net_index[1])

        component_dict = {
            'component_type': 'Voltage',
            'port_connection': {'Positive': port_positive, 'Negative': port_negative},}
        
        self.json_netlist.append(component_dict)

        return
    def create_diso(self, component_info):
        name, ports, type, values = component_info
        self.diso_name[name] = self.num_diso
        self.num_diso += 1
        net_index = self.get_net_index(ports)
        self.edge_diso_vp2n.append([self.diso_name[name], net_index[0]])
        self.edge_diso_vn2n.append([self.diso_name[name], net_index[1]])
        self.edge_diso_vout2n.append([self.diso_name[name], net_index[2]])
        
        if self.skip_json:
            return
        port_inp = self.get_key_from_value(self.net_name, net_index[0])
        port_inn = self.get_key_from_value(self.net_name, net_index[1])
        port_out = self.get_key_from_value(self.net_name, net_index[2])

        component_dict = {
            'component_type': 'Diso_amp',
            'port_connection': {'InP': port_inp, 'InN': port_inn, 'Out': port_out},}
        
        self.json_netlist.append(component_dict)

        return
    def create_dido(self, component_info):
        name, ports, type, values = component_info
        self.dido_name[name] = self.num_dido
        self.num_dido += 1
        net_index = self.get_net_index(ports)
        self.edge_dido_vp2n.append([self.dido_name[name], net_index[0]])
        self.edge_dido_vn2n.append([self.dido_name[name], net_index[1]])
        self.edge_dido_voutp2n.append([self.dido_name[name], net_index[2]])
        self.edge_dido_voutn2n.append([self.dido_name[name], net_index[3]])
        
        if self.skip_json:
            return
        port_inp = self.get_key_from_value(self.net_name, net_index[0])
        port_inn = self.get_key_from_value(self.net_name, net_index[1])
        port_outp = self.get_key_from_value(self.net_name, net_index[2])
        port_outn = self.get_key_from_value(self.net_name, net_index[3])

        component_dict = {
            'component_type': 'Dido_amp',
            'port_connection': {'InP': port_inp, 'InN': port_inn, 'OutP': port_outp, 'OutN': port_outn},}
        
        self.json_netlist.append(component_dict)

        return
    def create_siso(self, component_info):
        name, ports, type, values = component_info
        self.siso_name[name] = self.num_siso
        self.num_siso += 1
        net_index = self.get_net_index(ports)
        self.edge_siso_vin2n.append([self.siso_name[name], net_index[0]])
        self.edge_siso_vout2n.append([self.siso_name[name], net_index[1]])
        
        if self.skip_json:
            return
        port_in = self.get_key_from_value(self.net_name, net_index[0])
        port_out = self.get_key_from_value(self.net_name, net_index[1])

        component_dict = {
            'component_type': 'Siso_amp',
            'port_connection': {'In': port_in, 'Out': port_out},}
        
        self.json_netlist.append(component_dict)

        return

    def generate_all_from_spectre_netlist(self,label,sp_netlist,is_json_generated = False):
        self.set_label(label)
        self.set_sp_netlist(sp_netlist)
        self.reset_globals()
        self.skip_json = is_json_generated

        is_successful = True
        data = self.extract_content(sp_netlist)

        for line in data.strip().split('\n'):  # 处理每行
            component_info = self.get_component_info(line)
            if not component_info:
                continue
            try:
                
                # 设置索引序号, 创建边
                if component_info[2] == 'pmos': 
                    self.create_pmos(component_info)
                elif component_info[2] == 'pmos4':
                    self.create_pmos4(component_info)
                elif component_info[2] == 'nmos':
                    self.create_nmos(component_info)
                elif component_info[2] == 'nmos4':
                    self.create_nmos4(component_info)
                elif component_info[2] == 'npn':
                    self.create_npn(component_info)
                elif component_info[2] == 'pnp':
                    self.create_pnp(component_info)
                elif component_info[2] in ['resistor', 'res']:
                    self.create_resistor(component_info)
                elif component_info[2] in ['capacitor', 'cap']:
                    self.create_capacitor(component_info)
                elif component_info[2] == 'inductor':
                    self.create_inductor(component_info)
                elif component_info[2] == 'diode':
                    self.create_diode(component_info)
                elif component_info[2] == 'switch':
                    self.create_switch(component_info)
                elif component_info[2] == 'isource':
                    self.create_isource(component_info)
                elif component_info[2] == 'vsource':
                    self.create_vsource(component_info)
                elif component_info[2] == 'amp':
                    self.create_siso(component_info)
                elif component_info[2] in ['diffamp', 'opamp']:
                    self.create_diso(component_info)
                elif component_info[2] in ['dido', 'fullydiffamp']:
                    self.create_dido(component_info)
                else:
                    print(f"unsupported instance occurs")
                    print(component_info)
                    raise UnsupportedInstanceError("Unsupported instance type encountered")

            except UnsupportedInstanceError as e:
                print(f"Error: {e}")
                is_successful = False
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                is_successful = False
        graph_data = {
            ('PMOS', 'dp2n', 'net'): (
                th.tensor(self.extract_col(self.edge_dp2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_dp2n)[1], dtype=th.int64)),
            
            ('PMOS', 'gp2n', 'net'): (
                th.tensor(self.extract_col(self.edge_gp2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_gp2n)[1], dtype=th.int64)),
            ('PMOS', 'sp2n', 'net'): (
                th.tensor(self.extract_col(self.edge_sp2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_sp2n)[1], dtype=th.int64)),
            ('PMOS', 'bp2n', 'net'): (
                th.tensor(self.extract_col(self.edge_bp2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_bp2n)[1], dtype=th.int64)),
            ('NMOS', 'dn2n', 'net'): (
                th.tensor(self.extract_col(self.edge_dn2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_dn2n)[1], dtype=th.int64)),
            ('NMOS', 'gn2n', 'net'): (
                th.tensor(self.extract_col(self.edge_gn2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_gn2n)[1], dtype=th.int64)),
            ('NMOS', 'sn2n', 'net'): (
                th.tensor(self.extract_col(self.edge_sn2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_sn2n)[1], dtype=th.int64)),
            ('NMOS', 'bn2n', 'net'): (
                th.tensor(self.extract_col(self.edge_bn2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_bn2n)[1], dtype=th.int64)),
            ('PNP', 'cpnp2n', 'net'): (
                th.tensor(self.extract_col(self.edge_cpnp2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_cpnp2n)[1], dtype=th.int64)),
            ('PNP', 'bpnp2n', 'net'): (
                th.tensor(self.extract_col(self.edge_bpnp2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_bpnp2n)[1], dtype=th.int64)),
            ('PNP', 'epnp2n', 'net'): (
                th.tensor(self.extract_col(self.edge_epnp2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_epnp2n)[1], dtype=th.int64)),
            ('NPN', 'cnpn2n', 'net'): (
                th.tensor(self.extract_col(self.edge_cnpn2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_cnpn2n)[1], dtype=th.int64)),
            ('NPN', 'bnpn2n', 'net'): (
                th.tensor(self.extract_col(self.edge_bnpn2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_bnpn2n)[1], dtype=th.int64)),
            ('NPN', 'enpn2n', 'net'): (
                th.tensor(self.extract_col(self.edge_enpn2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_enpn2n)[1], dtype=th.int64)),
            ('R', 'r2n', 'net'): (
                th.tensor(self.extract_col(self.edge_r2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_r2n)[1], dtype=th.int64)),
            ('C', 'c2n', 'net'): (
                th.tensor(self.extract_col(self.edge_c2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_c2n)[1], dtype=th.int64)),
            ('L', 'l2n', 'net'): (
                th.tensor(self.extract_col(self.edge_l2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_l2n)[1], dtype=th.int64)),
            ('S', 's2n', 'net'): (
                th.tensor(self.extract_col(self.edge_s2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_s2n)[1], dtype=th.int64)),
            ('DIODE', 'diode_p2n', 'net'): (
                th.tensor(self.extract_col(self.edge_diode_p2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_diode_p2n)[1], dtype=th.int64)),
            ('DIODE', 'diode_n2n', 'net'): (
                th.tensor(self.extract_col(self.edge_diode_n2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_diode_n2n)[1], dtype=th.int64)),
            ('ISOURCE', 'iin2n', 'net'): (
                th.tensor(self.extract_col(self.edge_iin2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_iin2n)[1], dtype=th.int64)),
            ('ISOURCE', 'iout2n', 'net'): (
                th.tensor(self.extract_col(self.edge_iout2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_iout2n)[1], dtype=th.int64)),
            ('VSOURCE', 'vp2n', 'net'): (
                th.tensor(self.extract_col(self.edge_vp2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_vp2n)[1], dtype=th.int64)),
            ('VSOURCE', 'vn2n', 'net'): (
                th.tensor(self.extract_col(self.edge_vn2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_vn2n)[1], dtype=th.int64)),
            ('DISO', 'diso_vp2n', 'net'): (
                th.tensor(self.extract_col(self.edge_diso_vp2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_diso_vp2n)[1], dtype=th.int64)),
            ('DISO', 'diso_vn2n', 'net'): (
                th.tensor(self.extract_col(self.edge_diso_vn2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_diso_vn2n)[1], dtype=th.int64)),
            ('DISO', 'diso_vout2n', 'net'): (
                th.tensor(self.extract_col(self.edge_diso_vout2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_diso_vout2n)[1], dtype=th.int64)),
            ('DIDO', 'dido_vp2n', 'net'): (
                th.tensor(self.extract_col(self.edge_dido_vp2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_dido_vp2n)[1], dtype=th.int64)),
            ('DIDO', 'dido_vn2n', 'net'): (
                th.tensor(self.extract_col(self.edge_dido_vn2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_dido_vn2n)[1], dtype=th.int64)),
            ('DIDO', 'dido_voutp2n', 'net'): (
                th.tensor(self.extract_col(self.edge_dido_voutp2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_dido_voutp2n)[1], dtype=th.int64)),
            ('DIDO', 'dido_voutn2n', 'net'): (
                th.tensor(self.extract_col(self.edge_dido_voutn2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_dido_voutn2n)[1], dtype=th.int64)),
            ('SISO', 'siso_vin2n', 'net'): (
                th.tensor(self.extract_col(self.edge_siso_vin2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_siso_vin2n)[1], dtype=th.int64)),
            ('SISO', 'siso_vout2n', 'net'): (
                th.tensor(self.extract_col(self.edge_siso_vout2n)[0], dtype=th.int64),
                th.tensor(self.extract_col(self.edge_siso_vout2n)[1], dtype=th.int64))
        }
       
        

        self.het_graph = None
        self.het_graph = dgl.heterograph(graph_data) # 没有设置特征值，仅设置定义了连接关系。
        


        self.reset_globals()




        if not is_json_generated:
            self.json = {}
            self.json['ckt_type'] = self.label
            self.json['ckt_netlist'] = self.json_netlist
            self.json_netlist = []
            print('json netlist cleared.')
        
        return self.json,self.het_graph, is_successful

    def ports_in_order(self,component_type, port_connection):
        # 定义不同component_type对应的端口顺序
        order_map = {
            'PMOS': ['Drain', 'Gate', 'Source'],
            'NMOS': ['Drain', 'Gate', 'Source'],
            'PNP': ['Collector', 'Base', 'Emitter'],
            'NPN': ['Collector', 'Base', 'Emitter'],
            'Res': ['Pos', 'Neg'],
            'Cap': ['Pos', 'Neg'],
            'Ind': ['Pos', 'Neg'],
            'Switch': ['Pos', 'Neg'],
            'Diode': ['In', 'Out'],
            'Current': ['In', 'Out'],
            'Voltage': ['Positive', 'Negative'],
            'Diso_amp': ['InP', 'InN', 'Out'],
            'Dido_amp': ['InP', 'InN', 'OutP', 'OutN'],
            'Siso_amp': ['In', 'Out']
        }

        # 获取当前component_type的顺序，如果不在order_map中，使用原始顺序
        order = order_map.get(component_type, [])
        
        # 建立新的有序字典
        ordered_ports = {key: port_connection[key] for key in order if key in port_connection}
        
        # 如果存在body-bodyvalue，将其放入最后
        if 'Body' in port_connection:
            ordered_ports['Body'] = port_connection['Body']
        
        return ordered_ports
        

    def generate_spectre_netlist_from_json(self):

        netlist_lines = []
        device_num = 0
        for component in self.json_netlist:
            component_type = component.get('component_type')
            port_connection = component.get('port_connection', {})

            if not component_type or not port_connection:
                continue  # Skip invalid components
            
            # Map the component type to the corresponding Spectre type
            spectre_type = self.type_mapping.get(component_type)
            if (spectre_type == 'pmos' or spectre_type == 'nmos') and len(port_connection) == 4:
                spectre_type = 'pmos4' if spectre_type == 'pmos' else 'nmos4'

            ordered_ports = self.ports_in_order(component_type, port_connection)

            if not spectre_type:
                continue  # Skip if the component type is not recognized

            # Generate the netlist line for the given component
            ports = ' '.join(ordered_ports.values())
            netlist_line = f"X{device_num} ({ports}) {spectre_type}"
            netlist_lines.append(netlist_line)
            device_num += 1
        # Join all netlist lines into a single string
        
        netlist_str = '\n'.join(netlist_lines)
        self.sp_netlist = netlist_str
        return 

    def generate_all_from_json(self,json):
        self.set_json(json)
        self.generate_spectre_netlist_from_json()
        _,_,is_successful= self.generate_all_from_spectre_netlist(self.label,self.sp_netlist,is_json_generated=True)
        return self.sp_netlist, self.het_graph,is_successful



## utils

def to_MG(nx_g):
    # 创建一个多重图
    MG = nx.MultiGraph()
    # 添加节点并赋予属性
    for node, data in nx_g.nodes(data=True):
        MG.add_node(node, ntype=data['ntype'])
    # 添加多条边并赋予属性
    for u, v, key, data in nx_g.edges(keys=True, data=True):
        MG.add_edge(u, v, key=key, etype=str(data['etype'][1]))
    return MG

def node_match(dicta,dictb):
    if dicta['ntype'] == dictb['ntype']:
        return True
    else:
        return False

def edge_match4iso(dicta,dictb):
    if next(iter(dicta.values()))['etype'] == next(iter(dictb.values()))['etype']:
        return True
    else:
        return False
def edge_match(dicta,dictb):
    if dicta['etype'] == dictb['etype']:
        return True
    else:
        return False


import matplotlib.pyplot as plt

def plot_het(nx_g1,filename):
    color_map = {
        'net': '#6C30A2',
        'C': '#007E6D',
        'R': '#FFB749',
        'PMOS': '#0043AC',
        'NMOS': '#D82D5B'
    }
    # 获取节点类型并为每个节点分配颜色
    node_colors = []
    for n, d in nx_g1.nodes(data=True):
        ntype = d['ntype']
        node_colors.append(color_map[ntype])
    # 绘制图形
    pos = nx.spring_layout(nx_g1)
    nx.draw_networkx_nodes(nx_g1, pos, alpha = 1,node_size=150, node_color=node_colors)
    ax = plt.gca()
    for e in nx_g1.edges:
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="0",
                                    shrinkA=10, shrinkB=10,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                    ),
                                    ),
                    )
    plt.axis('off')

    plt.savefig(f'/home/public/public/{filename}.png')
    plt.cla()
    return



def ged(graph1, graph2, id, timeout, task_name):
    # 计算graph1和graph2的图编辑距离
    from networkx import is_isomorphic
    nx_1 = to_MG(graph1.to_networkx().to_undirected())
    nx_2 = to_MG(graph2.to_networkx().to_undirected())
    
    iso_first = 1
    print(f'开始测试测例 {id}')
    if iso_first and is_isomorphic(nx_1,nx_2,node_match,edge_match4iso):
        ged_val = 0
        cost = 0
        print(f'测例{id}正确')
    else:   
        #plot_het(nx_2,'true')
        #plot_het(nx_1,'gen')
        
        #import ged4py as gm

        #ged = gm.GraphEditDistance(1,1,1,1)
        #ged.set_attr_graph_used("ntype","etype")
        ged_val = 0.0
        N=10
        #for i in range(0,N): 
        #    result=ged.compare([nx_1,nx_2],None)
            
        #    ged_val +=result[0][1]
        #
        #ged_val = ged_val / N
        ged_val, history = my_graph_edit_distance(nx_1,nx_2,node_match,edge_match,timeout = 3, task_name=task_name)
        print(f'测例{id}识别有误。近似ged,供参考：{ged_val}')
        print('正在进一步检查（约1分钟），请耐心等待...')
        cost, history = my_graph_edit_distance(nx_1,nx_2,node_match,edge_match,timeout = timeout, task_name=task_name)
        
        
        #print('cost by graph_edit_distance:',ged_val)
    return ged_val,cost


def minimum(lst):
    if not lst:
        raise ValueError("列表不能为空")
    
    minimum = lst[0]
    for num in lst[1:]:
        if num < minimum:
            minimum = num
    return minimum
def average(lst):
    # 检查列表是否为空
    if not lst:
        raise ValueError("列表不能为空")
    
    # 初始化总和和计数器
    total_sum = 0
    count = 0
    
    # 遍历列表，累加总和并计数
    for num in lst:
        total_sum += num
        count += 1
    
    # 计算平均数
    average = total_sum / count
    return average

def load_from_pkl(pkl_path):
    import pickle
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
    print(f'test cases are loaded from {pkl_path}')

def align_sp(ckt_sp,true_sp):
    

    import re
    # 正则表达式
    pattern = r'^(X\d+) (.*)$'
    ckt_lines = ckt_sp.splitlines
    i = 0
    for ckt_line in ckt_lines:     
        ckt_lines[i] = re.search(pattern, ckt_line).group(2)
        i+=1
        
    true_lines = true_sp.splitlines
    i = 0
    for true_line in true_lines:     
        true_lines[i] = re.search(pattern, true_line).group(2)
        i+=1

    aligned_ckt_lines = []
    i=0
    for i in range(0,len(ckt_lines)):
        index = 0
        for j in range(0,len(true_lines)):
            if ckt_lines[i] == true_lines[j]:
                index = j
                break
            else:
                continue
  



