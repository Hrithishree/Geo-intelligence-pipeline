
from graph_pipeline.node_detection import get_neighbors

def trace_edges(skel, endpoints, junctions):
    visited = set()
    edges = []
    node_set = set(endpoints + junctions)

    for node in node_set:
        for nbr in get_neighbors(skel, *node):

            if (node, nbr) in visited:
                continue

            path = [node]
            curr = nbr
            prev = node

            while True:
                path.append(curr)
                visited.add((prev, curr))

                if curr in node_set and curr != node:
                    edges.append(path)
                    break

                nbrs = [p for p in get_neighbors(skel, *curr) if p != prev]

                if len(nbrs) == 0:
                    break

                prev = curr
                curr = nbrs[0]

    return edges
