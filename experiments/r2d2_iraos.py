#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# rao* on r2d2 model

import sys
from models.r2d2model import R2D2Model
from iraostar import iRAO_Executive

# Now you can give command line cc argument after filename
if __name__ == '__main__':
    # default chance constraint value
    cc = 0.08
    if len(sys.argv) > 1:
        cc = float(sys.argv[1])

    length = 3
    model = R2D2Model(length)
    b_init = model.initial_belief()

    algo = iRAO_Executive(model, b_init, cc=cc, debugging=True, cc_type='o')

    # incremental_search(self, b_current, previous_graph, previous_policy,
    # spent_risk=0, time_limit=np.inf, iter_limit=np.inf):

    # P, G = algo.incremental_search(b_init, None, None, 0)
    # print(P)
    # P = clean_policy(P)
    #
    # model.print_model()
    # model.print_policy(P)
    # print(P)
    #
    # most_likely_policy(G, model)
    #
    # print(len(G.nodes), G.nodes)
    # print(G.root)
    # next = next_child(G, G.root)
    # next = next_child(G, next)
    # next = next_child(G, next)
    # G.update_root_and_purge(next)
    # print(len(G.nodes), G.nodes)
    # print(G.all_descendants(G.root))

    # g = graph_to_json.policy_to_json(G, chance_constraint, "results/r2d2_raos.json")
    # print(g)
