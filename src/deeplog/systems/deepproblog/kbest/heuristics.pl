%  Copyright (c) 2024-2026. KU Leuven
% Heuristic state constructors and updates, adapted from DeepProbLog's
% heuristics_heap.pl. Only partial_probability and geometric_mean are
% supported — the other upstream heuristics depend on machinery that
% deeplog does not model (max/extern/UCS).

partial_probability(pp(-1.0)).
geometric_mean(gm(0.0, 0)).

% Partial probability: H <- H * P. Initial H = -1.0; SWI heap is a min-heap
% and higher-probability proofs yield more-negative H, so they are popped
% first.
add_probability_to_heuristic(P, pp(H1), pp(H2)) :- !, H2 is H1 * P.

% Geometric mean of -log probabilities. H starts at 0; each fact adds
% -log(P) and averages over the depth. Min-heap pops the smallest average
% (most-probable path on average) first.
add_probability_to_heuristic(P, gm(H1, D1), gm(H2, D2)) :- !,
    D2 is D1 + 1,
    (P =< 0 -> LogP = 1.0e12 ; LogP is -log(P)),
    H2 is (H1 * D1 + LogP) / D2.

% Identity fallback: if the heuristic doesn't recognize the label's shape
% (e.g. non-numeric metadata), leave the state unchanged.
add_probability_to_heuristic(_, H, H).
