%  Copyright (c) 2024-2026. KU Leuven
:- module(simplify,[simplify/2]).

is_true(true).
is_false(false).

simplify(true,true) :- !.
simplify(false,false) :- !.

simplify(and(L),Simplified) :-
    !,
    maplist(simplify,L,L2),
    exclude(is_true,L2,L3),
    L4 = L3,
%    list_to_set(L3,L4),
    simplify_(and(L4),Simplified).

simplify(or(L),Simplified) :-
    !,
    maplist(simplify,L,L2),
    exclude(is_false,L2,L3),
    L4 = L3,
%    list_to_set(L3,L4),
    simplify_(or(L4),Simplified).

simplify(not(P), Simplified) :-
    !,
    simplify(P,P2),
    simplify_(not(P2),Simplified).

simplify(X,X).

simplify_(and([]),true) :- !.
simplify_(and([P]),P) :- !.
simplify_(and(L),false) :- member(false,L),!.

simplify_(or([]),false) :- !.
simplify_(or([P]),P) :- !.
simplify_(or(L),true) :- member(true,L),!.

simplify_(not(true),false) :- !.
simplify_(not(false),true) :- !.
simplify_(not(not(P)),P) :- !.

simplify_(X,X).
