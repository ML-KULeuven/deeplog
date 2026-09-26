%  Copyright (c) 2024-2026. KU Leuven
% DeepLog's Prolog library for provers run by a JanusProver, loaded with
% `:- use_module(deeplog(grounding)).` Its export list is its public contract.

:- module(deeplog_grounding, [
    to_symbol/2,
    from_symbol/2,
    is_builtin/2,
    call_builtin/2,
    unknown_predicate/1,
    non_ground_open_predicate/1
]).

%! to_symbol(+Term, -Symbol) is det.
%  Symbol is Term as a DeepLog symbol. A list becomes a cons/nil chain.
to_symbol(Term, Symbol) :- to_symbol(_,Term,Symbol).

% The ``nonvar`` guards keep the list-cell pattern from binding an unbound
% Prolog variable into a fresh ``[H|T]`` cell.
to_symbol(_,Term, -("nil")) :- nonvar(Term), Term == [], !.
to_symbol(Variables,Term, Symbol) :-
    nonvar(Term), Term = [Head|Tail], !,
    to_symbol(Variables, Head, HSym),
    to_symbol(Variables, Tail, TSym),
    Symbol = -("cons", HSym, TSym).

to_symbol(_,Term,-(Symbol)) :-
    atomic(Term), atom_string(Term,Symbol).


to_symbol(Variables,Variable,-(VarString)) :-
    var(Variable),
    term_string(Variable,VarString),
    memberchk(VarString-Variable, Variables).

to_symbol(Variables,Term,Symbol) :-
    compound(Term),
    Term =.. [Functor | Args],
    maplist(to_symbol,Variables,Args,SymbolArgs),
    atom_string(Functor,FunctorString),
    Symbol =.. ['-', FunctorString | SymbolArgs].

%! from_symbol(+Symbol, -Term) is det.
%  Term is the DeepLog symbol Symbol as a Prolog term. A cons/nil chain becomes
%  a list.
from_symbol(Symbol,Term) :- from_symbol(_,Symbol,Term).

% Janus marshals Python tuple functors as Prolog atoms, while to_symbol writes
% them as strings; ``atom_string/2`` recognises either form.
from_symbol(_, Sym, []) :-
    nonvar(Sym), Sym = -(Name),
    atom_string(NameAtom, Name), NameAtom == nil, !.
from_symbol(Variables, Sym, [Head|Tail]) :-
    nonvar(Sym), Sym =.. ['-', Name, HSym, TSym],
    atom_string(NameAtom, Name), NameAtom == cons, !,
    from_symbol(Variables, HSym, Head),
    from_symbol(Variables, TSym, Tail).

from_symbol(Variables, Symbol,Term) :-
     Symbol =.. ['-', FunctorString | SymbolArgs],
     \+variable_string(FunctorString),
     atom_string(FunctorAtom,FunctorString),
     ((atom_number(FunctorAtom,Functor),!) ; Functor = FunctorAtom),
     maplist(from_symbol(Variables),SymbolArgs,Args),
     Term =.. [Functor | Args].

from_symbol(Variables, -(VarString), Variable) :-
    variable_string(VarString),
    memberchk(VarString-Variable, Variables).

variable_string(String) :- get_string_code(1,String,Code), code_type(Code,upper).
variable_string(String) :- get_string_code(1,String,95).

% The SWI-Prolog builtins a program may call.
allowed_builtin(between(_,_,_)).
allowed_builtin(nth0(_,_,_)).
allowed_builtin(member(_,_)).
allowed_builtin(length(_,_)).
allowed_builtin(\==(_,_)).
allowed_builtin(=:=(_,_)).
allowed_builtin(is(_,_)).

%! is_builtin(+Prover, +Goal) is semidet.
%  Goal is a builtin: an SWI-Prolog builtin a program may call, or one
%  registered on the JanusProver Prover.
is_builtin(_, Goal) :-
    nonvar(Goal),
    allowed_builtin(Goal), !.
is_builtin(Prover, Goal) :-
    nonvar(Goal),
    functor(Goal, Name, Arity),
    py_call(Prover:'_is_builtin'(Name, Arity), @(true)).

%! call_builtin(+Prover, ?Goal) is nondet.
%  Prove the builtin Goal, binding its arguments once per answer.
call_builtin(_, Goal) :-
    allowed_builtin(Goal), !,
    call(Goal).
call_builtin(Prover, Goal) :-
    to_symbol(Goal, Symbol),
    py_call(Prover:'_call_builtin'(Symbol), Answers),
    member(Answer, Answers),
    from_symbol(Answer, Term),
    Goal = Term.

%! unknown_predicate(+Goal) is det.
%  Throw the error a JanusProver query raises as UnknownPredicateException.
unknown_predicate(Goal) :-
    functor(Goal, Name, Arity),
    throw(error(unknown_procedure(Name, Arity), _)).

%! non_ground_open_predicate(+Goal) is det.
%  Throw the error a JanusProver query raises as ValueError: Goal, an instance of
%  an open predicate, is not ground after proving its rule body.
non_ground_open_predicate(Goal) :-
    functor(Goal, Name, Arity),
    throw(error(open_predicate_not_ground(Name, Arity), _)).
