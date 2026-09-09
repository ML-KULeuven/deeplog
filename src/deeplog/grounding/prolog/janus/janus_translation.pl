%  Copyright (c) 2024-2026. KU Leuven
:- module(janus_translation,[to_symbol/2, from_symbol/2]).

to_symbol(Term, Symbol) :- to_symbol(_,Term,Symbol).

% Translate Prolog list syntax to deeplog's cons/nil Symbol encoding so
% list-producing builtins (nth0/3, member/2, ...) round-trip cleanly. The
% ``nonvar`` guards prevent the list-cell pattern from binding an unbound
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

from_symbol(Symbol,Term) :- from_symbol(_,Symbol,Term).

% Mirror of the to_symbol list clauses: a cons/nil-encoded Symbol becomes
% a native Prolog list when materialized as a Term. Janus marshals Python
% tuple functors as Prolog atoms; deeplog's own to_symbol writes them as
% strings. Use ``atom_string/2`` here so either form is recognised.
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
