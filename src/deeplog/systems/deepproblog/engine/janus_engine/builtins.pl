%  Copyright (c) 2024-2026. KU Leuven
% Shared list of pure Prolog builtins allowed inside engine goals. Both
% JanusEngine and KBestJanusEngine import this module so the two provers
% agree on what a "builtin" is without duplicating the fact table.

:- module(builtins, [allowed_builtin/1]).

allowed_builtin(between(_,_,_)).
allowed_builtin(nth0(_,_,_)).
allowed_builtin(member(_,_)).
allowed_builtin(length(_,_)).
allowed_builtin(\==(_,_)).
allowed_builtin(=:=(_,_)).
allowed_builtin(is(_,_)).
