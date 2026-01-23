"""
Module for the Lark parser instanciated with PFOQ grammar.
"""


import lark


class PfoqParser(lark.Lark):
    """Parser for a PFOQ program.

    This parser will output an abstract syntax tree associated to any
    valid PFOQ program.

    Examples
    --------
    >>> parser = PfoqParser()
    >>> prg = "decl f(q){q[0]*=H;call f(q-[0]);}::call f(q);"
    >>> parser.parse(prg)
    Tree(Token('RULE', 'prg'), ...)

    """

    def __init__(self):
        super().__init__(PFOQGRAMMAR, start="prg")


PFOQGRAMMAR = r"""
disjunction: conjunction ("||" conjunction)+                                                 -> multiple_conjs
| conjunction                                                                                -> conjunction
conjunction: invert ("&&" invert)+                                                           -> multiple_disjs
| invert                                                                                     -> disjonction
invert: "!" "(" disjunction ")"                                                              -> inversion
| boolean_expression                                                                         -> boolean_expression
boolean_expression: BOOLEAN_LITERAL                                                          -> bool_literal
                    | int_expression ">" int_expression                                      -> bool_greater_than
                    | int_expression ">=" int_expression                                     -> bool_greatereq_than
                    | int_expression "<" int_expression                                      -> bool_smaller_than
                    | int_expression "<=" int_expression                                     -> bool_smallereq_than
                    | int_expression "==" int_expression                                     -> bool_equals
                    | int_expression "!=" int_expression                                     -> bool_different
                    | "(" disjunction ")"                                                    -> par_disj
register_expression: REGISTER_IDENTIFIER                                                     -> register_expression_identifier
| REGISTER_VARIABLE                                                                          -> register_variable
| parenthesed_register_expression                                                            -> register_expression_parenthesed
| parenthesed_register_expression "^-"                                                       -> register_expression_parenthesed_first_half
| parenthesed_register_expression "^+"                                                       -> register_expression_parenthesed_second_half
| register_expression "-" "[" int_expression ("," int_expression)* "]"                       -> register_expression_minus
parenthesed_register_expression : "(" register_expression ")"
qubit_expression: REGISTER_IDENTIFIER "[" int_expression "]"                                -> qubit_expression_identifier
| REGISTER_VARIABLE "[" int_expression "]"                                                  -> qubit_expression_variable
| parenthesed_register_expression "[" int_expression "]"                                    -> qubit_expression_parenthesed
gate_expression: "H"                                                                        -> hadamard_gate
| "NOT"                                                                                     -> not_gate
| "Rot" "[" LAMBDA_EXPR "]"  "(" int_expression ")"                                         -> rotation_gate
| "Ph" "[" LAMBDA_EXPR "]" "(" int_expression ")"                                           -> phase_shift_gate
|  STRING_LITERAL                                                                           -> other_gates
int_expression : SIGNED_NUMBER                                                              -> int_expression_literal
|  INT_IDENTIFIER                                                                           -> int_expression_identifier
|  int_expression OPBIN int_expression                                                      -> binary_op
| "|" register_expression "|"                                                               -> size_of_register
| parenthesed_int_expression "/2"                                                           -> parenthesed_int_expression_half
| "|" register_expression "|" "/2"                                                          -> int_expression_half_size
| parenthesed_int_expression                                                                -> parenthesed_int_expression
parenthesed_int_expression : "(" int_expression ")"
statement : "call" PROC_IDENTIFIER ("[" int_expression (","  int_expression)* "]")? "(" register_expression ("," register_expression)* ")" ";"   -> procedure_call
| "qcase" "(" qubit_expression ")" "of" "{" "0" "->" lstatement "," "1" "->" lstatement "}"     -> qcase_statement
| "qcase" "(" qubit_expression "," qubit_expression ")" "of" "{" "00" "->" lstatement "," "01" "->" lstatement "," "10" "->" lstatement "," "11" "->" lstatement  "}"  -> qcase_statement_two_qubits
| "if" "(" disjunction ")" "then" "{" lstatement "}" ("else" "{" lstatement "}")?               -> if_statement
| qubit_expression "*=" gate_expression ";"                                                     -> gate_application
| "CNOT" "(" qubit_expression "," qubit_expression ")" ";"                                          -> cnot_gate
| "SWAP" "(" qubit_expression "," qubit_expression ")" ";"                                        -> swap_gate
| "TOF(" qubit_expression "," qubit_expression "," qubit_expression ")" ";"                       -> toffoli_gate
| "skip" ";"                                                                                      -> skip_statement
lstatement : (statement)*
def : "define" (REGISTER_IDENTIFIER)+ ";"
decl : "decl" PROC_IDENTIFIER ("[" INT_IDENTIFIER  (","  INT_IDENTIFIER)* "]")? "(" REGISTER_VARIABLE ("," REGISTER_VARIABLE)* ")" "{" lstatement "}"
prg : (decl)* "::" def "::" lstatement
BOOLEAN_LITERAL: "true"
                | "false"
STRING_LITERAL : "\"" /[a-zA-Z][a-zA-Z0-9{}]*/ "\""
REGISTER_IDENTIFIER : /(q|p|r)[0-9]*/
REGISTER_VARIABLE : /(a|b|c)[0-9]*/
PROC_IDENTIFIER : /[a-zA-Z][a-zA-Z_0-9]*/
INT_IDENTIFIER : /(x|y|z)[0-9]*/
LAMBDA_EXPR : "lambda" /[^:\[\]]+/ ":" /[^:\[\]]+/
OPBIN : /[+-]/
COMMENT: "//" /[^\n]/* | "#" /[^\n]/*
%ignore COMMENT
%import common.WS
%import common.SIGNED_NUMBER
%ignore WS
"""


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
