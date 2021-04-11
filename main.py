import sys
import string
from typing import Dict, Tuple, Optional, List

# Transition= word, pop, push
Transition = (str, str, str)
WORD = 0
POP = 1
PUSH = 2
State = int
TransitionNFA = str
count_state: State = 0

# Tokeni
LETTERS = [chr(letter) for letter in range(ord('a'), ord('z') + 1)]
CONCATENATION = '#'
REUNION = '|'
KLEENSTAR = '*'
OPEN = '('
CLOSE = ')'
EPS = ''

# Stiva
class Stack:
    def __init__(self):
        self.stack = []

    # Adaugare
    def push(self, el):
        self.stack.append(el)

    # Returnare elemt
    def peek(self, pos):
        if pos < len(self.stack):
            return self.stack[-(pos + 1)]

        return None

    # Returnare si scoatere element din varful stivei
    def pop(self):
        el = self.peek(0)
        self.stack.pop()
        return el

    # Stiva goala
    def empty(self):
        return self.stack == []

    # Returnare dimensiune stiva
    def size(self):
        return len(self.stack)

class Expr:
    None

class Reunion(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left: Expr = left
        self.right: Expr = right

    def GenNFA(self):
        global count_state
        
        # Se construieste un nou NFA
        aux_nfa_left: NFA = self.left.GenNFA()
        aux_nfa_right: NFA = self.right.GenNFA()
        aux_nfa : NFA = NFA()

        aux_nfa.states = aux_nfa_left.states + aux_nfa_right.states + [count_state, count_state+1]
        aux_nfa.initialState = count_state
        aux_nfa.finalStates = [count_state+1]
        aux_nfa.transitions = dict(aux_nfa_left.transitions)
        aux_nfa.transitions.update(aux_nfa_right.transitions)
        # Se adauga 2 tranzitii pe epsilon de stare initiala la cele doua stari
        # initiale ale nfa_left si nfa_right
        aux_nfa.transitions[count_state,EPS] = [aux_nfa_left.initialState]
        aux_nfa.transitions[count_state,EPS].append(aux_nfa_right.initialState)
        
        # Se adauga 2 tranzitii pe epsilon de la starile finale ale celor 2 nfa
        # la noua stare finala
        aux_nfa.transitions[aux_nfa_left.finalStates[0], EPS] = [count_state+1]
        aux_nfa.transitions[aux_nfa_right.finalStates[0], EPS] = [count_state+1]
        count_state = count_state + 2
        return aux_nfa

class Concatenation(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left: Expr = left
        self.right: Expr = right

    def GenNFA(self):
        global count_state
        #Se genereza un nou NFA
        aux_nfa_left: NFA = self.left.GenNFA()
        aux_nfa_right: NFA = self.right.GenNFA()
        aux_nfa : NFA = NFA()
        
        # Starea initiala este starea initiala a nfa_left
        aux_nfa.initialState = aux_nfa_left.initialState

        # Starea finala este starea finala a nfa_right
        aux_nfa.finalStates = aux_nfa_right.finalStates
        aux_nfa.states = aux_nfa_left.states + aux_nfa_right.states
        aux_nfa.transitions = dict(aux_nfa_left.transitions)
        aux_nfa.transitions.update(aux_nfa_right.transitions)

        # Se adauga o tranzitie pe epsilon de la starea finala a nfa_left
        # la starea initiala a nfa_right
        aux_nfa.transitions[aux_nfa_left.finalStates[0],EPS] = [aux_nfa_right.initialState]
        return aux_nfa

class Kleen(Expr):
    def __init__(self, expr: Expr):
        self.expr: Expr = expr

    def GenNFA(self):
        global count_state
        aux_nfa: NFA = self.expr.GenNFA()

        aux_int_state: State = aux_nfa.initialState
        aux_int_final: State = aux_nfa.finalStates[0]
        
        # Se adauga o noua stare initiala si finala
        aux_nfa.initialState = count_state
        aux_nfa.finalStates = [count_state+1]
        aux_nfa.states.append(count_state)
        aux_nfa.states.append(count_state+1)

        # Se adauga tranzitii pe epsilon de la noua starea initiala la vechea stare 
        # initial si la noua stare finala
        aux_nfa.transitions[aux_nfa.initialState, EPS] = [aux_int_state, aux_nfa.finalStates[0]]
        
        # Se afauga tranzitii pe epsilon de la vechea stare finala la vechea starea initiala
        # si la noua stare finala
        aux_nfa.transitions[aux_int_final, EPS] = [aux_int_state,aux_nfa.finalStates[0]] 
        count_state = count_state + 2
        return aux_nfa

class Par(Expr):
    def __init__(self, expr: Expr):
        self.expr: Expr = expr

    def GenNFA(self):
        return self.expr.GenNFA()

class Letter(Expr):
    def __init__(self, num: str):
        self.num = num

    def GenNFA(self):
        global count_state
        # Se genereza un NFA, cu doua stari si
        # o tranzitie pe simbol
        aux_nfa = NFA()
        aux_nfa.initialState = count_state
        aux_nfa.states.append(count_state)
        aux_nfa.states.append(count_state+1)
        aux_nfa.finalStates.append(count_state+1)
        aux_nfa.transitions[count_state, self.num] = [count_state+1]
        count_state = count_state + 2
        return aux_nfa

# Generare PDA
class PDA:
    def __init__(self):
        self.states: List[State] = [0, 1, 2, 3]
        self.initialState: State = 0
        self.finalStates: List[State] = [1]

        self.transitions: Dict[Tuple[State, Transition], State] = {}

        # Tranzitii PDA
        for letter in LETTERS:
            self.transitions[0, (letter, EPS, letter)] = 1

        self.transitions[0, (OPEN, EPS, OPEN)] = 2

        self.transitions[1, (CLOSE, EPS, CLOSE)] = 1
        self.transitions[1, (KLEENSTAR, EPS, KLEENSTAR)] = 1
        self.transitions[1, (CONCATENATION, EPS, CONCATENATION)] = 0
        self.transitions[1, (REUNION, EPS, REUNION)] = 0

        self.transitions[2, (OPEN, EPS, OPEN)] = 2

        for letter in LETTERS:
            self.transitions[2, (letter, EPS, letter)] = 3

        self.transitions[3, (CLOSE, EPS, CLOSE)] = 1
        self.transitions[3, (KLEENSTAR, EPS, KLEENSTAR)] = 3
        self.transitions[3, (CONCATENATION, EPS, CONCATENATION)] = 2
        self.transitions[3, (REUNION, EPS, REUNION)] = 2  

        self.stack: Stack = Stack()

    def nextState(self, currentState: State, word: str) -> Optional[State]:
        # Se parcurg tranzitiile PDA
        for (state, transition) in self.transitions.keys():
            # Se cauta o tranzitie pentru starea curenta
            if state == currentState:
                # Verificam daca tranzitia este pentru simbolul respectiv
                if word[0] == transition[WORD]:
                    # Se adauga pe stiva
                    if transition[PUSH] != EPS:
                        self.stack.push(transition[PUSH])

                    # Se returneaza starea urmatoare
                    return self.transitions[(state, transition)]

        # Nu se regaseste nicio tranzitie valida
        return None

    # Functiile de reducere
    def reduceLetter(self):
        letter = self.stack.pop()
        self.stack.push(Letter(letter))

    def reducePar(self):
        self.stack.pop()
        prevExpr = self.stack.pop()
        self.stack.pop()
        self.stack.push(Par(prevExpr))

    def reduceConcatenation(self):
        prevExpr1 = self.stack.pop()
        self.stack.pop()
        prevExpr2 = self.stack.pop()
        self.stack.push(Concatenation(prevExpr2, prevExpr1))

    def reduceReunion(self):
        prevExpr1 = self.stack.pop()
        self.stack.pop()
        prevExpr2 = self.stack.pop()
        self.stack.push(Reunion(prevExpr2, prevExpr1))

    def reduceKleen(self):
        self.stack.pop()
        prevExpr = self.stack.pop()
        self.stack.push(Kleen(prevExpr))

    # Se determina ce regula de reducere se poate aplica
    def reduce(self,word: str) -> bool:
        # Daca in varf se gaseste o liitera
        if self.stack.peek(0) in LETTERS:
            self.reduceLetter()
            return True
        
        # Daca in varf se gaseste o paranteza inchisa, dupa o expresie
        # si dupa o paranteza deschisa
        if (self.stack.peek(0) == CLOSE and isinstance(self.stack.peek(1), Expr) and 
            self.stack.peek(2) == OPEN):
            self.reducePar()
            return True
        
        # Daca in varf se gaseste kleenstar urmat de o expresie
        if self.stack.peek(0) == KLEENSTAR and isinstance(self.stack.peek(1), Expr) :
            self.reduceKleen()
            return True
        
        # Daca in varf se gaseste o expresie concatenare expresie si urmatorul caracter
        # din cuvant nu este kleenstar
        if (self.stack.peek(1) == CONCATENATION and isinstance(self.stack.peek(0), Expr) and 
            isinstance(self.stack.peek(2), Expr) and (len(word) == 0 or (KLEENSTAR != word[0]))):
            self.reduceConcatenation()
            return True
        
        # Daca in varf se gaseste o expresie reuniune expresie si urmatorul caracter
        # din cuvant nu este kleenstar sau concatenare
        if (self.stack.peek(1) == REUNION and isinstance(self.stack.peek(0), Expr) and
            isinstance(self.stack.peek(2), Expr) and
            (len(word) == 0 or ((KLEENSTAR != word[0]) and (CONCATENATION != word[0])))):
            self.reduceReunion()
            return True
        
        return False

    def parse(self, word: str) -> Optional[Expr]:
        currentState = self.initialState
        # Se parcurge cuvantul
        while len(word) != 0:
            # Se determina starea urmatoarea
            currentState = self.nextState(currentState, word)
            
            # Daca nu exista o astfel de tranzitie, se opreste
            # parcurgerea cuvantului
            if currentState is None:
                break
            word = word[1:]

            # Se aplica reducerile
            while self.reduce(word):
                continue

        # Daca stiva contine mai mult de un element
        # sau daca cuvantul nu este parcurs in totalitate
        # inseamna ca acest cuvant nu este acceptat
        if self.stack.size() != 1 or len(word) != 0 or currentState not in self.finalStates:
            return None

        # Se returneaza rezultatul
        return self.stack.pop()

# Se utilizeaza pentru retinerea informatiilor unui NFA
class NFA():
    def __init__(self):
        self.states: List[State] = []
        self.initialState: State
        self.finalStates: List[State] = []
        self.transitions: Dict[Tuple[State, TransitionNFA], List[State]] = {}

E_one_state = []

# Functia determina epsilon pentru starea s
# Rezultatul este salvat in E_one_state
def det_epsilon(delta_nfa,s):
    # Daca starea a fost deja vizitata
    if s in E_one_state:
        return None
    
    # Se adauga starea
    E_one_state.append(s)
    
    # Daca starea nu are "drumuri" catre alte
    # stari, cautarea nu isi mai are sens
    if s not in delta_nfa.keys():
        return None

    # Se parcurg toate drumurile starii respective
    for symbol,nex_st in delta_nfa[s].items():
        
        # Daca exista o epsilon tranzitie
        if symbol == "eps":

            # Se aplica recursiv functia pentru fiecare epsilon tranzitie
            for nxt in nex_st:
                det_epsilon(delta_nfa,nxt)
        
    return None

if __name__ == "__main__":
    # Se deschide fisierul
    f = open(sys.argv[1], "r")
    
    # Se citeste expresia regulata
    expr = f.readline().replace('\n','')

    lexic = []
    len_expr = len(expr)

    # Etapa lexicala (Se construieste o lista cu tokeni din input)
    for i in range(0,len_expr):
        token = expr[i]

        # Token este LETTERS
        if token in LETTERS:
            lexic.append(token)

            # Se verifica daca urmatorul token este CONCATENATION
            if i < len_expr - 1 and (expr[i+1] in OPEN or 
            expr[i+1] in LETTERS):

                lexic.append(CONCATENATION);    

        # Token este OPEN
        elif token in OPEN:
            lexic.append(OPEN)

        # Token este CLOSE
        elif token in CLOSE:
            lexic.append(CLOSE)

            # Se verifica daca urmatorul token este CONCATENATION
            if i < len_expr - 1 and (expr[i+1] in OPEN or 
            expr[i+1] in LETTERS):

                lexic.append(CONCATENATION)

        # Token este REUNION
        elif token in REUNION:
            lexic.append(REUNION)
        
        # Token este KLEENSTAR
        elif token in KLEENSTAR:
            lexic.append(KLEENSTAR)

            # Se verifica daca urmatorul token este CONCATENATION
            if i < len_expr - 1 and (expr[i+1] in OPEN or 
            expr[i+1] in LETTERS):

                lexic.append(CONCATENATION)

    # Etapa sintactica
    
    #Generare arbor de parsare
    pda = PDA()
    arbore = pda.parse(lexic)
    
    #Se construieste nfa
    nfa : NFA= arbore.GenNFA()

    delta_nfa = {}
    sigma = []

    # Se fac modificari a datelor din nfa, pentru a se aplica 
    # algoritmul dezvoltat la tema 2(NFA -> DFA)
    for (current_state, symbol), next_states in nfa.transitions.items():
        # Se modifica simbolul de epsilon
        if symbol == '':
            symbol = 'eps'

        # Se determina alfabetul
        if (symbol not in sigma) and symbol != 'eps':
            sigma.append(symbol)

        # Se seteaza starea 0 ca fiind cea initiala
        if current_state == 0:
            current_state = nfa.initialState
        elif current_state == nfa.initialState:
            current_state = 0

        for n, i in enumerate(next_states):
            if i == 0:
                next_states[n] = nfa.initialState
            elif i == nfa.initialState:
                next_states[n] = 0 
        
        # Se construieste delta_nfa
        if str(current_state) not in delta_nfa.keys():
        
            # Daca nu exista, este adaugat un dictionar nou,
            # reprezentand "salturile" pentru starea respectiva
            next_step = {}                  
            next_step[symbol] = [str(x) for x in next_states]
            delta_nfa[str(current_state)] = next_step
        else:
            # Daca exista, se adauga noul simbol cu starile in care
            # se duce acesta
            delta_nfa[str(current_state)][symbol] = [str(x) for x in next_states]

    states = len(nfa.states)
    epsilon = {}
    # Se citesc starile finale
    finals = [str(x) for x in nfa.finalStates]

    # Se sorteaza in functie de cheie
    delta_nfa = dict(sorted(delta_nfa.items(), key = lambda kv:kv[0]))

    f2 = open(sys.argv[2], "w")

    # Se scriu numarul total de stari a nfa
    f2.write(str(len(nfa.states)) +'\n')
    
    # Se scriu starile finale a nfa
    for elem in finals:
        f2.write(str(elem) + ' ')
    
    f2.write('\n')

    # Se scriu drumurile intre starile nfa
    for stare,next_step in delta_nfa.items():
        for symbol,next_states in next_step.items():
            
            f2.write(str(stare) +' '+ 
            str(symbol) + ' ')

            for x in next_states:
                f2.write(str(x))
                if next_states[-1] != x:
                    f2.write(' ')
            f2.write('\n')

    # Se inchide fisierul
    f2.close()

    # Se creaza dictionarul epsilon, care memoreaza epsilon
    # tranzitiile pentru fiecare stare din nfa
    for elem in range(0,int(states)):
        E_one_state.clear()
        det_epsilon(delta_nfa,str(elem))
        epsilon[str(elem)] = E_one_state.copy()

    delta_dfa = {}

    # In dfa_states se adauga starile noi descoperite,
    # urmand ca apoi sa se extraga cate o stare
    dfa_states = []

    # In selected_states se momoreaza starile care au
    # fost deja prelucrate
    selected_states = []

    # Se prelucreaza epsilon de 0, pentru a avea starile compuse
    # formate in aceeasi ordine
    aux = list(map(int, epsilon['0']))
    aux.sort()
    aux = list(map(str, aux))

    # Se incepe cu starea formata de epsilon aplicat pe starea initiala 
    # (Starile compuse se noteza cu . intre ele, pentru a putea
    # deosebi starea 12 de starea compusa 1.2
    dfa_states.append('.'.join(aux))

    while dfa_states:
        
        # Se extrage cate o stare 
        current_state = dfa_states.pop()
        
        # Se verifica daca aceasta nu a fost deja prelucrata
        if current_state not in selected_states:
            selected_states.append(current_state)
        else:
            continue
        
        next_s = {}
        # Se ia fiecare simbol din alfabet
        for elem in sigma:
            sink = 0
            nxt = []
            # Se selecteaza fiecare stare, din cea curenta
            # Pentru starea compusa 0.1 , se ia prima data
            # starea 0 dupa starea 1.
            for state in current_state.split('.'):
                
                # Se verifica daca starea are drumuri care pleaca din ea
                if state not in delta_nfa.keys():
                    continue
                
                # Se extrage dictionarul cu drumuri pentru starea respectiva
                new_states = delta_nfa[state]

                # Se concateneaza epsilonul starilor in care se ajunge
                if elem in new_states.keys():
                    for i in new_states[elem]:
                        if i in epsilon.keys():
                            nxt = nxt + epsilon[i]
            
            # Se elimina duplicatele
            nxt = list(set(nxt))

            # Se convertesc in int si se sorteaza
            nxt = list(map(int, nxt))
            nxt.sort()

            # Se converteste la loc in string
            nxt = list(map(str, nxt))

            # Se adauga starea la care se ajunge
            next_s[elem] = '.'.join(nxt)

            # Se verifica daca starea nu a mai fost selectata
            if next_s[elem] not in selected_states:
                dfa_states.append(next_s[elem])

        # Se declara drumurile pentru noua stare in dfa
        delta_dfa[current_state] = next_s
    
    final_states_dfa = []

    # Se genereaza starile finale dfa
    # Se ia fiecare stare din noul dfa creat si se verifica
    # daca ea contine cel putin o stare finala a nfa
    for elem in selected_states:
        for i in elem.split('.'):
            if i in finals:
                if elem not in final_states_dfa:
                    final_states_dfa.append(elem)

    # Se deschide fisierul pentru scriere
    f2 = open(sys.argv[3], "w")

    # Se scriu numarul total de stari a dfa
    f2.write(str(len(selected_states)) +'\n')
    
    # Se scriu starile finale a dfa
    for elem in final_states_dfa:
        f2.write(str(selected_states.index(elem)) + ' ')
    
    f2.write('\n')

    # Se scriu drumurile intre starile dfa
    for stare,next_step in delta_dfa.items():
        for symbol,next_states in next_step.items():
            
            f2.write(str(selected_states.index(stare)) +' '+ 
            str(symbol) + ' '+ str(selected_states.index(next_states)) +'\n')

    # Se inchide fisierul
    f2.close()
    



            
            




    