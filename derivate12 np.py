import numpy as np
import matplotlib.pyplot as plt
import math 
from functools import partial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class abstract_gate:
    def __init__(self) -> None:
        self.is_inital_variable = False
    def foward():
        pass
    def backward():
        pass
    def compute_operation(self):
        pass
    def get_prev_gate_var_list(self):
        pass


class variable(abstract_gate):#can be seen as a particula gate that returns his value, doesn't modify his gradient, and has only a [self] list of variables

    def __init__(self, name, value =1, is_constant =False) -> None:
        self.is_inital_variable = True
        self.is_constant = is_constant
        self.name = name
        self.init_value(value)
        

    def init_value(self, value):
        self.value = np.array([])
        self.value = np.append(self.value, value)
        lun:int = len(self.value)
        self.grad = np.ones((lun))
        print("self.value ", self.value, " lun ", lun, " grad ", self.grad)

    def stamp(self):
        print(self.name, " -> ", " value: ", self.value, " grad: ", self.grad)

    def compute_operation(self):
        return self.value
    
    def backward(self):
        pass

    def get_prev_gate_var_list(self):
        return [self]

    def get_grad(self):
        return self.grad

    def multiply_grad(self, gate_grad):
        print("name ", self.name, " is_costant ", self.is_constant,  " self grad ", self.grad, " gate grad ", gate_grad)
        self.grad *= gate_grad

#cioa

class abstract_gate_1_in(abstract_gate):
    def __init__(self, prev_gate_a:abstract_gate =None) -> None:
        self.is_inital_variable = False
        self.prev_gate_a:abstract_gate = prev_gate_a
        self.prev_gate_a_var_list:list[variable] = []


    def compute_operation(self):
        prev_a_ris:float

        prev_a_ris = self.prev_gate_a.compute_operation()
        self.prev_gate_a_var_list = self.prev_gate_a.get_prev_gate_var_list()

        gate_foward_result = self.foward(prev_a_ris)
        self.backward(prev_a_ris, self.prev_gate_a_var_list, gate_foward_result)
        return(gate_foward_result)
    

    def get_prev_gate_var_list(self):
        return self.prev_gate_a_var_list




class abstract_gate_2_in(abstract_gate):
    def __init__(self, prev_gate_a:abstract_gate = None, prev_gate_b:abstract_gate = None) -> None:
        self.is_inital_variable = False
        self.prev_gate_a:abstract_gate = prev_gate_a
        self.prev_gate_b:abstract_gate = prev_gate_b
        self.prev_gate_a_var_list:list[variable] = []
        self.prev_gate_b_var_list:list[variable] = []


    def compute_operation(self):
        prev_a_ris:float #is either the value of the variable a or the result of the prev a gates
        prev_b_ris:float

        prev_a_ris = self.prev_gate_a.compute_operation()
        #print("prev a ris: ", prev_a_ris)
        self.prev_gate_a_var_list = self.prev_gate_a.get_prev_gate_var_list()

        prev_b_ris = self.prev_gate_b.compute_operation()
        self.prev_gate_b_var_list = self.prev_gate_b.get_prev_gate_var_list()

        gate_foward_result = self.foward(prev_a_ris, prev_b_ris)
        self.backward(prev_a_ris, prev_b_ris, self.prev_gate_a_var_list, self.prev_gate_b_var_list, gate_foward_result)
        return gate_foward_result
    

    def get_prev_gate_var_list(self):
        return self.prev_gate_a_var_list + self.prev_gate_b_var_list
    


class plus_gate(abstract_gate_2_in):
    def foward(self,a:float,b:float):
        ris = a+ b
        print("plus_gate: ",ris)
        return ris
    
    def backward(self, prev_a_ris:float, prev_b_ris:float, prev_gate_a_var_list:list[variable], prev_gate_b_var_list:list[variable], gate_foward_result:float):
        pass
    


class multiply_gate(abstract_gate_2_in): 
    def foward(self,a:float, b:float):
        ris = a * b
        print("multiply_gate: ",ris)
        return ris
    
    def backward(self, prev_a_ris:np.array, prev_b_ris:np.array, prev_gate_a_var_list:list[variable], prev_gate_b_var_list:list[variable], gate_foward_result:float):
        for v_a in prev_gate_a_var_list:
            print("gradient of var_a: ", v_a.value, " *prev_b_ris: ", prev_b_ris)
            if(not v_a.is_constant):
                v_a.multiply_grad(prev_b_ris)

        for v_b in prev_gate_b_var_list:
            print("var_b: ", v_b.value, " *prev_a_ris: ", prev_a_ris)
            if(not v_b.is_constant):
                v_b.multiply_grad(prev_a_ris)



class e_gate(abstract_gate_1_in):
    def foward(self, a:np.array):
        ris = math.e**a
        print("e_gate: ", ris)
        return ris    

    def backward(self, prev_a_ris:float, prev_gate_a_var_list:list[variable], gate_foward_result):
        for v_a in prev_gate_a_var_list:
            if(not v_a.is_constant):
                v_a.multiply_grad(math.e**prev_a_ris)



class power_gate(abstract_gate_2_in):
    def foward(self, a:float, b:float):
        ris = a**b #a^b 
        print("power_gate: ", ris)
        return ris
    
    def backward(self, prev_a_ris:np.array, prev_b_ris:np.array, prev_gate_a_var_list:list[variable], prev_gate_b_var_list:list[variable], gate_foward_result:float):
        for v_a in prev_gate_a_var_list: 
            if(not v_a.is_constant):
                tp = prev_b_ris*(prev_a_ris**(prev_b_ris-1))  #b*(a^(b-1) #x^3 -> 3*(grad)^2
                print("tp: ", tp)
                v_a.multiply_grad(tp)

        for v_b in prev_gate_b_var_list:
            if(not v_b.is_constant):
                tp:np.array = np.array([])
                for el_a in prev_a_ris:
                    tp = np.append(tp, math.log(el_a, math.e))
                tp1:np.array = (prev_a_ris**prev_b_ris)*tp
                v_b.multiply_grad(tp1)



class sin_gate(abstract_gate_1_in):
    def foward(self, a:np.array):
        ris:np.array = np.array([])
        for el in a:
            ris = np.append(ris, math.sin(el))
        print("sin_gate: ", ris)
        return ris
    
    def backward(self, prev_a_ris:np.array, prev_gate_a_var_list:list[variable], gate_foward_result:np.array):
        for v_a in prev_gate_a_var_list:
            if(not v_a.is_constant):
                tp:np.array = np.array([])
                for el in prev_a_ris:
                    tp = np.append(tp, math.cos(el))
                v_a.multiply_grad(tp)
    


class cos_gate(abstract_gate_1_in):
    def foward(self, a:np.array):
        ris:np.array = np.array([])
        for el in a:
            ris = np.append(ris, math.cos(el))
        print("cos_gate: ", ris)
        return ris
    
    def backward(self, prev_a_ris:np.array, prev_gate_a_var_list:list[variable], gate_foward_result:np.array):
        for v_a in prev_gate_a_var_list:
            if(not v_a.is_constant):
                tp:np.array = np.array([])
                for el in prev_a_ris:
                    tp = np.append(tp, -math.sin(el))
                v_a.multiply_grad(tp)



class tan_gate(abstract_gate_1_in):
    def foward(self, a:np.array):
        ris:np.array = np.array([])
        for el in a:
            ris = np.append(ris, math.tan(el))
        print("tan_gate: ", ris)
        return ris
    
    def backward(self, prev_a_ris:np.array, prev_gate_a_var_list:list[variable], gate_foward_result:np.array):
        for v_a in prev_gate_a_var_list:
            if(not v_a.is_constant):
                tp:np.array = np.array([])
                for el in prev_a_ris:
                    tp = np.append(tp, 1/(math.cos(el))**2)
                v_a.multiply_grad(tp)



class cotan_gate(abstract_gate_1_in):
    def foward(self, a:np.array):
        ris:np.array = np.array([])
        for el in a:
            ris = np.append(ris, 1/math.tan(el))
        print("cotan_gate: ", ris)
        return ris
    
    def backward(self, prev_a_ris:np.array, prev_gate_a_var_list:list[variable], gate_foward_result:np.array):
        for v_a in prev_gate_a_var_list:
            if(not v_a.is_constant):
                tp:np.array = np.array([])
                for el in prev_a_ris:
                    tp = np.append(tp, -1/(math.sin(el))**2)
                v_a.multiply_grad(tp)



class log_gate(abstract_gate_2_in):
    def foward(self, a:np.array, b:np.array):
        ris:np.array = np.array([])
        for el_ix in range(len(a)):
            ris = np.append(ris, math.log(a[el_ix], b[el_ix]))
        print("log(a,b): ", ris)
        return ris
    
    def backward(self, prev_a_ris:np.array, prev_b_ris:np.array, prev_gate_a_var_list:list[variable], prev_gate_b_var_list:list[variable], gate_foward_result:float):
        for v_a in prev_gate_a_var_list:
            if(not v_a.is_constant):
                if(not v_a.is_constant):
                    tp:np.array = np.array([])
                    for el_ix in range(len(prev_a_ris)):
                        tp = np.append(tp, prev_a_ris[el_ix]*math.log(prev_b_ris[el_ix], math.e))
                    v_a.multiply_grad(1/tp)

        for v_b in prev_gate_b_var_list:
            if(not v_b.is_constant):
                tp1:np.array = np.array([])
                tp2:np.array = np.array([])
                for el_ix in range(len(prev_a_ris)):
                    tp1 = np.append(tp1, -math.log(prev_a_ris[el_ix], math.e))
                    tp2 = np.append(tp2, prev_b_ris[el_ix]*(math.log(prev_b_ris[el_ix], math.e)**2))

                tp3 = tp1/tp2
                v_b.multiply_grad(tp3)
        


x = variable("x",np.array([4,2]))
y = variable("y",np.array([4,3]))

gate_prova = tan_gate(x)
gate_prova.compute_operation()

print("controllo finale per d8")
x.stamp()
y.stamp()



class indexed_gate():
    def __init__(self, gate, ix) -> None:
        self.gate:abstract_gate =gate
        self.ix:int =ix

class expression_decoder():# last of + > last of * > last pf ^

    def __init__(self) -> None:
        self.gate_tree:abstract_gate = None
        self.expression_variable_list:list[variable] = None
        self.foward_result:np.array = None


    def check_brackets(self, input_str): #True == ok, False == error
        open_brackets_counter:int = 0
        for el in input_str:
            if el == "(":
                open_brackets_counter +=1
            
            if el == ")":
                if open_brackets_counter > 0:
                    open_brackets_counter -= 1
                else:
                    return False

        if open_brackets_counter != 0:
            return False
        
        return True


    def get_variables_list_and_compute(self, value_list:list[float] = []):

        variable_list_len = len(self.expression_variable_list)

        if(self.gate_tree != None and variable_list_len != 0):
            for ix in range(min(len(value_list), variable_list_len)):
                self.expression_variable_list[ix].init_value(value_list[ix]) 
            self.foward_result = self.gate_tree.compute_operation()
            print("\n final result: ", self.foward_result)


            print("variable result: ")
            if(len(self.expression_variable_list) == 0):
                print("None")
            for var in self.expression_variable_list:
                var.stamp()

    def plot_graph(self): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        selected_variable_ix = 0
        x = self.expression_variable_list[selected_variable_ix].value
        y = self.foward_result

        y_grad = self.expression_variable_list[selected_variable_ix].grad


        plt.ion()
        fig = plt.figure()
        diag = fig.add_subplot(1,1,1)
        diag.set_xlabel(self.expression_variable_list[selected_variable_ix].name)
        diag.set_ylabel("result")
        diag.set_ylim([0,1000])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()
        print("after")

        manager = plt.get_current_fig_manager()
        manager.window.attributes('-topmost', 1)

        
        res_line = diag.plot(x,y)[0]
        res_line.set_color(color="BLUE")
        grad_line = diag.plot(x,y_grad)[0]
        grad_line.set_color(color="RED")

        plt_in:str = "3"
        while(plt_in != "0" and plt_in != "q"): #!!!!!!!!!!!!!!!!!!!!!!!!!!

            print("res = 1, grad = 2, both = 3, other_var = var_name, exit = q")

            if(plt_in == "1"):
                res_line.set_alpha(1)
            elif(plt_in == "2"):
                grad_line.set_alpha(1)
            elif(plt_in == "3"):
                res_line.set_alpha(1)
                grad_line.set_alpha(1)
            else:
                variable_exists = False
                for var in self.expression_variable_list:
                    print("ciao ", var.name == plt_in)
                    if var.name == plt_in:
                        diag.set_xlabel(var.name)
                        res_line.set_xdata(var.value)
                        grad_line.set_ydata(var.grad) #here #! not grad_line.set_ydata = var.grad #here dumbass
                        variable_exists = True
                        print(var.grad)
                        res_line.set_alpha(1)
                        grad_line.set_alpha(1)
                    
                    if(not variable_exists):
                        print("charachter not recognized")

            plt_in = input("enter sandman> ")
            
            res_line.set_alpha(0)
            grad_line.set_alpha(0)
            
        plt.close()
        
        #res_line.remove()
        #fig.canvas.draw()

        


    def decode_expression(self, input_str:str):

        if(not self.check_brackets(input_str)):
            print("Unable to read, Bracket Error")
            return
        
        for ix in range(len(input_str)-1, 0, -1):
            print("ix ", ix)
            if(input_str[ix] == " "):
                input_str = input_str[0:ix] + input_str[(ix+1):]

        self.expression_variable_list:list[variable] = []
        self.gate_tree = self.expression_decoder_helper(input_str, self.expression_variable_list)

        variable_list_len = len(self.expression_variable_list)


        print("\n", "varibles: ")
        if(variable_list_len == 0):
            print("None")
        for var in self.expression_variable_list:
            print(var.name)
        print()


    def is_numeric(self, input_str:str):
        for el in input_str:
            if(el.isalpha()):
                return False
        return True
        


    def expression_decoder_helper(self, input_str:str, expression_variable_list:list[variable]):
        
        if input_str[0] == "(" and input_str[-1] == ")": # (1+2) == 1+2 -> rimuove parentesi se inutili
            if(self.check_brackets(input_str[1:-1])):
                input_str = input_str[1:-1]


        new_gate:abstract_gate

        open_bracket_counter:int = 0
        last_plus_ix:int= -1
        last_mul_ix:int= -1
        last_div_ix:int= -1
        last_pow_ix:int= -1
        last_op_ix:int= -1

        for i in range(len(input_str)):
                op = input_str[i]

                if(op == "("):
                    open_bracket_counter +=1
                if(op == ")"):
                    open_bracket_counter -=1

                if(open_bracket_counter == 0): #le parentesi (...) sono considerate numeri nell'espressione complessiva 
                    if(op == "+"):
                        last_plus_ix = i
                        last_op_ix = last_plus_ix
                    if(op == "*"):
                        last_mul_ix = i
                    if(op == "/"):
                        last_div_ix = i
                    if(op == "^"):
                        last_pow_ix = i
                    


        if(last_plus_ix == -1 and last_mul_ix == -1 and last_pow_ix == -1 and last_div_ix == -1): #sia log(a,b) che le funzioni trigonometriche sono considerate numeri nell espressione complessiva
            
            if(len(input_str)>3 and not self.is_numeric(input_str)): 
                first_operator:str = ""
                for ix in range(3):
                    first_operator += input_str[ix]

                if(first_operator == "log"): #la funzione log(a,b) spezza la stringa il modo particolare -> ricorsivo con 2 argomenti -> DIVIDI et IMPERA
                    ix_virgola = -1
                    for ix in range(3,len(input_str)):
                        if(input_str[ix] == ","):
                            ix_virgola = ix

                    left_str = input_str[ix_virgola+1:-1]
                    result_left = self.expression_decoder_helper(left_str, expression_variable_list)
                    right_str = input_str[4:ix_virgola] #quarto compreso, ix , non compreso
                    result_right = self.expression_decoder_helper(right_str, expression_variable_list)
                    
                    new_gate = log_gate()
                    new_gate.prev_gate_a = result_right
                    new_gate.prev_gate_b = result_left


                else: #le funzioni trigonometriche prenodono solo 1 arg -> caso ricorsivo con 1 argomento

                    if(first_operator == "sin"):
                        new_gate = sin_gate()
                    if(first_operator == "cos"):
                        new_gate = cos_gate()
                    if(first_operator == "tan"):
                        new_gate = tan_gate()
                    if(first_operator == "cot"):
                        new_gate = cotan_gate()

                    result_in:abstract_gate = self.expression_decoder_helper(input_str[3:], expression_variable_list)
                    new_gate.prev_gate_a = result_in
                    
                
            else: #Caso base singola varibile o cifra
                if(self.is_numeric(input_str)):
                    print("new const ", input_str)
                    return variable(input_str, float(input_str), is_constant=True)
                else:
                    print("new var: ", input_str) #!New addition!!
                    var:variable = None
                    is_present:bool = False
                    for el in expression_variable_list:
                        if(input_str == el.name):
                            var = el
                            is_present = True

                    if(not is_present):
                        var = variable(input_str)
                        expression_variable_list.append(var)
                    return var

        else:#gli operatori +-*/ usano logica simile per spezzare la stringa -> ricorsivo con 2 argomenti -> DIVIDI et IMPERA

            is_last_op_div:bool = False

            if(last_plus_ix != -1):
                new_gate = plus_gate()
            elif(last_mul_ix != -1):
                last_op_ix = last_mul_ix
                new_gate = multiply_gate()
            elif(last_div_ix != -1): #problema 2/3*2 = (2/3) * 2 != 2/(3*2) #la \ ha sempre la priorità matematica? SI 2*3/2 == 2*(3/2)
                last_op_ix = last_div_ix
                new_gate = multiply_gate()
                is_last_op_div = True
            elif(last_pow_ix != -1):
                last_op_ix = last_pow_ix
                new_gate = power_gate()
            

            print("splitting char ",last_op_ix, " > ", input_str[last_op_ix])
            right_str = input_str[(last_op_ix+1):]
            left_str = input_str[0:last_op_ix]

            print("left string ", left_str)
            result_left = self.expression_decoder_helper(left_str, expression_variable_list)
            print("right string ", right_str)
            result_right = self.expression_decoder_helper(right_str, expression_variable_list) #da ricalcolare la queue

            if(is_last_op_div): #1/2 == 1*(2^(-1))
                result_right = power_gate(result_right, variable("-a", -1, True))

            new_gate.prev_gate_a = result_left
            new_gate.prev_gate_b = result_right

        return new_gate



prova = expression_decoder()

#st = "1+2*3^2+3"
#st = "((1+x)*3)^2"
#st = "((1+x)*3)^log(3,7)"
#st = "(x^2+y)*2.7^log(2,3)" #f(variables) = expression, f'(varibles) = df/d var[i]
st = "x*x"

prova.decode_expression(st)
#variable_list = [np.arange(-100, 100), np.arange(-100, 100)]
variable_list = [np.array([3])]
prova.get_variables_list_and_compute(variable_list)
#prova.plot_graph()


#how to use the graph??????

