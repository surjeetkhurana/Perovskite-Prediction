# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:16:55 2023

@author: Surjeet
"""

from pymatgen.core import Species, Element
from OxidationNumberCalculator import processI
#Element("O").oxidation_states
 
import numpy as np
import pymatgen

from pymatgen.io.vasp import Poscar
#poscar = Poscar.from_file("CaTiO3Real.poscar")
#structure = poscar.structure


ls = []
OSlist = []

ls1 = []
#tructure.to(filename = "Prototype.poscar")

firsts = ["Li", "Na", "K","Rb", "Cs","Be","Mg","Ca","Sr","Ba"]
seconds = ["Sc","Y","Ti","Zr","Hf","V","Nb","Ta","Cr","Mo","W","Mn","Tc","Re","Fe","Ru","Os","Co","Rh","Ir","Ni","Pd",
           "Pt","Cu","Ag","Au","Zn","Cd","Hg","Al","Ga","In","Tl","Ge","Sn","Pb","As","Sb","Bi","Se","Te"]
thirds = ["P","O","S","F","Cl","Br","I"]




fixed_anions = {"Li": 1,
                "Na": 1,
                "K":1,
                "Rb":1,
                "Cs":1,
                "Be": 2,
                "Mg": 2,
                "Ca": 2,
                "Sr": 2,
                "Ba": 2,
    } 

fixed_Cations = {"O": -2,
                 "S": -2,
                 "F": -1,
                 "Cl": -1,
                 "Br": -1,
                 "I": -1,
                 "P":-1,
    }






i=0
j=0
k=0
num = 0
for first in firsts:
    #print(Firsts[i])
    for second in seconds:
        
        #print(seconds[j])
        for third in thirds:
            #print(thirds[k])
            fname = firsts[i] + seconds[j] + thirds[k] + "3"
            #print(fname)
            #filename = str(num) +"-" + fname + "3.poscar"
            #structure.to(filename = filename)
            #number = processI(fname,1)
            #OSlist.insert(1,number)  
            ls.insert(1,fname)
            
            
            
            
            oxs1 = Element(firsts[i]).oxidation_states
            oxs2 = Element(seconds[j]).oxidation_states
            oxs3 = Element(thirds[k]).oxidation_states
            
            
            
            
            #ar = np.array(number[1:2])
            one = fixed_anions[firsts[i]]
            
            #two = 2
            three = fixed_Cations[thirds[k]]
            
            two = -3*three - one
            
            if (one in oxs1) and (two in oxs2) and (three in oxs3):
                #print("Oxidation State Available:", fname)
                Sp = Species(firsts[i], oxidation_state = one)
                Sp2 = Species(seconds[j], oxidation_state = two)
                Sp3 = Species(thirds [k], oxidation_state = three)
                
                rA = Sp.ionic_radius
                rB = Sp2.ionic_radius
                rX = Sp3.ionic_radius
                nA = one
                
                if(rA != None) and (rB != None) and (rX != None):
                    
                    rA = float(Sp.ionic_radius)
                    rB = float(Sp2.ionic_radius)
                    rX = float(Sp3.ionic_radius)
                    nA = float(one)
                    
                    
                    ln = np.log(rA/rB)
                    #tau = (rX/rB) - (nA*(nA - ((rA/rB)/ln)))
                    tau = rX/rB - nA * (nA - (rA/rB)/np.log(rA/rB))
                    print (fname, " tau =", tau)
                    
                    str1 = fname + str(tau)
                    ls1.insert(1, str1)
            else:
                print("Not Available:", fname)
                
            
               
            
            
            #ionicRadius =  
            
            
            k=k+1
            num = num+1
            #print("K=",k)
        j=j+1
        k=0
        #print("j=",j)
    i=i+1
    j=0            
    #print("i=",i)



