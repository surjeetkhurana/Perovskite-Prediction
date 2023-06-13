# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 05:40:57 2023

@author: Surjeet
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:32:23 2022

@author: Surjeet
"""

import pymatgen

from pymatgen.io.vasp import Poscar

from pymatgen.core import Species, Element

import numpy as np
import pymatgen


ls = []

#tructure.to(filename = "Prototype.poscar")

firsts = ["Li", "Na", "K","Rb", "Cs","Be","Mg","Ca","Sr","Ba"]
seconds = ["Sc","Y","Ti","Zr","Hf","V","Nb","Ta","Cr","Mo","W","Mn","Tc","Re","Fe","Ru","Os","Co","Rh","Ir","Ni","Pd",
           "Pt","Cu","Ag","Au","Zn","Cd","Hg","Al","Ga","In","Tl","Ge","Sn","Pb","As","Sb","Bi","Se","Te"]
thirds = ["P","O","S","F","Cl","Br","I"]
fourths = ["Sc","Y","Ti","Zr","Hf","V","Nb","Ta","Cr","Mo","W","Mn","Tc","Re","Fe","Ru","Os","Co","Rh","Ir","Ni","Pd",
           "Pt","Cu","Ag","Au","Zn","Cd","Hg","Al","Ga","In","Tl","Ge","Sn","Pb","As","Sb","Bi","Se","Te"]


    
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


ls = []
OSlist = []

ls1 = []
    
    
#6th Space Group 
#poscar = Poscar.from_file("Cs2AgSbBr6Real.poscar")
#structure = poscar.structure





i=0
j=0
k=0
l = 0
num = 14350

for first in firsts:
    for second in seconds:
        for fourth in fourths:    
            for third in thirds:
                if seconds[j] != fourths[l]:
                    fname = firsts[i] + "2" + seconds[j] + fourths[l] + thirds[k] + "6"
                    
                    one_oxs = Element(firsts[i]).oxidation_states
                    two_oxs = Element(seconds[j]).oxidation_states
                    three_oxs = Element(fourths[l]).oxidation_states
                    four_oxs = Element(thirds[k]).oxidation_states
                    
                    one = fixed_anions[firsts[i]]
                     
                    four = fixed_Cations[thirds[k]]
                    
                    for two_ox in two_oxs:
                        for three_ox in three_oxs:
                            if ((one*2 + two_ox + three_ox + four*6) == 0):
                                two = two_ox
                                three = three_ox
                            
                                sp = Species(firsts[i], oxidation_state = one)
                                sp2 = Species(seconds[j], oxidation_state = two)
                                sp3 = Species(fourths[l], oxidation_state = three)
                                sp4 = Species(thirds[k], oxidation_state = four)
                                
                               
                                rA = sp.ionic_radius
                                rB = sp2.ionic_radius 
                                rC = sp3.ionic_radius
                                rX = sp4.ionic_radius
                                nA = one
                                
                                
                                if(rA != None) and (rB != None) and (rC != None) and (rX != None):
                                    
                                    rA = float(rA)
                                    rB = float((rB + rC)/2)
                                    rX = float(rX)
                                    nA = float(one)
                                    
                                    tau = rX/rB - nA * (nA - (rA/rB)/np.log(rA/rB))
                                    print (fname, " tau =", tau)
                                    
                                    str1 = fname + " " + str(tau)
                                    
                                    break
                            else:
                                str1 = fname + " " + "0"
                                #ls1.insert(1, str1)
                                break
                                    
                            break
                        
                        
                    
                   
                    
                            
                            
                            
                    ls1.insert(1, str1)
                    
                    ls.insert(1,fname)
                    num = num+1
                k=k+1
                #print("K=",k)
            l=l+1
            k=0
            #print("j=",j)
        j=j+1
        l=0
    i=i+1
    j=0            
    #print("i=",i)
    
    
