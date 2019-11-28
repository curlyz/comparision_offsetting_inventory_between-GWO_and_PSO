# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:15:12 2019

@author: USER
"""
import copy
import xlrd 
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

wb = xlrd.open_workbook('data.xlsx', on_demand=True)
sheets=(wb.sheet_names())
df=pd.read_excel('data.xlsx',sheet_name=sheets)
sh = '200'




def getdata(Quantity,Point,Cycle,Period):
    array = []
    result = []
    for i in range(Period):
        mod = i%Cycle
        if mod == 0:
            temp = Quantity
        if mod != 0:
            temp = round(-(Quantity/Cycle)*mod  + Quantity,2)
        array.append(temp)
    for j in range(Period):
        result.append(array[j-Point])
    return result


def initialization(num_searchagent, Ub, Lb):
    Positions=np.zeros((num_searchagent, len(Ub)),dtype = int)
    dim=len(Lb);
    for i in range(num_searchagent):
        for j in range(dim):
            Positions[i][j]=(np.random.randint(low=Lb[j],high=Ub[j]))
    return Positions

def Cost(arr_Quantity,arr_Point,arr_Cycle,int_Period):
    data = []
    sum_temp = 0
    arr_temp = []
    for i in range(len(arr_Quantity)):
        data.append(getdata(arr_Quantity[i],arr_Point[i],arr_Cycle[i],int_Period))
    for j in range(int_Period):
        for k in range(len(data)):    
            sum_temp+=data[k][j]
        arr_temp.append(sum_temp)
        sum_temp = 0
    return max(arr_temp)



def GWO(SearchAgents_no,Max_iter,ub,lb,dim):
    
    Alpha_pos=np.zeros(dim)
    Alpha_score=np.inf
    
    Beta_pos=np.zeros(dim)
    Beta_score=np.inf
    
    Delta_pos=np.zeros(dim)
    Delta_score=np.inf
    
    Positions=initialization(SearchAgents_no,ub,lb)
	
    Convergence_curve=np.zeros(Max_iter+1)
    l=0
    while l<Max_iter:
        fitness = []
        for i in range(0,SearchAgents_no):
            Flag4ub=Positions[i]>ub
            Flag4lb=Positions[i]<lb
            Positions[i]=(Positions[i]*(~(Flag4ub+Flag4lb)))+ub*Flag4ub+lb*Flag4lb
#            print(Positions[i])
            fitness.append(Cost(Quan,Positions[i],Cycle,Period))
            
            
            
        for i in range(0,SearchAgents_no):
            if fitness[i]<Alpha_score:
                Alpha_score=fitness[i]
                Alpha_pos=Positions[i].copy()
            if ((fitness[i]>Alpha_score) and (fitness[i]<Beta_score)):
                Beta_score=fitness[i]
                Beta_pos=Positions[i].copy()
            if (fitness[i]>Alpha_score) and (fitness[i]>Beta_score) and (fitness[i]<Delta_score):
                Delta_score=fitness[i]
                Delta_pos=Positions[i].copy()
        #a=10-l*l*((10)/(Max_iter*Max_iter))
        a=2-l*((2)/(Max_iter))
        
        for i in range(0,SearchAgents_no):
            for j in range(len(Positions[0])):
                r1=random.random()
                r2=random.random()
                A1=2*a*r1-a
                C1=2*r2
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i][j])
                X1=Alpha_pos[j]-A1*D_alpha
#                rand=np.random.rand()
#                if rand<0.5:
#                    D_alpha=np.random.rand()*np.sin(np.random.rand())*abs(C1*Alpha_pos[j]-Positions[i][j])
#                else:
#                    D_alpha=np.random.rand()*np.cos(np.random.rand())*abs(C1*Alpha_pos[j]-Positions[i][j])
#                X1=Alpha_pos[j]-A1*D_alpha
                
                
                
                
                r1=random.random()
                r2=random.random()
                A2=2*a*r1-a
                C2=2*r2
                D_beta=abs(C2*Beta_pos[j]-Positions[i][j])
                X2=Beta_pos[j]-A2*D_beta
                
                
                
                
                
                r1=random.random()
                r2=random.random()
                A3=2*a*r1-a
                C3=2*r2
                D_delta=abs(C3*Delta_pos[j]-Positions[i][j])
                X3=Delta_pos[j]-A3*D_delta
                
                
#                wr1 = random.random()
#                wr2 = random.random()
#                wr3 = random.random()
#                
#                w1 = round((wr1/(wr1 + wr2 + wr3)),2)*10
#                w2 = round((wr2/(wr1 + wr2 + wr3)),2)*10
#                w3 = round((wr3/(wr1 + wr2 + wr3)),2)*10
#                
#                
#                Positions[i][j]=round((w1*X1+w2*X2+w3*X3)/3,0)
                
                Positions[i][j]=round((X1+X2+X3)/3,0)
        
        Convergence_curve[l]=Alpha_score
        print('Iteration:',l,'-Obj - ',Alpha_score,'- w -', a)
        l+=1
        
    Convergence_curve[l] = Convergence_curve[l-1]
    return Alpha_score, Alpha_pos, Convergence_curve
    


Quan = df[sh]['Quan']
Cycle = df[sh]['Cycle']

SearchAgents_no=30
Max_iter=100

Ub = np.array([max(Cycle)]*len(Cycle))
Lb = np.array([0]*len(Cycle))
dim=len(Lb)
Period = 300





Best_score, Best_pos, CC=GWO(SearchAgents_no,Max_iter,Ub,Lb,dim)
print(type(CC))
plt.plot(CC)
plt.xlabel('Iteration')
plt.ylabel('Obj Value')
plt.title('Convergence rate ' +str(Best_score))
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()
