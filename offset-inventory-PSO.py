# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:30:57 2019

@author: TanPham
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
sh = '10'


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


def CheckVelocity(velocity,vmax,vmin):
    temp1=velocity>vmax
    temp2=velocity<vmin
    index1,index2=[],[]
    for i in temp1:
        if i==True:
            index1.append(i)
    for j in temp2:
        if j==True:
            index2.append(j)
    return temp1, temp2

def CheckPosition(position,ub,lb):
    temp1=position>ub
    temp2=position<lb
    index1,index2=[],[]
    for i in temp1:
        if i==True:
            index1.append(i)
    for j in temp2:
        if j==True:
            index2.append(j)
    return temp1, temp2
    
    

class Particle:
    
    def __init__(self):
        self.Position=-1
        self.Velocity=-1
        self.Obj_val=0
        self.Personalbest_P=-1
        self.Personalbest_Value=-1
    
    def __repr__(self):
        return str(self.Position)

class Swarm:
    
    def __init__(self):
        self.Particle_list=[]
        self.Global_Best_Pos=[]
        self.Global_Best_Value=np.inf
    
    def Create_Swarm(self, no_P):
        for i in range(no_P):
            self.Particle_list.append(Particle())
        return self.Particle_list
    
    def Initialization(self,no_P):
            for i in range(no_P):
                self.Particle_list[i].Position=np.random.uniform(low=0, high=max(Cycle), size=len(Cycle))
                self.Particle_list[i].Velocity=np.zeros(dim)
                self.Particle_list[i].Personalbest_P=np.zeros(dim)
                self.Particle_list[i].Personalbest_Value=np.inf
            self.Global_Best_Pos=np.zeros(dim)
            self.Global_Best_Value=np.inf
            return self.Particle_list, self.Global_Best_Pos, self.Global_Best_Value

def main():
    
    CC=np.zeros(maxIter+1)
    for i in range(maxIter):
        for k in range(noP):
            currentX=swarm.Particle_list[k].Position.copy()
            
            for m in range(len(currentX)):
                temp = currentX[m]
                currentX[m] = round(temp,0)
            
            currentX = [int(x) for x in currentX]
        
            
            swarm.Particle_list[k].Obj_val=Cost(Quan,currentX,Cycle,Period)
            if swarm.Particle_list[k].Obj_val<swarm.Particle_list[k].Personalbest_Value:
                swarm.Particle_list[k].Personalbest_P=currentX.copy()
                swarm.Particle_list[k].Personalbest_Value=swarm.Particle_list[k].Obj_val
            if swarm.Particle_list[k].Obj_val<swarm.Global_Best_Value:
                swarm.Global_Best_Pos=currentX.copy()
                swarm.Global_Best_Value=swarm.Particle_list[k].Obj_val
        'Update'
        w=wMax-i*((wMax-wMin)/maxIter)
        
        for k in range(noP):
            c1=1.2-swarm.Particle_list[k].Obj_val/swarm.Global_Best_Value
            c2=0.5+swarm.Particle_list[k].Obj_val/swarm.Global_Best_Value
            
            swarm.Particle_list[k].Velocity=w*swarm.Particle_list[k].Velocity\
        + c1*np.random.rand(dim)*(swarm.Particle_list[k].Personalbest_P-swarm.Particle_list[k].Position)\
        + c2*np.random.rand(dim)*(swarm.Global_Best_Pos-swarm.Particle_list[k].Position)
            'Check velocity'
            index1,index2=CheckVelocity(swarm.Particle_list[k].Velocity,vMax,vMin)
            swarm.Particle_list[k].Velocity[index1]=vMax[index1]
            swarm.Particle_list[k].Velocity[index2]=vMin[index2]
            swarm.Particle_list[k].Position+=swarm.Particle_list[k].Velocity
            'Check position'
            index3,index4=CheckPosition(swarm.Particle_list[k].Position,ub,lb)
            swarm.Particle_list[k].Position[index3]=ub[index3]
            swarm.Particle_list[k].Position[index4]=lb[index4]
        
        CC[i]=swarm.Global_Best_Value
        print('Iteration:',i,'-Obj - ',swarm.Global_Best_Value,'- w -',w)
    print(swarm.Global_Best_Pos)
    return CC
            

Quan = df[sh]['Quan']
Cycle = df[sh]['Cycle']
Period = 300

ub = np.array([max(Cycle)]*len(Cycle))
lb = np.array([0]*len(Cycle))
dim=len(lb)

noP = 40
maxIter = 100


wMax = 1.5
wMin = 0.1
vMax = (ub - lb) * 0.2
vMin  = -vMax
swarm=Swarm()
swarm.Create_Swarm(noP)
swarm.Initialization(noP)
#print(s.Particle_list,s.Global_Best_Pos,s.Global_Best_Value)
#particle=Particle()
AA=main()

plt.plot(AA)
plt.xlabel('Iteration')
plt.ylabel('Obj Value')
#plt.title('Convergence rate ' +str(Best_score))
fig = plt.gcf()
fig.set_size_inches(20, 15)
plt.show()
