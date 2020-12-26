# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:42:05 2020

@author: WNZ
"""


import numpy as np
import matplotlib.pyplot as plt










################################################################################
#####定义函数搜获隔根区间
###################################################################################
def search(f,a,h,n):
#%功能：找到发f(x)在区间[a,+∞)上所有隔根区间
#%输入：f(x):所求方程函数；[a,+∞):有根区间；h:步长；n:所需要区间个数
#%输出：隔根区间[c,d] 
    c=np.empty((n,1))
    d=np.empty((n,1))
    k=0
    while k<=n-1:
        if f(a)*f(a+h)<=0:
            c[k]=a
            d[k]=a+h
            k=k+1
        a=a+h
    return c,d    


#############################################################################
###牛顿迭代法及其修改形式
##############################################################################
def newton(fname,dfname,x0,tol,N,m):
#
# 输入：初值x0,最大迭代步数N，误差限tol，m=1为牛顿迭代法，m>1为修改的牛顿迭代法
# 输出：近似根y，迭代次数k    
    y=x0
    x0=y+2*tol
    k=0
    while np.abs(x0-y)>tol and k<N:
        k=k+1
        x0=y
        y=x0-m*fname(x0)/dfname(x0)
        
    if k==N:
        print("warning")
        
    return y,k





##############################################################################
###求解一维特征值 
##############################################################################
def eigen_value_solution(eta,L,var,Num_Root):
#% 输入：eta:相关长度   L：区域长度   var:方差   Num_Root: 前N个实根
#% 输出：Lamda：特征值  w0：对应特征方程正实根   
    w0=np.empty((Num_Root,1))
    lamda=np.empty((Num_Root,1))
    cumulate_lamda=np.empty((Num_Root,1))

    ##############################################################################
    ##定义方程形式
    #########################################################################
    def ff(x):
        ff=(eta**2*x**2-1)*np.sin(x*L)-2*eta*x*np.cos(x*L)
        return ff
    
    ##############################################################################
    ##定义方程导数形式
    #########################################################################
    def dff(x):
        dff=(2*eta**2*x-1)*np.sin(x*L)+(eta**2*x**2-1)*np.cos(x*L)*L-2*eta*np.cos(x*L)+2*eta*x*np.sin(x*L)*L
        return dff
    
    
    
    ##用函数搜索隔根区间
    c,d=search(ff,0.00001,0.00001,Num_Root)
    w00=(c+d)/2
    
   ##%用牛顿法精确求解
    for i in range(Num_Root):
       w0[i],k= newton(ff,dff,w00[i],1e-8,10000,1)
        
    
    ## 根据特征方程正实根，计算特征值λ（Lamda） %%%%
    for flag in range(Num_Root):
        lamda[flag]=2*eta*var/(eta**2*w0[flag]**2+1)
        if flag==0:
            cumulate_lamda[flag]=lamda[flag]
        else:
            cumulate_lamda[flag]=lamda[flag]+cumulate_lamda[flag-1]
        
    return lamda,w0,cumulate_lamda
    
    
def sort_lamda(lamda_x,w0_x,lamda_y,w0_y,domain,var,weight):
#  Sort_Lamda 二维特征值组合并排序
#  输入 Lamda_x，w0_x:x方向特征值以及对应特征方程实根   Lamda_y,w0_y:y方向特征值以及对应特征方程实根
#       Domain：矩形域范围  var：方差
#       N_X,N_Y:X,Y方向特征值个数
#  返回 lamda:按递减顺序排列的二维特征值
#       w_x,w_y:特征值对应特征方程在不同方向上的正实根
#       n：特征值截断个数，权重weight, cum_lamda:特征根累计值
    n_x=len(w0_x)
    n_y=len(w0_y)
    num=n_x*n_y
    lamda_2d=np.zeros((num,1))
    flag=0
    lamda_index=list()
    for i in range(n_x):
        for j in range(n_y):
            lamda_2d[flag]=lamda_x[i]*lamda_y[j]/var
#            print(lamda_2d)
            lam_ind=[lamda_2d[flag],i,j]
#            print(lam_ind)
            
            lamda_index.append(lam_ind)
#            print(lamda_index)
            flag=flag+1
    
    lamda_index_sorted=sorted(lamda_index, key = lambda x: x[0], reverse=True)
    
    sum_lamda=np.zeros((num,1))
    lamda_all=np.zeros((num,1))
    w_x_all=np.zeros((num,1))
    w_y_all=np.zeros((num,1))
#    sum_lamda[0]=lamda_index_sorted[0][0]
    
    lab=1
    
    for k in range(num):
        lamda_all[k]=lamda_index_sorted[k][0]
        w_x_all[k]=w0_x[lamda_index_sorted[k][1]]
        w_y_all[k]=w0_y[lamda_index_sorted[k][2]]
        
#        print(w_x_all)
        if k==0:
            sum_lamda[k]=lamda_index_sorted[k][0]
        else:
            sum_lamda[k]=sum_lamda[k-1]+lamda_index_sorted[k][0]
            
        if lab and sum_lamda[k]/domain/var>=weight:
            n=k+1
#            print(n)
            lab=0 
#    print(k)
#    print(num)
            
    fig, ax1 = plt.subplots()
    ax1.plot(range(num),lamda_all/domain/var)
    
#    ax1.set_xlim([1,n_eigen])
#    plt.legend()
    ax1.set_xlabel('n')
    ax1.set_ylabel('Lamda 2D / (Domain*Var)')  
    plt.title('Series of Engenvalues in 2 Demensions')
    
    
    fig, ax2 = plt.subplots()
    ax2.plot(range(num),sum_lamda/domain/var)
    
#    ax1.set_xlim([1,n_eigen])
#    plt.legend()
    ax2.set_xlabel('n')
    ax2.set_ylabel('cumulate Lamda/ (Domain*Var)')  
    plt.title('Finite Sums')
    
    
    
#    print(n)
    cum_lamda=np.zeros((n,1))
    lamda=np.zeros((n,1))
    w_x=np.zeros((n,1))
    w_y=np.zeros((n,1))    
    
    
    
    for kk in range(n):
        lamda[kk]=lamda_all[kk]
        w_x[kk]=w_x_all[kk]
        w_y[kk]=w_y_all[kk]
        cum_lamda[kk]=sum_lamda[kk]
      



  
  
    return lamda,w_x,w_y,n,cum_lamda
  
  

def sort_lamda_3D(lamda_x,w0_x,lamda_y,w0_y,lamda_z,w0_z,domain,var,weight):
#  Sort_Lamda 三维特征值组合并排序
#  输入 Lamda_x，w0_x:x方向特征值以及对应特征方程实根   
#       Lamda_y,w0_y:y方向特征值以及对应特征方程实根
#       Lamda_z,w0_z:z方向特征值以及对应特征方程实根
#       Domain：三维立体范围  var：方差

#  返回 lamda:按递减顺序排列的三维特征值
#       w_x,w_y,w_z:特征值对应特征方程在不同方向上的正实根
#       n：特征值截断个数，权重weight, cum_lamda:特征根累计值
    n_x=len(w0_x)
    n_y=len(w0_y)
    n_z=len(w0_z)
    num=n_x*n_y*n_z
    lamda_3d=np.zeros((num,1))
    flag=0
    lamda_index=list()
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                lamda_3d[flag]=lamda_x[i]*lamda_y[j]*lamda_z[k]/var
    #            print(lamda_2d)
                lam_ind=[lamda_3d[flag],i,j,k]
    #            print(lam_ind)
                
                lamda_index.append(lam_ind)
    #            print(lamda_index)
                flag=flag+1
    
    lamda_index_sorted=sorted(lamda_index, key = lambda x: x[0], reverse=True)
    
    sum_lamda=np.zeros((num,1))
    lamda_all=np.zeros((num,1))
    w_x_all=np.zeros((num,1))
    w_y_all=np.zeros((num,1))
    w_z_all=np.zeros((num,1))
   
    lab=1
    
    for k in range(num):
        lamda_all[k]=lamda_index_sorted[k][0]
        w_x_all[k]=w0_x[lamda_index_sorted[k][1]]
        w_y_all[k]=w0_y[lamda_index_sorted[k][2]]
        w_z_all[k]=w0_z[lamda_index_sorted[k][3]]
        
#        print(w_x_all)
        if k==0:
            sum_lamda[k]=lamda_index_sorted[k][0]
        else:
            sum_lamda[k]=sum_lamda[k-1]+lamda_index_sorted[k][0]
            
        if lab and sum_lamda[k]/domain/var>=weight:
            n=k+1
#            print(n)
            lab=0 
#    print(k)
#    print(num)
            
    fig, ax1 = plt.subplots()
    ax1.plot(range(num),lamda_all/domain/var)
    
#    ax1.set_xlim([1,n_eigen])
#    plt.legend()
    ax1.set_xlabel('n')
    ax1.set_ylabel('Lamda 3D / (Domain*Var)')  
    plt.title('Series of Engenvalues in 3 Demensions')
    
    
    fig, ax2 = plt.subplots()
    ax2.plot(range(num),sum_lamda/domain/var)
    
#    ax1.set_xlim([1,n_eigen])
#    plt.legend()
    ax2.set_xlabel('n')
    ax2.set_ylabel('cumulate Lamda/ (Domain*Var)')  
    plt.title('Finite Sums')
    
    
    
#    print(n)
    cum_lamda=np.zeros((n,1))
    lamda=np.zeros((n,1))
    w_x=np.zeros((n,1))
    w_y=np.zeros((n,1))    
    w_z=np.zeros((n,1)) 
    
    
    for kk in range(n):
        lamda[kk]=lamda_all[kk]
        w_x[kk]=w_x_all[kk]
        w_y[kk]=w_y_all[kk]
        w_z[kk]=w_z_all[kk]
        cum_lamda[kk]=sum_lamda[kk]
      



  
  
    return lamda,w_x,w_y,w_z,n,cum_lamda
  
  
   
  


#############################################################################
####计算特征值对应的特征函数值    
#############################################
def eigen_func(n,w,eta,L,x):
    
#输入：   n:特征值截断个数  w：特征方程正实根  eta:相关长度    L:区域长度   x:位置
#输出：   f:特征值对应的特征函数值
    f=np.empty((n,1))
    for i in range(n):
        f[i]=(eta*w[i]*np.cos(w[i]*x)+np.sin(w[i]*x))/np.sqrt((eta**2*w[i]**2+1)*L/2+eta)
    
    return f
    




