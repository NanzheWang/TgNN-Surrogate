# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 21:42:16 2020

@author: WNZ
"""




import torch
from torch.nn import Linear,Sequential
from torch.autograd import Variable
import torch.nn as nn
import math
from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import os.path
from KLE import eigen_value_solution,sort_lamda



"""##################function defination###################"""

##########################################################
#定义激活函数swish
class Swish(nn.Module):
	def __init__(self, inplace=True):
		super(Swish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x * torch.sigmoid(x)


##############################################################################
#渗透率计算函数
def k2(kesi,f_x,f_y,mean_logk,lamda):
    logk=mean_logk+torch.sum(torch.sqrt(lamda)*f_x*f_y*kesi,1)
    kk=torch.exp(logk)
    return kk

#############################################################################
####计算特征值对应的特征函数值    
#############################################
def eigen_func2(n,w,eta,L,x):
    f=torch.empty((n,len(x)))
    for i in range(n):
        
        f[i,:]=(eta*w[i]*torch.cos(w[i]*x)+torch.sin(w[i]*x))/torch.sqrt((eta**2*w[i]**2+1)*L/2+eta)
    
    return f



"""##################parameters setting###################"""


##################################
###参数设置
#渗透率场设置
mean_logk=0
var1=1.0

L_x= 1020    #区域长度
L_y= 1020
eta=408 #相关长度

x=np.arange(1,52,1)
x=x*20
y=np.arange(1,52,1)
y=y*20


nx=51     #网格个数
ny=51
nt=50      #时间步数
L_t=10     #时间总长

x=np.arange(1,52,1)
x=x*20
y=np.arange(1,52,1)
y=y*20

t=np.linspace(0.2,10,nt)

Ss=0.0001

weight=0.8


#######
#边界条件
h_boun1_0=202
h_boun2_0=200

n_logk=30  #渗透率场实现个数

num_epoch=2000

N_h=40000

#########################################################################
#设备设置
device = torch.device('cuda:0')

#################################################
###Data Processing
#################################################
bigx=100
L_x=L_x/bigx
L_y= L_y/bigx
eta=eta/bigx
domain=L_x*L_y
x=x/bigx
y=y/bigx
Ss=Ss*bigx*bigx

#################################################
#h的无量纲化
h_boun1=(h_boun1_0-h_boun2_0)/(h_boun1_0-h_boun2_0)
h_boun2=(h_boun2_0-h_boun2_0)/(h_boun1_0-h_boun2_0)

###########################################
###计算所需特征值个数
#########################################
n_eigen=50

lamda_x,w_x0,cumulate_lamda_x=eigen_value_solution(eta,L_x,var1,n_eigen)
lamda_y,w_y0,cumulate_lamda_y=eigen_value_solution(eta,L_y,var1,n_eigen)


############################################################
#二维特征值计算，混合，排序，截断
lamda_xy,w_x,w_y,n,cum_lamda=sort_lamda(lamda_x,w_x0,lamda_y,w_y0,domain,var1,weight)

n_eigen=n




"""##################training data preparation###################"""

#########################################################
#数据导入
#################################################
#修改工作目录
path = "../"

# 查看当前工作目录
retval = os.getcwd()
print ("当前工作目录为 %s" % retval)

# 修改当前工作目录
os.chdir( path )

# 查看修改后的工作目录
retval = os.getcwd()

print ("目录修改成功 %s" % retval)

#################################################################################
#切换工作目录
path = "\\2_data\\"

# 查看当前工作目录
now = os.getcwd()
print ("当前工作目录为 %s" % now)

# 修改当前工作目录
os.chdir( now+path )

# 查看修改后的工作目录
new_path = os.getcwd()
print ("目录修改成功 %s" % new_path)

#2导入数据
#读取二进制文件npy

hh=np.load('hh_train_data_N=%d_weight=%f_seed=100.npy'%(n_logk,weight))
kesi=np.load('kesi_train_data_N=%d_weight=%f_seed=100.npy'%(n_logk,weight))
logk=np.load('logk_train_data_N=%d_weight=%f_seed=100.npy'%(n_logk,weight))

#渗透率场对数转化
k=np.exp(logk)


########################################################################
########################################################################
#定义数据空间
xx=np.empty((n_logk,nt,nx,ny))
yy=np.empty((n_logk,nt,nx,ny))
tt=np.empty((n_logk,nt,nx,ny))
kk=np.empty((n_logk,nt,nx,ny))
kesi_array=np.empty((n_logk,nt,nx,ny,n_eigen))


Y,T,X = np.meshgrid(y,t,x)
for i in range(n_logk):
    xx[i,:,:,:]=X
    yy[i,:,:,:]=Y
    tt[i,:,:,:]=T
    for i_t in range(nt):
        kk[i,i_t,:,:]=k[i,:,:]
        for i_x in range(nx):
            for i_y in range(ny):
                kesi_array[i,i_t,i_x,i_y,:]=kesi[i,:]
                

#########################################################################
#提取训练数据

H_col=hh[0,:,:,:].flatten()[:,None]
T_col=tt[0,:,:,:].flatten()[:,None]
X_col=xx[0,:,:,:].flatten()[:,None]
Y_col=yy[0,:,:,:].flatten()[:,None]
K_col=kk[0,:,:,:].flatten()[:,None]
col_len=nt*nx*ny
kesi_col=np.empty((col_len,n_eigen))

for i_eigen in range(n_eigen):
    kesi_col[:,i_eigen]=kesi_array[0,:,:,:,i_eigen].flatten()[:,None].reshape(col_len,)

TXY_kesi_K = np.hstack((T_col,X_col,Y_col,kesi_col,K_col))
idx = np.random.choice(TXY_kesi_K.shape[0], N_h, replace=False)
TXY_kesi_K_train =TXY_kesi_K[idx,:]
H_train = H_col[idx,:]  

for i in range(1,n_logk):
    H_col=hh[i,:,:,:].flatten()[:,None]
    T_col=tt[i,:,:,:].flatten()[:,None]
    X_col=xx[i,:,:,:].flatten()[:,None]
    Y_col=yy[i,:,:,:].flatten()[:,None]
    K_col=kk[i,:,:,:].flatten()[:,None]

    for i_eigen in range(n_eigen):
        kesi_col[:,i_eigen]=kesi_array[i,:,:,:,i_eigen].flatten()[:,None].reshape(col_len,)

    TXY_kesi_K = np.hstack((T_col,X_col,Y_col,kesi_col,K_col))
    idx = np.random.choice(TXY_kesi_K.shape[0], N_h, replace=False)
    txy_kesi_k_train =TXY_kesi_K[idx,:]
    h_train = H_col[idx,:]
    TXY_kesi_K_train =np.vstack((TXY_kesi_K_train,txy_kesi_k_train))
    H_train = np.vstack((H_train,h_train))

n_train=len(H_train)    
TXY_kesi_train=TXY_kesi_K_train[:,0:23]

#################################################
###Data Processing
#################################################
#h的无量纲化
H_train=(H_train-h_boun2_0)/(h_boun1_0-h_boun2_0)



#打乱数据
TXY_kesi_H_train=np.hstack((TXY_kesi_train,H_train))
np.random.shuffle(TXY_kesi_H_train)

TXY_kesi_train=TXY_kesi_H_train[:,0:23]
H_train=TXY_kesi_H_train[:,23:24]



############################################################################
#提取配点数据
Nf=1000000

#随机取点
lb=np.array([0,x.min(),y.min()])
ub=np.array([t.max(),x.max(),y.max()])
TXY_f_train=lb + (ub-lb)*lhs(3, Nf)
TXY_kesi_f=np.hstack((TXY_f_train,np.zeros((Nf,n_eigen))))
kesi_f=np.random.randn(Nf,n_eigen)   #随机数数组

TXY_kesi_f[:,3:(n_eigen+3)]=kesi_f

np.random.shuffle(TXY_kesi_f)
n_colloc=TXY_kesi_f.shape[0]


############################################################################
#提取边界数据
N_boun1=100000

X_boun_col=x[0]*np.ones((N_boun1,1))
Y_boun_col=y[0]+(y[50]-y[0])*lhs(1, N_boun1)
T_boun_col=0+(t[49]-0)*lhs(1, N_boun1)
kesi_boun_col=np.random.randn(N_boun1,n_eigen)
H_boun_col=h_boun1*np.ones((N_boun1,1))


TXY_kesi_boun = np.hstack((T_boun_col,X_boun_col,Y_boun_col,kesi_boun_col))   
H_boun=H_boun_col


TXY_kesi_boun = torch.from_numpy(TXY_kesi_boun)
TXY_kesi_boun = TXY_kesi_boun.type(torch.FloatTensor)
TXY_kesi_boun = TXY_kesi_boun.to(device)

H_boun = torch.from_numpy(H_boun)
H_boun = H_boun.type(torch.FloatTensor)
H_boun = H_boun.to(device)


#提取边界数据2
N_boun2=100000
X_boun2_col=x[nx-1]*np.ones((N_boun2,1))
Y_boun2_col=y[0]+(y[50]-y[0])*lhs(1, N_boun2)
T_boun2_col=0+(t[49]-0)*lhs(1, N_boun2)
kesi_boun2_col=np.random.randn(N_boun2,n_eigen)
H_boun2_col=h_boun2*np.ones((N_boun2,1))


TXY_kesi_boun2 = np.hstack((T_boun2_col,X_boun2_col,Y_boun2_col,kesi_boun2_col))   
H_boun2=H_boun2_col


TXY_kesi_boun2 = torch.from_numpy(TXY_kesi_boun2)
TXY_kesi_boun2 = TXY_kesi_boun2.type(torch.FloatTensor)
TXY_kesi_boun2 = TXY_kesi_boun2.to(device)

H_boun2 = torch.from_numpy(H_boun2)
H_boun2 = H_boun2.type(torch.FloatTensor)
H_boun2 = H_boun2.to(device)


#提取初始条件数据
N_ic=100000
X_ic_col=x[1]+(x[50]-y[1])*lhs(1, N_ic)
Y_ic_col=y[0]+(y[50]-y[0])*lhs(1, N_ic)
T_ic_col=np.zeros((N_ic,1))
kesi_ic_col=np.random.randn(N_ic,n_eigen)
 
H_ic_col=h_boun2*np.ones((N_ic,1))

TXY_kesi_ic = np.hstack((T_ic_col,X_ic_col,Y_ic_col,kesi_ic_col)) 
  
H_ic=H_ic_col


TXY_ic_train=TXY_kesi_ic
H_ic_train=H_ic


TXY_ic_train = torch.from_numpy(TXY_ic_train)
TXY_ic_train = TXY_ic_train.type(torch.FloatTensor)
TXY_ic_train = TXY_ic_train.to(device)

H_ic_train = torch.from_numpy(H_ic_train)
H_ic_train = H_ic_train.type(torch.FloatTensor)
H_ic_train = H_ic_train.to(device)


#提取初始条件数据2
N_ic2=100000
X_ic2_col=x[0]*np.ones((N_ic2,1))
Y_ic2_col=y[0]+(y[50]-y[0])*lhs(1, N_ic2)
T_ic2_col=np.zeros((N_ic2,1))
kesi_ic2_col=np.random.randn(N_ic2,n_eigen)

H_ic2_col=h_boun1*np.ones((N_ic2,1))


TXY_kesi_ic2 = np.hstack((T_ic2_col,X_ic2_col,Y_ic2_col,kesi_ic2_col)) 
H_ic2=H_ic2_col


TXY_ic2_train=TXY_kesi_ic2
H_ic2_train=H_ic2


TXY_ic2_train = torch.from_numpy(TXY_ic2_train)
TXY_ic2_train = TXY_ic2_train.type(torch.FloatTensor)
TXY_ic2_train = TXY_ic2_train.to(device)

H_ic2_train = torch.from_numpy(H_ic2_train)
H_ic2_train = H_ic2_train.type(torch.FloatTensor)
H_ic2_train = H_ic2_train.to(device)


############################################################################
#提取无流边界数据
N_noflow1=10000
X_noflow1_col=x[0]+(x[50]-x[0])*lhs(1, N_noflow1)
Y_noflow1_col=y[0]*np.ones((N_noflow1,1))
T_noflow1_col=0+(t[49]-0)*lhs(1, N_noflow1)
TXY_noflow1 = np.hstack((T_noflow1_col,X_noflow1_col,Y_noflow1_col))   


#提取无流边界数据2
N_noflow2=10000
X_noflow2_col=x[0]+(x[50]-x[0])*lhs(1, N_noflow2)
Y_noflow2_col=y[50]*np.ones((N_noflow2,1))
T_noflow2_col=0+(t[49]-0)*lhs(1, N_noflow2)
TXY_noflow2 = np.hstack((T_noflow2_col,X_noflow2_col,Y_noflow2_col))   
kesi_noflow_col=np.random.randn(N_noflow1+N_noflow2,n_eigen)
TXY_noflow=np.vstack((TXY_noflow1,TXY_noflow2))
TXY_kesi_noflow=np.hstack((TXY_noflow,kesi_noflow_col))
TXY_kesi_noflow = torch.from_numpy(TXY_kesi_noflow)
TXY_kesi_noflow = TXY_kesi_noflow.type(torch.FloatTensor)
TXY_kesi_noflow = TXY_kesi_noflow.to(device)

TXY_kesi_noflow= Variable(TXY_kesi_noflow, requires_grad=True)

TXY_kesi_f = torch.from_numpy(TXY_kesi_f)
TXY_kesi_f = TXY_kesi_f.type(torch.FloatTensor)

TXY_kesi_train = torch.from_numpy(TXY_kesi_train)
TXY_kesi_train=TXY_kesi_train.type(torch.FloatTensor)

H_train = torch.from_numpy(H_train)
H_train=H_train.type(torch.FloatTensor)


w_x_tf = torch.from_numpy(w_x)
w_x_tf = w_x_tf.type(torch.FloatTensor)

w_y_tf = torch.from_numpy(w_y)
w_y_tf = w_y_tf.type(torch.FloatTensor)

lamda_xy_tf = torch.from_numpy(lamda_xy)
lamda_xy_tf = lamda_xy_tf.type(torch.FloatTensor)

x_tf=TXY_kesi_f[:,1]
y_tf=TXY_kesi_f[:,2]

x_tf= Variable(x_tf, requires_grad=True)
y_tf= Variable(y_tf, requires_grad=True)

fx_train=eigen_func2(n_eigen,w_x_tf,eta,L_x,x_tf)
fy_train=eigen_func2(n_eigen,w_y_tf,eta,L_y,y_tf)

k_train=k2(TXY_kesi_f[:,3:23],fx_train.transpose(0,1),fy_train.transpose(0,1),mean_logk,lamda_xy_tf.transpose(0,1))


k_x = torch.autograd.grad(outputs=k_train.sum(), inputs=x_tf, \
                          create_graph=True,allow_unused=True)[0].view(n_colloc, 1).detach()

k_y = torch.autograd.grad(outputs=k_train.sum(), inputs=y_tf, \
                          create_graph=True,allow_unused=True)[0].view(n_colloc, 1).detach()


k_train=k_train.detach().numpy().reshape(n_colloc,1)

TXY_kesi_K_kxky_f_train =np.hstack((TXY_kesi_f,k_train,k_x,k_y))
TXY_kesi_K_kxky_f_train = torch.from_numpy(TXY_kesi_K_kxky_f_train)
TXY_kesi_K_kxky_f_train = TXY_kesi_K_kxky_f_train.type(torch.FloatTensor)

##############################################################################
#训练数据分批处理

BATCH_SIZE = 50000      # 批训练的数据个数
N_batch=math.ceil(n_colloc/BATCH_SIZE )
TXY_kesi_K_kxky_f_train_set=[]
for i_batch in range(int(N_batch)):
    TXY_kesi_K_kxky_f_train_set.append(TXY_kesi_K_kxky_f_train[BATCH_SIZE*i_batch:BATCH_SIZE*(i_batch+1),:])

    
BATCH_SIZE2=math.ceil(n_train/N_batch )
TXY_kesi_train_set=[]
H_train_set=[]
for i_batch in range(int(N_batch)):
    TXY_kesi_train_set.append(TXY_kesi_train[BATCH_SIZE2*i_batch:BATCH_SIZE2*(i_batch+1),:])
    H_train_set.append(H_train[BATCH_SIZE2*i_batch:BATCH_SIZE2*(i_batch+1),:])


"""##################defination of neural network###################"""

#############################################################
#定义神经网络
Net0=Sequential(
    Linear(23,50),
    Swish(),
    Linear(50,50),
    Swish(),
    Linear(50,50),
    Swish(),
    Linear(50, 50),
    Swish(),
    Linear(50, 50),
    Swish(),
    Linear(50, 50),
    Swish(),
    Linear(50, 50),
    Swish(),
    Linear(50, 1),
)

class Net_hardway(nn.Module):
    def __init__(self,**kwargs):
        super(Net_hardway, self).__init__(**kwargs)
        self.net=Net0
    def forward(self, x):
        out=self.net(x)
        h=(x[:,1:2]-0.2)*h_boun2/10+(10.2-x[:,1:2])*h_boun1/10+(x[:,1:2]-0.2)*(x[:,1:2]-10.2)*out
        return h

Net= Net_hardway()

# 定义神经网络优化器
LR=0.001
optimizer=torch.optim.Adam([
    {'params': Net.parameters()},
    
],lr=LR)
    
lr_list=[]

################################################
##使用GPU
Net=Net.to(device)
  
    
"""##################training of network###################"""

#定义loss数组
loss_set=[]
f1_set=[]
f2_set=[]  
f3_set=[]  
f4_set=[]  
f5_set=[] 


tol=0.00035
lamd1=1
lamd2=1
lamd3=1
lamd4=1
lamd5=1
epoch_loss=[]

start_time = time.time()  
##########################################################################
##分批训练
for epoch in range(num_epoch):
    for ite in range(N_batch):
        
        batch_x=TXY_kesi_train_set[ite]
        batch_y=H_train_set[ite]
        
        batch_x=batch_x.to(device)
        batch_y=batch_y.to(device)
        
        batch_xkf=TXY_kesi_K_kxky_f_train_set[ite]
        batch_xkf=Variable(batch_xkf,requires_grad=True)
        batch_xkf=batch_xkf.to(device)
        
        K=batch_xkf[:,-3:-2]
        batch_xf=batch_xkf[:,0:23]
        k_x_=batch_xkf[:,-2:-1]
        k_y_=batch_xkf[:,-1:]
        n_batch=len(K)
 

        optimizer.zero_grad()
        prediction=Net(batch_x)
        noflow_pred=Net(TXY_kesi_noflow)
        ic_pred=Net(TXY_ic_train)
        ic2_pred=Net(TXY_ic2_train)
        prediction_f=Net(batch_xf)
        
        H_grad = torch.autograd.grad(outputs=prediction_f.sum(), inputs=batch_xf, create_graph=True)[0]
        
        H_noflow_grad = torch.autograd.grad(outputs=noflow_pred.sum(), inputs=TXY_kesi_noflow, create_graph=True)[0]
        Hy_noflow=H_noflow_grad[:,2:3].contiguous()
        
        Ht = H_grad[:, 0].contiguous().view(n_batch, 1)
        Hx = H_grad[:, 1].contiguous().view(n_batch, 1)
        Hy = H_grad[:, 2].contiguous().view(n_batch, 1)
        Hxx=torch.autograd.grad(outputs=Hx.sum(), inputs=batch_xf,\
                                create_graph=True)[0][:,1].contiguous().view(n_batch, 1)
        Hyy=torch.autograd.grad(outputs=Hy.sum(), inputs=batch_xf,\
                                create_graph=True)[0][:,2].contiguous().view(n_batch, 1)
        f1=torch.pow((Ss*Ht-K*Hxx-K*Hyy-k_x_*Hx-k_y_*Hy)*lamd1,2).mean()
        f2=torch.pow((ic_pred-H_ic_train)*lamd2,2).mean()+torch.pow((ic2_pred-H_ic2_train)*lamd2,2).mean()
        f3=torch.pow(Hy_noflow*lamd4,2).mean()
        f4= torch.pow((prediction-batch_y)*lamd5,2).mean()

        loss=f1+f2+f3+f4
        loss.backward()
        optimizer.step()
        
        loss=loss.data
        f1=f1.data
        f2=f2.data
        f3=f3.data
        f4=f4.data
        loss_set.append(loss)
        f1_set.append(f1)
        f2_set.append(f2)
        f3_set.append(f3)
        f4_set.append(f4)
        print('Epoch: ', epoch, '| Step: ', ite, '|loss: ',loss)
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        epoch_loss.append(loss)
    
    if max(epoch_loss)<=tol:
        break
    else:
        epoch_loss=[]


end_time = time.time() 

training_time = end_time  - start_time                
print('Training time: %.4f' % (training_time))

plt.figure()     
plt.plot(range(len(loss_set)),loss_set)
plt.xlabel('Iteration')
plt.ylabel('loss')


plt.figure()   
plt.plot(range(len(f1_set)),f1_set)
plt.xlabel('Iteration')
plt.ylabel('f1_loss')

plt.figure()     
plt.plot(range(len(f2_set)),f2_set)      
plt.xlabel('Iteration')
plt.ylabel('f2_loss')    

plt.figure()     
plt.plot(range(len(f3_set)),f3_set)      
plt.xlabel('Iteration')
plt.ylabel('f3_loss') 


plt.figure()     
plt.plot(range(len(f4_set)),f4_set)      
plt.xlabel('Iteration')
plt.ylabel('f4_loss') 

plt.figure()     
plt.plot(range(len(f5_set)),f5_set)      
plt.xlabel('Iteration')
plt.ylabel('f5_loss') 


plt.figure()  
plt.plot(range(len(lr_list)),lr_list,color = 'r')  


################################################################################
#修改工作目录
path = "../"

# 查看当前工作目录
retval = os.getcwd()
print ("当前工作目录为 %s" % retval)

# 修改当前工作目录
os.chdir( path )

# 查看修改后的工作目录
retval = os.getcwd()

print ("目录修改成功 %s" % retval)
#切换工作目录
path = "\\3_network_parameters_results\\"
# 查看当前工作目录
now = os.getcwd()
print ("当前工作目录为 %s" % now)

# 修改当前工作目录
os.chdir( now+path )

# 查看修改后的工作目录
new_path = os.getcwd()
print ("目录修改成功 %s" % new_path)


# Save and load only the model parameters
torch.save(Net.state_dict(), '2D_epoch=%d_N_logk=%d_Nh=%d_n_train=%d_\
N_colloc=%d_batchsize=%d_LR=%f_tol=%f_lamd1=%d_lamd4=%d_l5=%d_T=%.3lf.ckpt'%(num_epoch,\
n_logk,N_h,n_train,n_colloc,BATCH_SIZE,LR,tol,lamd1,lamd4,lamd5,training_time))


"""##################network testing###################"""

##################### Testing ######################
    
n_logk_test=200 #渗透率场实现个数
#修改工作目录
path = "../"

# 查看当前工作目录
retval = os.getcwd()
print ("当前工作目录为 %s" % retval)

# 修改当前工作目录
os.chdir( path )

# 查看修改后的工作目录
retval = os.getcwd()
print ("目录修改成功 %s" % retval)
###############################################
#切换工作目录
path = "\\2_data\\"

# 查看当前工作目录
now = os.getcwd()
print ("当前工作目录为 %s" % now)

# 修改当前工作目录
os.chdir( now+path )

# 查看修改后的工作目录
new_path = os.getcwd()
print ("目录修改成功 %s" % new_path)


#2导入数据
#读取二进制文件npy

hh_test=np.load('hh_test_data_N=%d_weight=%f_seed=200.npy'%(n_logk_test,weight))
kesi_test=np.load('kesi_test_data_N=%d_weight=%f_seed=200.npy'%(n_logk_test,weight))
logk_test=np.load('logk_test_data_N=%d_weight=%f_seed=200.npy'%(n_logk_test,weight))

#########################################################################
#########################################################################
#提取测试数据

T_test_col=T.flatten()[:,None]
X_test_col=X.flatten()[:,None]
Y_test_col=Y.flatten()[:,None]
h_test_array_pred_all =np.zeros((n_logk_test,nt,nx,ny))


error_l2_set=np.empty((n_logk_test))
R2_set=np.empty((n_logk_test))


for i in range(n_logk_test):
    kesi_test_col=np.ones((nt*nx*ny,1))*kesi_test[i,:]
    TXY_kesi_test=np.hstack((T_test_col,X_test_col,Y_test_col,kesi_test_col))
    TXY_kesi_test = torch.from_numpy(TXY_kesi_test)
    TXY_kesi_test = TXY_kesi_test.type(torch.FloatTensor)
    TXY_kesi_test = TXY_kesi_test.to(device) 

    h_pred_test = Net(TXY_kesi_test)
    h_pred_test=h_pred_test.cpu().detach().numpy()

    h_pred_test=h_pred_test*(h_boun1_0-h_boun2_0)+h_boun2_0
    TXY_kesi_test=TXY_kesi_test.cpu()
    
    h_test_array_pred= h_pred_test.reshape(50,51,51)



    h_test_array_pred_all[i,:,:,:]=h_test_array_pred
#    print('Predicting realization %d'%(i+1))

    ##############################################
    #calculate error    
    error_l2 = np.linalg.norm(hh_test[i].flatten()-h_test_array_pred.flatten(),2)/np.linalg.norm(hh_test[i].flatten(),2)
#    print('Error L2: %e' % (error_l2))
    error_l2_set[i]=error_l2
    
    R2=1-np.sum((hh_test[i].flatten()-h_test_array_pred.flatten())**2)/np.sum((hh_test[i].flatten()-hh_test[i].flatten().mean())**2)
#    print('coefficient of determination  R2: %e' % (R2))
    R2_set[i]=R2



L2_mean=np.mean(error_l2_set)
L2_var=np.var(error_l2_set)
R2_mean=np.mean(R2_set)
R2_var=np.var(R2_set)

print('L2 mean:')
print(L2_mean)
print('L2 var:')
print(L2_var)

print('R2 mean:')
print(R2_mean)
print('R2 var:')
print(R2_var)




"""##################results visualization###################"""


x=x*bigx
y=y*bigx


############# Plotting #################
#结果展示    

#选择观测时空点
obs_t1=24
obs_x1=9
obs_y1=9

obs_t2=39
obs_x2=9
obs_y2=39

real_h1=hh_test[:,obs_t1,obs_x1,obs_y1].flatten()
pred_h1=h_test_array_pred_all[:,obs_t1,obs_x1,obs_y1].flatten()

real_h2=hh_test[:,obs_t2,obs_x2,obs_y2].flatten()
pred_h2=h_test_array_pred_all[:,obs_t2,obs_x2,obs_y2].flatten()


col_x_ticks = np.arange(199.7,202.3, 0.4)
plt.figure(figsize=(5,5))
plt.scatter(real_h1,pred_h1,marker='o',c='',edgecolors='b',label='Point1')
plt.scatter(real_h2,pred_h2,marker='o',c='',edgecolors='r',label='Point2')
plt.xlabel('Prediction')
plt.ylabel('Exact')
plt.title("Correlation between reference and prediction")
plt.xlim(199.8,202)
plt.ylim(199.8,202)
plt.xticks(col_x_ticks)
plt.yticks(col_x_ticks)


############################# Plotting_2 #############################
#统计结果展示    
TgNN_R2_set=R2_set
TgNN_error_l2_set=error_l2_set
num_bins = 15
l2_x_ticks = np.arange(0,0.0016, 0.0003)
plt.figure(figsize=(6,4))
plt.hist(TgNN_error_l2_set, num_bins)
plt.title(r'$Histogram\ \ of\ \  relative\ \ L_2\ \ error$')
plt.xticks(l2_x_ticks)


num_bins2 = 15
plt.figure(figsize=(6,4))
plt.hist(TgNN_R2_set, num_bins2)
plt.title(r'$Histogram\ \ of\ \  R^2\ \ score$')
plt.xlim(0.9,1)









