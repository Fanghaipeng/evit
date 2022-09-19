import numpy as np
import matplotlib.pyplot as plt
#生成按照一定规律产生的数据
x=np.arange(30)
random_y=np.random.uniform(-5,5,30)
y=2*x*x+random_y+1

#绘制误差函数三维可视图
s1=np.arange(-5,5,0.1)
s2=np.arange(-5,5,0.1)
S1,S2=np.meshgrid(s1,s2)
Z0=0
for i in x:
    Zs=(S1+x[i]*S2-y[i])**2*0.5+Z0
    Z0=Zs
plt.figure(1)
ax3 = plt.axes(projection='3d')
ax3.plot_surface(S1,S2,Zs,cmap='rainbow')
