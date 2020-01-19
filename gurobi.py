import gurobipy as gp
from gurobipy import GRB
import numpy as np
d=np.loadtxt("./city_30.txt",encoding='unicode_escape')
d=d.astype(int)
index,columns=d.shape#实际上行列是相等的，都是n


# try:

# Create a new model
m = gp.Model("tsp")
v=m.addVars(index,columns,lb=0,ub=1,vtype=GRB.INTEGER,name="v")#matrix v_ij
u=m.addVars(index,vtype=GRB.INTEGER,name="u")
m.setObjective(gp.quicksum(d[i,j]*v[i,j]
                           for i in range(index) for j in range(index)),GRB.MINIMIZE)
for i in range(index):
    m.addConstr(gp.quicksum(v[i,j] for j in range(index) if i !=j)==1)
for j in range(index):
    m.addConstr(gp.quicksum(v[i,j] for i in range(index) if i !=j)==1)
for i in range(1,index):
    for j in range(1,index):
        if i !=j:
            m.addConstr(u[i] - u[j] + index * v[i, j] <= index - 1)

m.optimize()

# for v in m.getVars():
#     print('%s %g' % (v.varName, v.x))

print('Obj: %g' % m.objVal)
print(m.status==GRB.OPTIMAL)
# except gp.GurobiError as e:
#     print('Error code ' + str(e.errno) + ': ' + str(e))
#
# except AttributeError:
#     print('Encountered an attribute error')
