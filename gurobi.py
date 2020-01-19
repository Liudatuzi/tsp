
import gurobipy as gp
from gurobipy import GRB
import numpy as np
d=np.loadtxt("./tsp_dist_matrix.txt",encoding="gbk")
d=d.astype(int)
index,columns=d.shape#实际上行列是相等的，都是n


try:

    # Create a new model
    m = gp.Model("tsp")
    v=m.addVars(index,columns,lb=0,ub=1,vtype=GRB.INTEGER,name="v")#matrix v_ij
    m.setObjective(gp.quicksum(d[i,j]*v[i,k]*v[j,k+1] for i in range(index) for j in range(index) for k in range(index-1))+gp.quicksum(d[i,j]*v[i,index-1]*v[j,0] for i in range(index) for j in range(index)),GRB.MINIMIZE)
    for i in range(index):
        m.addConstr(gp.quicksum(v[i,j] for j in range(index))==1)
    for j in range(index):
        m.addConstr(gp.quicksum(v[i,j] for i in range(index))==1)
    m.optimize()

    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % m.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
