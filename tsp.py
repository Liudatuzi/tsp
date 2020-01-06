# travelling salesman problem
import numpy as np
import math
import sys
n = 10  # ten cities
epsilon = 0.01
beta_0 = 200
mu_k = 0.95
rho = 80

def alpha_ij_v(i, j, v, beta):
    temp = 0
    if j == 0:
        for k in range(n):
            temp += d[k][i] * v[k * n + n - 1] + d[i][k] * v[k * n + j + 1]
    elif j == n - 1:
        for k in range(n):
            temp += d[k][i] * v[k * n + j - 1] + d[i][k] * v[k * n]
    else:
        for k in range(n):
            temp += d[k][i] * v[k * n + j - 1] + d[i][k] * v[k * n + j + 1]

    temp = (temp - rho * v[i * n + j]) / beta
    temp = math.exp(temp)
    return temp


def hij_vrc(i, j, v, r_i, c_j, beta):
    alpha = alpha_ij_v(i, j, v, beta)
    h = 1 / (1 + r_i * c_j * alpha)
    return h


def xi_rc(i, r, c, v, beta):
    temp = 0
    for j in range(n):
        alpha = alpha_ij_v(i, j, v, beta)
        temp += 1 / (1 + r[i] * c[j] * alpha)
    temp = temp - 1
    return temp * r[i]


def yj_rc(j, r, c, v, beta):
    temp = 0
    for i in range(n):
        alpha = alpha_ij_v(i, j, v, beta)
        temp += 1 / (1 + r[i] * c[j] * alpha)
    temp = temp - 1
    return temp * c[j]


def f(r, c, v, beta):
    result = 0
    for i in range(n):
        part1 = 0
        for j in range(n):
            alpha = alpha_ij_v(i, j, v, beta)
            part1 += 1 / (1 + r[i] * c[j] * alpha)
        part1 -= 1
        part1 = part1 ** 2
        result += part1
    for j in range(n):
        part2 = 0
        for i in range(n):
            alpha = alpha_ij_v(i, j, v, beta)
            part2 += 1 / (1 + r[i] * c[j] * alpha)
        part2 -= 1
        part2 = part2 ** 2
        result += part2
    result = result / 2
    return result


# fix v
def iterate(v, r, c, beta):
    x = np.zeros(n)
    y = np.zeros(n)
    while np.sqrt(f(r, c, v, beta)) >= 0.001:
        # initialize x(r,c) y(r,c)
        for i in range(n):
            x[i] = xi_rc(i, r, c, v, beta)
        for j in range(n):
            y[j] = yj_rc(j, r, c, v, beta)
        r = r + mu_k * x
        c = c + mu_k * y
    return r, c


# fix r,c
def calculate_v(v,r, c,beta):
    for i in range(n):
        for j in range(n):
            alpha = alpha_ij_v(i, j, v, beta)
            v[i * n + j] = 1 / (1 + r[i] * c[j] * alpha)
    return v


def e(v):
    result = 0
    for i in range(n):
        for j in range(n):
            for k in range(n - 1):
                result += d[i][j] * v[i * n + k] * v[j * n + k + 1]
            result += d[i][j] * v[i * n + k] * v[j * n + n - 1]
            result -= 0.5 * rho * v[i * n + j] ** 2
    return result


def gradient(v, lamda_rk, lamda_ck):
    dL = np.zeros_like(v)
    for i in range(n):
        for j in range(n):
            if j == 0:
                for k in range(n):
                    dL[i * n + j] = dL[i * n + j] + d[k][i] * v[k * n + n - 1] + d[i][k] * v[k * n + j + 1]
            elif j == n - 1:
                for k in range(n):
                    dL[i * n + j] = dL[i * n + j] + d[k][i] * v[k * n + j - 1] + d[i][k] * v[k * n]
            else:
                for k in range(n):
                    dL[i * n + j] = dL[i * n + j] + d[k][i] * v[k * n + j - 1] + d[i][k] * v[k * n + j + 1]
            dL[i * n + j] -= rho * v[i * n + j]
            dL[i * n + j] = dL[i * n + j] + lamda_rk[i] + lamda_ck[j] + beta * np.log(v[i * n + j] / (1 - v[i * n + j]))
    return dL


def theta_k(v, r, c, beta, h):
    lamda_rk = beta * np.log(r)
    lamda_ck = beta * np.log(c)
    xi = 0.6
    gamma = 0.8
    mk = 0
    while True:
        v1 = v + math.pow(xi, mk) * (h - v)
        L1 = e(v1)
        temp = 0
        for i in range(n):
            for j in range(n):
                temp += v1[i * n + j]
            temp -= 1
            L1 = L1 + lamda_rk[i] * temp
            L1 = L1 + lamda_ck[i] * temp
        L2 = e(v)
        for i in range(n):
            for j in range(n):
                temp += v[i * n + j]
            temp -= 1
            L2 = L2 + lamda_rk[i] * temp
            L2 = L2 + lamda_ck[i] * temp
        temp1 = gamma * (h - v).reshape(1, n * n)
        temp2 = gradient(v, lamda_rk, lamda_ck).reshape(n * n, 1)

        L2 += pow(xi, mk) * np.dot(temp1, temp2)[0, 0]
        if L1 <= L2:
            break
        mk = mk + 1
        # print("mk", mk)
        if mk>2000:
            mk=0

            break


    return math.pow(xi, mk)
def translate_v(v):
    newv=np.zeros_like(v)
    for i in range(n):
        for j in range(n):
            if v[i*n+j]>=0.9:
                newv[i*n+j]=1
            else:
                newv[i*n+j]=0
    return newv
def belong_P(v):
    for i in range(n):
        temp=0
        for j in range(n):
            if v[i*n+j]>1 or v[i*n+j]<0:
                return False
            temp+=v[i*n+j]
        if temp != 1:
            return False
    for j in range(n):
        temp=0
        for i in range(n):
            temp+=v[i*n+j]
        if temp!=1:
            return False
    return True






if __name__ == "__main__":

    # initialize distance between cities
    d = np.random.randint(1, 100, n * n)
    d = d.reshape(n, n)
    d = np.triu(d)
    d += d.T - np.diag(d.diagonal())
    for i in range(n):
        d[i][i] = 0
    v_ = np.random.rand(n * n)
    r_0 = np.random.rand(n)
    c_0 = np.random.rand(n)
    #step 0
    v = v_
    r = r_0
    c = c_0
    v_q=v
    beta = beta_0
    V = np.zeros([10000, n * n])
    R = np.zeros([10000, n])
    C = np.zeros([10000, n])
    r,c=iterate(v,r,c,beta)
    V[0]=calculate_v(v,r,c,beta)
    k = 0
    while True:
        #step 1
        print("k",k)
        v = V[k]
        r, c = iterate(V[k], r, c, beta)
        print("Step2 enter")
        #step 2
        h = np.zeros(n * n)
        for i in range(n):
            for j in range(n):
                h[i * n + j] = hij_vrc(i, j, V[k], r[i], c[j], beta)
        print("beta",beta)

        if np.linalg.norm(h - V[k], ord=2) < epsilon:
            if beta < 1:
                print("process1 terminate")
                break
            else:
                v_q = V[k]
                V[0] = V[k]
                beta = 0.95 * beta
                k = 0
                # print("here")
                # continue
        else:
            theta = theta_k(V[k], r, c, beta, h)
            print("theta",theta)
            V[k + 1] = V[k] + theta * (h - V[k])
            # V[k + 1] = V[k] + 0.3 * (h - V[k])
            k = k + 1
    # step 0
    beta=1
    V[0]=v_q
    k=0
    # step1
    v = translate_v(V[k])
    if (belong_P(v)):
        print("process2 terminate")
        print("d",d)
        v = v.reshape(n, n)
        print("v",v)
        sys.exit()
    rho += 2
    while True:
        # step2
        print("k",k)
        print("Step2")
        r, c = iterate(V[k], r, c, beta)
        #step3
        print("Step3")
        h = np.zeros(n * n)
        for i in range(n):
            for j in range(n):
                h[i * n + j] = hij_vrc(i, j, V[k], r[i], c[j], beta)
        if np.linalg.norm(h - V[k], ord=2) < epsilon:
            V[0]=V[k]
            k=0
            # step1
            print("Step1")
            v = translate_v(V[k])
            if (belong_P(v)):
                print("process2 terminate")
                print("d", d)
                v=v.reshape(n,n)
                print("v", v)
                sys.exit()
            rho += 2
        else:
            theta = theta_k(V[k], r, c, beta, h)
            print("theta", theta)
            V[k + 1] = V[k] + theta * (h - V[k])
            k=k+1
