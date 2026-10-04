import numpy as np

"""
f(x, y) = x**4 + y**4 + x**2 - 3*y**2 + x*y
"""
def grad(u):
    x, y = u
    return np.array(
        [4*x**3 + 2*x + y,
         4*y**3 - 6*y + x]
    )

def hessian(u): 
    x, y = u
    return np.array(
        [[12*x**2 + 2, 1],  
         [1, 12*y**2 - 6]]
    )

def newton_raphson(u0, grad_f, hessian_f, epsilon = 1e-8, max_iter = 1000):
    u = u0
    for _ in range(max_iter):
        g = grad_f(u)
        H = hessian_f(u)
        # cách 1 nhân ma trận
        # t = g @ np.linalg.inv(H)
        # cách 2 giải phương trình
        t = np.linalg.solve(H, g)
        u = u - t
        if np.linalg.norm(g) < epsilon:
            break
    return u

if __name__ == "__main__":
    u0 = np.array([-0.4, -2])
    solution = newton_raphson(u0, grad, hessian)
    print("Nghiệm tìm được:", solution)