import cvxpy as cp
import numpy as np
from scipy.special import comb

def solve_sdp(I, m, c, r,a, b, y, s, lambda_val):
    comb_dict = {}
    max_deg = I + 1
    for n in range(max_deg + 1):
        for k in range(max_deg + 1):
            if k <= n:
                comb_dict[(n, k)] = comb(n, k, exact=True)
    
    alpha = cp.Variable(I + 1)
    x = cp.Variable((I + 1, I + 1), PSD=True)
    q = cp.Variable((I + 1, I + 1), PSD=True)
    
    constraints = []
    if lambda_val == 0:
        beta = cp.Variable(I + 1)
        eta = cp.Variable(I + 1)
        # beta 约束
        constraints.append(beta[0] == alpha[0] - c * y)
        constraints.append(beta[1] == alpha[1] + r)
        for i in range(2, I + 1):
            constraints.append(beta[i] == alpha[i])
        # eta 约束
        constraints.append(eta[0] == alpha[0] + r * y - c * y)
        for i in range(1, I + 1):
            constraints.append(eta[i] == alpha[i])

        # x 约束 (向量化)
        for l in range(1, I + 1):
            indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l - 1]
            if indices:
                constraints.append(cp.sum([x[i, j] for (i, j) in indices]) == 0)

        # 预计算 y_powers 用于多项式
        y_powers = np.array([y**r for r in range(I + 1)])
        a_powers
        for l in range(I + 1):
            sum_val = 0
            for m_val in range(l + 1):
                for r_val in range(m_val, min(I + 1, I + m_val - l + 1)):
                    comb1 = comb_dict.get((r_val, m_val), 0)
                    comb2 = comb_dict.get((I - r_val, l - m_val), 0)
                    power_term = y_powers[r_val - m_val] * a**m_val
                    sum_val += beta[r_val] * comb1 * comb2 * power_term
        # q 约束 (向量化)
        for l in range(1, I + 1):
            indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l - 1]
            if indices:
                constraints.append(cp.sum([q[i, j] for (i, j) in indices]) == 0)

        # 预计算 b_powers 和 y_powers
        b_powers = np.array([b**m_val for m_val in range(I + 1)])
        for l in range(I + 1):
            sum_val = 0
            for m_val in range(l + 1):
                for r_val in range(m_val, min(I + 1, I + m_val - l + 1)):
                    comb1 = comb_dict.get((r_val, m_val), 0)
                    comb2 = comb_dict.get((I - r_val, l - m_val), 0)
                    power_term = y_powers[r_val - m_val] * b_powers[m_val]
                    sum_val += eta[r_val] * comb1 * comb2 * power_term
            indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l]
            constraints.append(sum_val == cp.sum([q[i, j] for (i, j) in indices]))

    else:
        beta = cp.Variable(I + 1)
        gamma = cp.Variable(I + 1)
        eta = cp.Variable(I + 1)
        z = cp.Variable((I + 1, I + 1), PSD=True)
        if s >= y:
            # beta 约束
            constraints.append(beta[0] == alpha[0] + lambda_val * c * s - c * y)
            constraints.append(beta[1] == alpha[1] + r - lambda_val * r)
            for i in range(2, I + 1):
                constraints.append(beta[i] == alpha[i])
            # gamma 约束
            constraints.append(gamma[0] == alpha[0] + r * y - c * y + lambda_val * c * s)
            constraints.append(gamma[1] == alpha[1] - lambda_val * r)
            for i in range(2, I + 1):
                constraints.append(gamma[i] == alpha[i])
            # eta 约束
            constraints.append(eta[0] == alpha[0] + r * y - c * y + lambda_val * c * s - lambda_val * r * s)
            for i in range(1, I + 1):
                constraints.append(eta[i] == alpha[i])
            # 预计算幂次
            a_powers = np.array([a**r for r in range(I + 1)])
            y_powers = np.array([y**r for r in range(I + 1)])
            s_powers = np.array([s**r for r in range(I + 1)])
            b_powers = np.array([b**m_val for m_val in range(I + 1)])
            # x 约束
            for l in range(1, I + 1):
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l - 1]
                if indices:
                    constraints.append(cp.sum([x[i, j] for (i, j) in indices]) == 0)
            for l in range(I + 1):
                sum_val = 0
                for m_val in range(l + 1):
                    for r_val in range(m_val, min(I + 1, I + m_val - l + 1)):
                        comb1 = comb_dict.get((r_val, m_val), 0)
                        comb2 = comb_dict.get((I - r_val, l - m_val), 0)
                        power_term = a_powers[r_val - m_val] * y_powers[m_val]
                        sum_val += beta[r_val] * comb1 * comb2 * power_term
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l]
                constraints.append(sum_val == cp.sum([x[i, j] for (i, j) in indices]))
            # z 约束
            for l in range(1, I + 1):
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l - 1]
                if indices:
                    constraints.append(cp.sum([z[i, j] for (i, j) in indices]) == 0)
            for l in range(I + 1):
                sum_val = 0
                for m_val in range(l + 1):
                    for r_val in range(m_val, min(I + 1, I + m_val - l + 1)):
                        comb1 = comb_dict.get((r_val, m_val), 0)
                        comb2 = comb_dict.get((I - r_val, l - m_val), 0)
                        power_term = y_powers[r_val - m_val] * s_powers[m_val]
                        sum_val += gamma[r_val] * comb1 * comb2 * power_term
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l]
                constraints.append(sum_val == cp.sum([z[i, j] for (i, j) in indices]))
            # q 约束
            for l in range(1, I + 1):
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l - 1]
                if indices:
                    constraints.append(cp.sum([q[i, j] for (i, j) in indices]) == 0)
            for l in range(I + 1):
                sum_val = 0
                for m_val in range(l + 1):
                    for r_val in range(m_val, min(I + 1, I + m_val - l + 1)):
                        comb1 = comb_dict.get((r_val, m_val), 0)
                        comb2 = comb_dict.get((I - r_val, l - m_val), 0)
                        power_term = s_powers[r_val - m_val] * b_powers[m_val]
                        sum_val += eta[r_val] * comb1 * comb2 * power_term
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l]
                constraints.append(sum_val == cp.sum([q[i, j] for (i, j) in indices]))
        else:
            # beta 约束
            constraints.append(beta[0] == alpha[0] + lambda_val * c * s - c * y)
            constraints.append(beta[1] == alpha[1] + r - lambda_val * r)
            for i in range(2, I + 1):
                constraints.append(beta[i] == alpha[i])

            # gamma 约束
            constraints.append(gamma[0] == alpha[0] - lambda_val * r * s - c * y + lambda_val * c * s)
            constraints.append(gamma[1] == alpha[1] + r)
            for i in range(2, I + 1):
                constraints.append(gamma[i] == alpha[i])
            # eta 约束
            constraints.append(eta[0] == alpha[0] + r * y - c * y + lambda_val * c * s - lambda_val * r * s)
            for i in range(1, I + 1):
                constraints.append(eta[i] == alpha[i])

            # 预计算幂次
            a_powers = np.array([a**r for r in range(I + 1)])
            s_powers = np.array([s**r for r in range(I + 1)])
            y_powers = np.array([y**r for r in range(I + 1)])
            b_powers = np.array([b**m_val for m_val in range(I + 1)])
            # x 约束
            for l in range(1, I + 1):
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l - 1]
                if indices:
                    constraints.append(cp.sum([x[i, j] for (i, j) in indices]) == 0)
            for l in range(I + 1):
                sum_val = 0
                for m_val in range(l + 1):
                    for r_val in range(m_val, min(I + 1, I + m_val - l + 1)):
                        comb1 = comb_dict.get((r_val, m_val), 0)
                        comb2 = comb_dict.get((I - r_val, l - m_val), 0)
                        power_term = a_powers[r_val - m_val] * s_powers[m_val]
                        sum_val += beta[r_val] * comb1 * comb2 * power_term
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l]
                constraints.append(sum_val == cp.sum([x[i, j] for (i, j) in indices]))
            # z 约束
            for l in range(1, I + 1):
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l - 1]
                if indices:
                    constraints.append(cp.sum([z[i, j] for (i, j) in indices]) == 0)
            for l in range(I + 1):
                sum_val = 0
                for m_val in range(l + 1):
                    for r_val in range(m_val, min(I + 1, I + m_val - l + 1)):
                        comb1 = comb_dict.get((r_val, m_val), 0)
                        comb2 = comb_dict.get((I - r_val, l - m_val), 0)
                        power_term = s_powers[r_val - m_val] * y_powers[m_val]
                        sum_val += gamma[r_val] * comb1 * comb2 * power_term
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l]
                constraints.append(sum_val == cp.sum([z[i, j] for (i, j) in indices]))
            # q 约束
            for l in range(1, I + 1):
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l - 1]
                if indices:
                    constraints.append(cp.sum([q[i, j] for (i, j) in indices]) == 0)
            for l in range(I + 1):
                sum_val = 0
                for m_val in range(l + 1):
                    for r_val in range(m_val, min(I + 1, I + m_val - l + 1)):
                        comb1 = comb_dict.get((r_val, m_val), 0)
                        comb2 = comb_dict.get((I - r_val, l - m_val), 0)
                        power_term = y_powers[r_val - m_val] * b_powers[m_val]
                        sum_val += eta[r_val] * comb1 * comb2 * power_term
                indices = [(i, j) for i in range(I + 1) for j in range(I + 1) if i + j == 2 * l]
                constraints.append(sum_val == cp.sum([q[i, j] for (i, j) in indices]))

    # 目标函数
    objective = cp.Minimize(cp.sum(cp.multiply(alpha, m)))
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False, max_iters=10000, eps=1e-4)
    
    return problem.value