
import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares,brentq
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import gammainc, gamma
#均匀分布下的最优订货量以及最优收益
def opt_uniform(b,r,c):
    y_opt=b*(r-c)/r
    profit_opt=b*(r-c)**2/(2*r)
    return y_opt,profit_opt
def uniform_expected_profit(y, b, r, c):
    if y < 0:
        return 0
    elif y <= b:
        return (r - c) * y - (r * y**2) / (2 * b)
    else:
        return r * b / 2 - c * y

#正态分布下的最优订货量以及最优收益
def trunc_normal_equations(params, mu_t, m2, b):
    mu, sigma = params
    alpha = -mu / sigma
    beta = (b - mu) / sigma
    
    phi_alpha = norm.pdf(alpha)
    phi_beta = norm.pdf(beta)
    Phi_alpha = norm.cdf(alpha)
    Phi_beta = norm.cdf(beta)
    
    Z = Phi_beta - Phi_alpha
    lambda_val = (phi_alpha - phi_beta) / Z
    delta_val = (alpha * phi_alpha - beta * phi_beta) / Z
    
    eq1 = mu + sigma * lambda_val - mu_t
    eq2 = mu**2 + sigma**2 + 2*mu*sigma*lambda_val + sigma**2*delta_val - m2
    
    return [eq1, eq2]

def opt_norm(mu_t,m_2,b,r,c):
    #计算截断之前的原始分布的均值和方差
    sigma_t2=m_2-mu_t**2
    initial_guess = [mu_t, np.sqrt(sigma_t2)]
    sol = least_squares(
    lambda p: trunc_normal_equations(p, mu_t, m_2, b),
    initial_guess,
    bounds=([0, 0.01], [b, np.inf]))
    mu_opt,sigma_opt=sol.x

    #计算最优订货量
    crit_frac = (r - c) / r
    alpha = -mu_opt / sigma_opt
    beta = (b - mu_opt) / sigma_opt
    target_cdf = norm.cdf(alpha) + crit_frac * (norm.cdf(beta) - norm.cdf(alpha))
    y_star = mu_opt + sigma_opt * norm.ppf(target_cdf)
    y_opt = np.clip(y_star, 0, b)  

    #计算最优收益
    def truncated_normal_cdf(x, mu, sigma, alpha, beta):
        z = (x - mu) / sigma
        return (norm.cdf(z) - norm.cdf(alpha)) / (norm.cdf(beta) - norm.cdf(alpha))

    integral, _ = quad(
        lambda x: truncated_normal_cdf(x, mu_opt, sigma_opt, alpha, beta),
        0, y_opt)
    optimal_profit = (r - c) * y_star - r * integral
    return y_opt,optimal_profit

def solve_truncated_normal(mu_t, m_2, b):
    sigma_t2 = m_2 - mu_t**2
    initial_guess = [mu_t, np.sqrt(sigma_t2)]
    
    # 使用最小二乘法求解方程
    sol = least_squares(
        lambda p: trunc_normal_equations(p, mu_t, m_2, b),
        initial_guess,
        bounds=([0, 0.01], [b, np.inf])
    )
    
    return sol.x
def expected_profit_norm(y, mu_t, m_2, b, r, c):
    # 求解原始正态分布参数
    mu, sigma = solve_truncated_normal(mu_t, m_2, b)
    
    # 计算截断参数
    alpha = -mu / sigma
    beta = (b - mu) / sigma
    Z = norm.cdf(beta) - norm.cdf(alpha)
    
    # 定义截断正态分布的CDF
    def truncated_cdf(x):
        if x <= 0:
            return 0
        elif x >= b:
            return 1
        z = (x - mu) / sigma
        return (norm.cdf(z) - norm.cdf(alpha)) / Z
    if y < 0:
        return 0
    elif y <= b:
        # 计算 ∫₀ʸ F(x) dx
        integral, _ = quad(truncated_cdf, 0, y)
        # 期望收益公式: (r-c)y - r∫₀ʸF(x)dx
        return (r - c) * y - r * integral
    else:
        # 当 y > b 时
        # 计算 ∫₀ᵇ F(x) dx
        integral_b, _ = quad(truncated_cdf, 0, b)
        # 期望收益公式: r·E[D] - c·y
        # 其中 E[D] = μ_t (截断后的均值)
        return r * mu_t - c * y

def trunc_normal_equations(params, mu_t, m2, b):
    mu, sigma = params
    alpha = -mu / sigma
    beta = (b - mu) / sigma
    phi_alpha, phi_beta = norm.pdf(alpha), norm.pdf(beta)
    Phi_alpha, Phi_beta = norm.cdf(alpha), norm.cdf(beta)
    Z = Phi_beta - Phi_alpha
    λ = (phi_alpha - phi_beta) / Z
    δ = (alpha * phi_alpha - beta * phi_beta) / Z
    eq1 = mu + sigma * λ - mu_t
    eq2 = mu**2 + sigma**2 + 2 * mu * sigma * λ + sigma**2 * δ - m2
    return [eq1, eq2]

def opt_truncnorm(mu_t, m2, b, r, c):
    sigma_t2 = m2 - mu_t**2
    initial_guess = [mu_t, np.sqrt(max(sigma_t2, 1e-6))]
    sol = least_squares(
        lambda p: trunc_normal_equations(p, mu_t, m2, b),
        initial_guess,
        bounds=([0, 0.01], [b, np.inf])
    )
    mu_opt, sigma_opt = sol.x
    crit_frac = (r - c) / r
    alpha, beta = -mu_opt / sigma_opt, (b - mu_opt) / sigma_opt
    target_cdf = norm.cdf(alpha) + crit_frac * (norm.cdf(beta) - norm.cdf(alpha))
    y_star = np.clip(mu_opt + sigma_opt * norm.ppf(target_cdf), 0, b)

    def truncated_cdf(x):
        z = (x - mu_opt) / sigma_opt
        return (norm.cdf(z) - norm.cdf(alpha)) / (norm.cdf(beta) - norm.cdf(alpha))
    integral, _ = quad(truncated_cdf, 0, y_star)
    profit_star = (r - c) * y_star - r * integral
    return y_star, profit_star


def profit_given_y_truncnorm(y, mu_t, m2, b, r, c):
    # 1️⃣ 矩匹配求解原始正态参数
    sigma_t2 = m2 - mu_t**2
    initial_guess = [mu_t, np.sqrt(max(sigma_t2, 1e-6))]
    sol = least_squares(
        lambda p: trunc_normal_equations(p, mu_t, m2, b),
        initial_guess,
        bounds=([0, 0.01], [b, np.inf])
    )
    mu_opt, sigma_opt = sol.x

    # 2️⃣ 构造截断分布的 CDF
    alpha, beta = -mu_opt / sigma_opt, (b - mu_opt) / sigma_opt
    Z = norm.cdf(beta) - norm.cdf(alpha)

    def truncated_cdf(x):
        if x <= 0: return 0
        elif x >= b: return 1
        z = (x - mu_opt) / sigma_opt
        return (norm.cdf(z) - norm.cdf(alpha)) / Z

    # 3️⃣ 按报童模型收益公式计算期望收益
    if y <= 0:
        return 0
    elif y <= b:
        integral, _ = quad(truncated_cdf, 0, y)
        profit = (r - c) * y - r * integral
    else:
        integral_b, _ = quad(truncated_cdf, 0, b)
        profit = r * mu_t - c * y
    return profit