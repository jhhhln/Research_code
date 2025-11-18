import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  
from Zi_y import Z1_y, Z2_y, Z3_y
from single_criteria import compute_Z1_values, compute_Z2_values, compute_Z3_values
from R_star_concave import optimize_y
from true_dis import (
    opt_uniform, uniform_expected_profit,
    opt_norm, expected_profit_norm,
    opt_truncnorm, profit_given_y_truncnorm
)


def run_single_experiment(c, r, I, m, b):
    Z1_star, y1 = compute_Z1_values(I, m, c, r, b)
    Z2_star, y2 = compute_Z2_values(I, m, c, r, b)
    Z3_star, y3 = compute_Z3_values(I, m, c, r, b)
    x1,Z_x1=optimize_y(I, m, c, r, b, lambda_val=0.25)
    x2,Z_x2=optimize_y(I, m, c, r, b, lambda_val=0.5)

    Z2_y1=Z2_y(y1,I,m,c,r,b)
    Z3_y1=Z3_y(y1,I,m,c,r,b)
    Z1_y2=Z1_y(y2,I,m,c,r,b)
    Z3_y2=Z3_y(y3,I,m,c,r,b)
    Z1_y3=Z1_y(y3,I,m,c,r,b)
    Z2_y3=Z2_y(y3,I,m,c,r,b)
    m_1=m[1]
    m_2=m[1]**2+m[1]/2
    y_star_norm, profit_star_norm = opt_norm(m_1,m_2, b, r, c)
    profit_y1_norm = expected_profit_norm(y1, m_1,m_2, b, r, c)
    profit_y2_norm =expected_profit_norm(y2, m_1,m_2, b, r, c)
    profit_y3_norm = expected_profit_norm(y3, m_1,m_2, b, r, c)
    y_star_uni, profit_star_uni = opt_uniform(b, r, c)
    profit_y1_uni = uniform_expected_profit(y1, b, r, c)
    profit_y2_uni = uniform_expected_profit(y2, b, r, c)
    profit_y3_uni = uniform_expected_profit(y3, b, r, c)


    return {
        'c': c, 'r': r, 'm': m.tolist(), 'b': b, 'I': I,
        'c/r': c / r,
        'Z1_star': Z1_star, 'Z2_star': Z2_star, 'Z3_star': Z3_star,
        'y1': y1, 'y2': y2, 'y3': y3,'Z2_y1':Z2_y1,'Z3_y1':Z3_y1,'Z1_y2':Z1_y2,'Z3_y2':Z3_y2,'Z1_y3':Z1_y3,'Z2_y3':Z2_y3,
        'y_star_norm':y_star_norm,'profit_star_norm':profit_star_norm,'profit_y1_norm':profit_y1_norm,
        'profit_y2_norm':profit_y2_norm,'profit_y3_norm':profit_y3_norm,
        'y_star_uni':y_star_uni,'profit_star_uni':profit_star_uni,
        'profit_y1_uni':profit_y1_uni,
        'profit_y2_uni':profit_y2_uni,
        'profit_y3_uni':profit_y3_uni
    }
