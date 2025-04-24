import math
import csv
import numpy as np
from scipy.special import digamma

target = 0.860697
sample = []
accum_u = 0
count_u = 0

# Поиск u по формуле
for u in np.arange(2, 10, 1e-5):
    a = math.lgamma(u + 0.5) / (math.lgamma(u) * (u ** 0.5))
    if abs(a - target) < 1e-4:
        accum_u += u
        count_u += 1

# Чтение выборки
with open('var_4_nakagami.csv', newline='') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        sample.append(float(row[0]))

n = len(sample)
accum_u_mle = 0
accum_loc_mle = 0
count_mle = 0

for u in np.arange(2, 10, 1e-2):
    for loc in np.arange(2, 10, 1e-2):
        y = [x - loc for x in sample]
        if any(yi <= 0 for yi in y):
            continue  # логарифм от <= 0 нельзя брать

        log_sum = sum(math.log(yi) for yi in y)
        sqr_sum = sum(yi ** 2 for yi in y)
        inv_sum = sum(1 / yi for yi in y)
        plain_sum = sum(yi for yi in y)

        a = n * (math.log(u) + 1 - digamma(u)) + 2 * log_sum - sqr_sum
        b = -(2 * u - 1) * inv_sum + 2 * u * plain_sum

        if abs(a) < 1e0 and abs(b) < 1e0:
            accum_u_mle += u
            accum_loc_mle += loc
            count_mle += 1

if count_mle > 0:
    print("MLE оценка u:", accum_u_mle / count_mle)
    print("MLE оценка loc:", accum_loc_mle / count_mle)
else:
    print("Подходящие параметры не найдены.")

if count_u > 0:
    print("Метод моментов — оценка u:", accum_u / count_u)