
import csv
import random
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.special import gammainc
from scipy.stats import nakagami
from scipy.special import gamma
from scipy.optimize import minimize
from scipy.optimize import brentq
sum_all = 0
sample_aver = 0

sample = []

def plot_mom_mle_ecdf(sample):
    """
    Строит на одном графике:
    - Теоретическую функцию распределения с параметрами методом моментов.
    - Теоретическую функцию распределения с параметрами методом максимального правдоподобия.
    - Эмпирическую функцию распределения по всей выборке.
    """

    # 1. Оценки параметров методом моментов
    mu_mom, loc_mom = method_of_moments_nakagami(sample)

    # 2. Оценки параметров методом максимального правдоподобия
    sorted_sample = sorted(sample)
    mu_mle, loc_mle, skale = nakagami.fit(sample, fscale = 1)

    # 3. Эмпирическая функция распределения
    ecdf_points = empirical_cdf(sample)
    ecdf_x = [point[0] for point in ecdf_points]
    ecdf_y = [point[1] for point in ecdf_points]

    # 4. Подготовка оси X
    x_min = min(sample)
    x_max = max(sample)
    x_values = np.linspace(x_min, x_max, 500)

    # 5. Вычисляем значения теоретических ФР
    cdf_mom = theoretical_cdf_nakagami(x_values, mu_mom, loc_mom)
    cdf_mle = theoretical_cdf_nakagami(x_values, mu_mle, loc_mle)

    # 6. Построение графика
    plt.figure(figsize=(10, 6))

    # Теоретическая по методу моментов
    plt.plot(x_values, cdf_mom, label=f"Метод моментов (μ={mu_mom:.2f}, loc={loc_mom:.2f})", color='red', linewidth=2)

    # Теоретическая по ММП
    plt.plot(x_values, cdf_mle, label=f"Метод МП (μ={mu_mle:.2f}, loc={loc_mle:.2f})", color='green', linewidth=2)

    # Эмпирическая функция распределения
    plt.step(ecdf_x, ecdf_y, where='post', label="Эмпирическая ФР", color='blue', linewidth=2)

    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Теоретические и эмпирическая функции распределения")
    plt.grid(True)
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(0, 1.05)
    plt.show()

def bootstrap_bias_mle(data, num_bootstrap_samples=1000):
    mu_estimates = []
    scale_estimates = []
    loc_estimates = []

    n = len(data)
    for _ in range(num_bootstrap_samples):
        bootstrap_sample = [random.choice(data) for _ in range(n)]
        sorted_bootstrap = sorted(bootstrap_sample)
        mu_hat, loc_hat, scale_hat = nakagami.fit(bootstrap_sample, floc=sorted_bootstrap[0], fscale=1)
        mu_estimates.append(mu_hat)
        loc_estimates.append(loc_hat)
        scale_estimates.append(scale_hat)

    # Средние значения оценок
    mean_mu = np.mean(mu_estimates)
    mean_loc = np.mean(loc_estimates)

    # Оригинальные оценки
    sorted_data = sorted(data)
    mu_original, loc_original, scale_original = nakagami.fit(data, floc=sorted_data[0], fscale=1)

    # Смещения
    bias_mu = mean_mu - mu_original
    bias_loc = mean_loc - loc_original


    return bias_mu, bias_loc

def rand_sub_sample(data, size):
    return random.sample(data,size)

# Функция для вычисления выборочной дисперсии
def sample_disp_count(data):
    return np.var(data, ddof=0)

# Функция для решения уравнения для mu методом бисекции
def solve_mu(S2):
    def eq_mu(mu):
        # Вычисляем коэффициент r = Gamma(mu+0.5)^2 / (Gamma(mu)^2 * mu)
        log_r = 2 * math.lgamma(mu + 0.5) - 2 * math.lgamma(mu) - math.log(mu)
        r = math.exp(log_r)
        return (1 - r) - S2  # Приравниваем к S2
    
    # Интервал для поиска mu
    mu_lower = 0.5001  # Начальный нижний предел
    mu_upper = 1e5     # Верхний предел
    # Метод бисекции для нахождения корня
    while mu_upper - mu_lower > 1e-5:  # Точность
        mu_mid = (mu_lower + mu_upper) / 2
        if eq_mu(mu_mid) > 0:
            mu_lower = mu_mid
        else:
            mu_upper = mu_mid
    
    return (mu_lower + mu_upper) / 2

# Функция для вычисления значения смещения (delta)
def solve_delta(mu, barX):
    log_delta_coeff = math.lgamma(mu + 0.5) - math.lgamma(mu) - 0.5 * math.log(mu)
    return barX - math.exp(log_delta_coeff)

# Основная функция для выполнения метода моментов
def method_of_moments_nakagami(data):
    # Вычисляем выборочное среднее и дисперсию
    barX = sample_aver_count(data)
    S2 = sample_disp_count(data)
    
    # 1. Находим mu из уравнения для дисперсии
    mu_hat = solve_mu(S2)
    
    # 2. Находим delta из первого уравнения для среднего
    delta_hat = solve_delta(mu_hat, barX)
    
    return mu_hat, delta_hat
def neg_log_likelihood(params, sample):
    """
    Вычисляет отрицательный логарифм правдоподобия для распределения Накагами
    с параметрами формы (mu) и сдвига (loc), фиксированного масштаба = 1.
    params: [mu, loc]
    Возвращает: -ln L(params) 
    """
    mu_val, loc_val = params

    n = 300
    #sum_log = Σ ln(x_i − loc),   sum_sq = Σ (x_i − loc)^2
    logL = n * (math.log(2) + mu_val * math.log(mu_val) - math.lgamma(mu_val))
    sum_log = 0.0
    sum_sq  = 0.0
    for x in sample:
        dx = x - loc_val
        
        sum_log += math.log(dx)
        sum_sq  += dx * dx
    # logL += (2μ − 1)·sum_log  − μ·sum_sq
    logL += (2 * mu_val - 1) * sum_log - mu_val * sum_sq
    return -logL

def moda_count(data):
    max_count = 0
    mod_val = None
    for i in range(len(data)):
        count = 0
        for j in range(len (data)):
            if(data[i]==data[j]):
                count+=1
        if count > max_count:
            max_count = count
            mod_val = data[i]
    if max_count == 1:
        return None
    else:
        return mod_val
def mediana_count(data):
    data = sorted(data)
    n = len(data)
    if n%2==1:
        return data[n/2]
    else:
        return ((data[int(n/2-1)]) + (data[int(n/2+1)]))/2
    
def sum_count(data):
    s = 0
    for row in data:
        s += row
    return s

def sample_aver_count(data):
    s = 0
    for row in data:
        s += row
    return s / len(data)

def sample_razm(data):
    srt = sorted(data)
    return srt[len(data) - 1] - srt[0]


def disp(data):
    sred = sample_aver_count(data)
    sum = 0
    for x in data:
        sum += (x - sred)**2
    smeshenaya = sum / len(data)
    if len(data)>1:
        nesmesenaya = sum / (len(data) - 1)
    else:
        nesmesenaya = None
    return (smeshenaya,nesmesenaya)

def moment_nach(data, poryadok):
    sum = 0
    for x in data:
        sum += x**poryadok
    return sum / len(data)

def moment_centr(data, poryadok):
    sum = 0
    sred = sample_aver_count(data)
    for x in data:
        sum += (x-sred)**poryadok
    return sum / len(data)
# Функция для вычисления эмпирической функции распределения (ЭФР)
def empirical_cdf(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    ecdf_points = []
    for i, x in enumerate(sorted_data):
        ecdf_points.append((x, (i + 1) / n))
    return ecdf_points
def graph_ecdf(data):
    # Сортируем данные
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Определяем границы по оси x для визуализации (расширим немного левее и правее)
    x_min = sample_aver_count(sorted_data) - sample_razm(sorted_data)/2
    x_max = sample_aver_count(sorted_data) + sample_razm(sorted_data)/2
    
    # Рисуем базовую горизонтальную линию: F(x)=0 для x < первый элемент
    plt.hlines(0, x_min, sorted_data[0], colors='blue', lw=2)
    
    # Рисуем горизонтальные полосы для каждого шага ЭФР
    for i in range(n):
        # Левая граница текущего интервала – значение текущего элемента
        x_left = sorted_data[i]
        # Правая граница:
        # если это не последний элемент, то интервал до следующего,
        # иначе до x_max (чтобы линия продолжалась до конца графика)
        if i < n - 1:
            x_right = sorted_data[i+1]
        else:
            x_right = x_max
        # Значение ЭФР на данном интервале – (i+1)/n
        y_val = (i + 1) / n
        plt.hlines(y_val, x_left, x_right, colors='blue', lw=2)
    
    # Настройки графика
    plt.xlabel("Значение x")
    plt.ylabel("F(x)")
    plt.title(f"Эмпирическая функция распределения из {n} элементов")
    plt.xlim(x_min, x_max)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()
def compute_histogram(data):
    """
    Вычисляет гистограмму для данных.
    data - список числовых значений,
    m - число интервалов (бинов).
    
    Возвращает:
    - freq: список абсолютных частот для каждого интервала,
    - bins: список границ интервалов (длина m+1),
    - bin_width: ширина одного интервала.
    """
    m = int(1 + 3.332*math.log(len(data)))
    min_val = min(data)
    max_val = max(data)
    bin_width = (max_val - min_val) / m
    freq = [0] * m
    # Определяем границы интервалов
    bins = [min_val + i * bin_width for i in range(m + 1)]
    
    for x in data:
        # Вычисляем индекс бина для x
        # Если x совпадает с max_val, присваиваем последний бин
        index = int((x - min_val) / (max_val - min_val) * m)
        if index == m:
            index = m - 1
        freq[index] += 1
        
    return freq, bins, bin_width

def graph_histogram(data):
    """
    Строит гистограмму для данных data с числом интервалов m.
    Частоты вычисляются вручную с помощью compute_histogram,
    а затем построение происходит через plt.bar.
    """
    n = len(data)
    freq, bins, bin_width = compute_histogram(data)
    m = int(1 + 3.332*math.log(len(data)))
    # Вычисляем центры интервалов для размещения столбцов
    bin_centers = [bins[i] + bin_width/2 for i in range(m)]
    
    plt.bar(bin_centers, freq, width=bin_width, align='center', edgecolor='black', color='skyblue')
    plt.xlabel("Значение x")
    plt.ylabel("Частота")
    plt.title(f"Гистограмма выборки из {n} элементов")
    plt.grid(True)
    plt.show()
def theoretical_cdf_nakagami(x, nu, loc):
    """
    Вычисляет теоретическую ФР распределения Накагами с параметрами:
      nu  - параметр формы,
      loc - параметр смещения.
    Принимает x (может быть массивом). Для x < loc функция возвращает 0.
    Для x ≥ loc вычисляется как gammainc(nu, nu * (x - loc)**2).
    """
    # Если x меньше loc, возвращаем 0, иначе вычисляем gammainc
    x = np.asarray(x)
    cdf = np.where(x < loc, 0, gammainc(nu, nu * (x - loc)**2))
    return cdf
# def plot_theoretical_cdf(nu, loc):
#     """
#     Строит график теоретической ФР распределения Накагами с параметрами:
#       nu  - параметр формы,
#       loc - параметр смещения.
#     """
#     # Выбираем диапазон x от loc до loc + 5 (при scale=1 диапазон можно корректировать)
#     x_min = loc
#     x_max = loc + 5
#     x_values = np.linspace(x_min, x_max, 300)
#     y_values = theoretical_cdf_nakagami(x_values, nu, loc)
    
#     plt.plot(x_values, y_values, 'r-', lw=2, label=f"ФР (nu={nu}, loc={loc})")
#     plt.xlabel("x")
#     plt.ylabel("F(x)")
#     plt.title("Теоретическая функция распределения (Накагами)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
def plot_three_theoretical_cdfs_on_one(nu1, loc1, nu2, loc2, nu3, loc3):
    """
    Строит три теоретические ФР (для распределения Накагами)
    на одной координатной плоскости.
    
    Аргументы:
      nu1, loc1 - параметры для первой кривой,
      nu2, loc2 - параметры для второй кривой,
      nu3, loc3 - параметры для третьей кривой.
    
    Теоретическая ФР определяется по формуле:
      F(x; nu, loc) = 0, если x < loc,
      F(x; nu, loc) = gammainc(nu, nu * (x - loc)**2), если x >= loc.
    """
    def theoretical_cdf(x, nu, loc):
        x = np.asarray(x)
        return np.where(x < loc, 0, gammainc(nu, nu * (x - loc)**2))
    
    # Определяем общий диапазон x для всех кривых
    # Берем минимальное значение среди loc и максимальное значение равное max(loc)+5 (эмулируем достаточный диапазон)
    locs = [loc1, loc2, loc3]
    x_min = min(locs)
    x_max = max(locs) + 5
    x_values = np.linspace(x_min, x_max, 300)
    
    # Вычисляем теоретические ФР для каждого набора параметров
    y1 = theoretical_cdf(x_values, nu1, loc1)
    y2 = theoretical_cdf(x_values, nu2, loc2)
    y3 = theoretical_cdf(x_values, nu3, loc3)
    
    plt.figure(figsize=(8, 6))
    # Строим каждую кривую своим цветом
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y1, color='red', linewidth=2, 
             label=f"Кривая 1: μ={nu1:.2f}, loc={loc1:.2f}")
    plt.plot(x_values, y2, color='green', linewidth=2, 
             label=f"Кривая 2: μ={nu2:.2f}, loc={loc2:.2f}")
    plt.plot(x_values, y3, color='blue', linewidth=2, 
             label=f"Кривая 3: μ={nu3:.2f}, loc={loc3:.2f}")
    
    plt.xlabel("x", fontsize=12)
    plt.ylabel("F(x)", fontsize=12)
    plt.title("Теоретические функции распределения Накагами", fontsize=14)
    plt.legend(loc='lower right', fontsize=10, framealpha=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(x_min, x_max)
    plt.ylim(-0.05, 1.05)
    plt.show()
def main():
    with open('var_4_nakagami.csv', newline ='') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            sample.append(float(row[0]))
        n = len(sample)

        print(sample[0], "первый элемент")


        sum_all = sum_count(sample)
        print(sum_all, "сумма")

        sample_aver = sample_aver_count(sample)
        print(sample_aver, "среднее выборочное")

        sorted_sample = sorted(sample)

        razmah = sample_razm(sample)
        print(razmah, "размах")

        mediana = mediana_count(sample)
        print(mediana, "медиана")

        moda = moda_count(sample)
        if moda == None:
            print("все элементы встречаются один раз")
        else:
            print(moda, "мода")
        disp_bias,disp_non_bias = disp(sample)
        print(disp_bias, "смещенная дисперсия")
        print(disp_non_bias, "несмещенная дисперсия")
        print(moment_nach(sample,1), "момент начальный 1")
        print(moment_nach(sample,10), "момент начальный 10")
        print(moment_centr(sample,1), "момент центральный 1")
        print(moment_centr(sample,10), "момент центральный 10")
        ecdf_10 = empirical_cdf(rand_sub_sample(sample,10))
        ecdf_100 = empirical_cdf(rand_sub_sample(sample,100))
        ecdf_200 = empirical_cdf(rand_sub_sample(sample,200))
        # graph_ecdf(rand_sub_sample(sample,10))
        # graph_ecdf(rand_sub_sample(sample,100))
        # graph_ecdf(rand_sub_sample(sample,200))
        # graph_histogram(rand_sub_sample(sample, 10))
        # graph_histogram(rand_sub_sample(sample, 100))
        # graph_histogram(rand_sub_sample(sample, 200))
        # plot_three_theoretical_cdfs_on_one(0.5,1,4,1,10,1)
        # plot_three_theoretical_cdfs_on_one(5,1,5,10,5,3)
        # Данные (например, амплитуда сигнала)

        # Оценка параметров
        nu, loc, scale = nakagami.fit(sample, fscale = 1)  # floc=0 фиксирует loc (сдвиг) на 0
        print(f"Оценки: nu = {nu:.2f}, scale = {scale:.2f},loc = {loc:.2f}")
        # Оценка методом моментов
        mu_mom, loc_mom = method_of_moments_nakagami(sample)

        print(f"Метод моментов:\nmû = {mu_mom:.6f}, δ̂ = {loc_mom:.6f}")

        min_x = min(sample)
        initial = [2, 2]
        bounds = [(1e-6, None), (None, min_x)]
        result = minimize(
            neg_log_likelihood,
            initial,
            args=(sample,),
            method='L-BFGS-B',
            bounds=bounds
        )
        mu_mle, loc_mle = result.x
        print(f"Метод ММП → mu = {mu_mle:.6f}, loc = {loc_mle:.6f}")

        bias_mu, bias_loc = bootstrap_bias_mle(sample)
        print(f"Bias mu: {bias_mu:.5f}")
        print(f"Bias loc: {bias_loc:.5f}")

        plot_mom_mle_ecdf(sample)
        
        

if __name__ == '__main__':
    main()
