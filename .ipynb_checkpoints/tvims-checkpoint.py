import csv
import random
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.special import gammainc
sum_all = 0
sample_aver = 0

sample = []

def rand_sub_sample(data, size):
    return random.sample(data,size)

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
    freq, bins, bin_width = compute_histogram(data)
    m = int(1 + 3.332*math.log(len(data)))
    # Вычисляем центры интервалов для размещения столбцов
    bin_centers = [bins[i] + bin_width/2 for i in range(m)]
    
    plt.bar(bin_centers, freq, width=bin_width, align='center', edgecolor='black', color='skyblue')
    plt.xlabel("Значение x")
    plt.ylabel("Частота")
    plt.title("Гистограмма выборки")
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
    plt.plot(x_values, y1, color='red', linewidth=2, label=f"nu={nu1}, loc={loc1}")
    plt.plot(x_values, y2, color='green', linewidth=2, label=f"nu={nu2}, loc={loc2}")
    plt.plot(x_values, y3, color='blue', linewidth=2, label=f"nu={nu3}, loc={loc3}")
    
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Теоретические функции распределения (Накагами) для разных параметров")
    plt.legend()
    plt.grid(True)
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
        # graph_ecdf(rand_sub_sample(sample,100))
        # graph_histogram(rand_sub_sample(sample, 100))
        # plot_theoretical_cdf(100,-5)
        plot_three_theoretical_cdfs_on_one(1,1,1,10,1,20)
        

if __name__ == '__main__':
    main()