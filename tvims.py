import csv
sum_all = 0
sample_aver = 0

sample = []

def moda_count(input):
    max_count = 0
    mod_val = None
    for i in range(len(input)):
        count = 0
        for j in range(len (input)):
            if(input[i]==input[j]):
                count+=1
        if count > max_count:
            max_count = count
            mod_val = input[i]
    if max_count == 1:
        return None
    else:
        return mod_val

def mediana_count(input):
    n = len(input)
    if n%2==1:
        return input[n/2]
    else:
        return ((input[int(n/2-1)]) + (input[int(n/2+1)]))/2
def sum_count(input):
    s = 0
    for row in input:
        s += row
    return s

def sample_aver_count(input):
    s = 0
    for row in input:
        s += row
    return s / len(input)

def sample_razm(input):
    srt = sorted(input)
    return srt[len(input) - 1] - srt[0]


def disp(input):
    sred = sample_aver_count(input)
    sum = 0
    for x in input:
        sum += (x - sred)**2
    smeshenaya = sum / len(input)
    if len(input)>1:
        nesmesenaya = sum / (len(input) - 1)
    else:
        nesmesenaya = None
    return (smeshenaya,nesmesenaya)

def moment_nach(input, poryadok):
    sum = 0
    for x in input:
        sum += x**poryadok
    return sum / len(input)

def moment_centr(input, poryadok):
    sum = 0
    sred = sample_aver_count(input)
    for x in input:
        sum += (x-sred)**poryadok
    return sum / len(input)

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

        print("aaa")


if __name__ == '__main__':
    main()