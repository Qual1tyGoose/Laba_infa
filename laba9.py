f, slovar = [], {}
with open('text_dlya_labi_9.txt', 'r', encoding='UTF-8') as file:
    for i in file.readlines():
        slovar[i.split()[0]] = sum(list(map(float, i.split()[1:])))
file = open('Good.txt', 'w+')
for i in dict(sorted(slovar.items(), key=lambda item: item[1], reverse=True)).items():
    file.write(f'{i[0]}: {i[1]}\n')
file.close()
