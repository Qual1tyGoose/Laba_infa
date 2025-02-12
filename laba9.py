with open('text_dlya_labi_9.txt', 'r') as file:
    for i in file.readlines():
        f = i.split()
index = 0
index_c = 0
mmax = str()
for i in f:
    if index % 2 == 0:
        if mmax == str():
            mmax = int(i)
        elif mmax < int(i):
            mmax = int(i)
            index_c = index
    index += 1
f[index_c] = str(index_c)
file = open('Good.txt', 'w+')
for i in f:
    file.write(f'{int(i)} ')
file.close()
