# Текст содержит слова и целые числа произвольного порядка. Найти сумму включенных в текст чисел
a = 'ffasgdfut763ghda5sghdg376dty6asdg'
f = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
s = 0
stroka = ''
index = 0
flag = False
for i in a:
    if i in f:
        stroka += i
        flag = True
    else:
        flag = False
    if flag:
        pass
    else:
        if stroka:
            s += int(stroka)
        stroka = ''
print(s)
