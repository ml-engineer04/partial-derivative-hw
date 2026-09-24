import numpy as np

# 5. Ikki xususiyatli uy modeli ★
# Maydon (×10 m²) 4  6  5  8
# Xonalar         2  3  2  4
# Narx (ming $)   36 49 41 62
# Model: narx = w1·maydon + w2·xonalar + b

# a) C(w1, w2, b) ni va uchta qisman hosila formulasini yozing.

maydon = np.array([4, 6, 5, 8])
xonalar = np.array([2, 3, 2, 4])
narx = np.array([36, 49, 41, 62])


def cost_func(w1, w2, b):
    prediction = w1 * maydon + w2 * xonalar + b
    return np.mean((prediction - narx) ** 2)


def gradient(w1, w2, b):
    h = 0.0001
    df_maydon = (cost_func(w1 + h, w2, b) - cost_func(w1, w2, b)) / h
    df_xonalar = (cost_func(w1, w2 + h, b) - cost_func(w1, w2, b)) / h
    df_db = (cost_func(w1, w2, b + h) - cost_func(w1, w2, b)) / h
    return np.array([df_maydon, df_xonalar, df_db])


# b) Gradient descentni lr = 0.02 bilan ishga tushiring. 100, 1000 va 5000 qadamdan
# keyingi natijalarni yozing.

w1 = 0.0
w2 = 0.0
b = 0.0
lr = 0.02

for step in range(5000):

    grad = gradient(w1, w2, b)

    w1 = w1 - lr * grad[0]
    w2 = w2 - lr * grad[1]
    b = b - lr * grad[2]

    condition = step + 1 == 100 or step + 1 == 1000 or step + 1 == 5000

    if condition:
        print(
            f"step {step+1}:  w1 = {w1:.4f};  w2 = {w2:.4f};  b = {b:.4f};  cost = {cost_func(w1, w2, b):.4f}"
        )

# ============================================================ RESULT ============================================================
# step 100:  w1 = 6.4209;  w2 = 2.4229;  b = 2.9957;  cost = 2.8753
# step 1000:  w1 = 5.7474;  w2 = 1.8622;  b = 8.7846;  cost = 0.0945
# step 5000:  w1 = 5.0035;  w2 = 2.9943;  b = 9.9953;  cost = 0.0000

# w1 = 5, w2 = 3, b = 10
# ============================================================ RESULT ============================================================


# c) Yakuniy w1, w2, b qiymatlari real hayotda nimani anglatadi?

# w1 -> maydonning narxga ta'siri
#      Maydon 1 birlik (10 m²) oshsa, narx w1=5 ming $ ga o'zgaradi.
#
# w2 -> xonalar sonining narxga ta'siri
#      Xonalar 1 taga oshsa, narx w2=3 ming $ ga o'zgaradi.
#
# b=10 -> modelning boshlang'ich (bazaviy) qiymati.


# d) Gradient vektor necha o'lchamli? Modelga yana 2 ta xususiyat qo'shsak-chi?

# Hozir: 3 ta parametr -> gradient 3 o'lchamli -> [w1, w2, b]

# Yana 2 ta xususiyat qo'shsak:
# 5 ta parametr -> gradient 5 o'lchamli -> [w1, w2, w3, w4, b]
