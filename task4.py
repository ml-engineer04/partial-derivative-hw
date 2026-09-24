import numpy as np

# 4. Toshkent kvartira narxlari (mini-loyiha)

# Maydon (×10 m²) 4  5  6  7  8
# Narx (ming $)   30 36 44 50 58
# Model: narx = w·maydon + b

# a) w = 0, b = 0 dan boshlab gradient descentni 5000 qadam ishga tushiring. lr = 0.005, 0.02 va 0.03 qiymatlarini sinang.

m = np.array([4, 5, 6, 7, 8])
y = np.array([30, 36, 44, 50, 58])


def price(w, b):
    pred = w * m + b
    return pred


def loss(w, b):
    pred = price(w, b)
    return np.mean((pred - y) ** 2)


def gradient(w, b):
    h = 0.0001

    dw = (loss(w + h, b) - loss(w, b)) / h
    db = (loss(w, b + h) - loss(w, b)) / h

    return np.array([dw, db])


w = 0
b = 0
lr1 = 0.005
lr2 = 0.02
lr3 = 0.03

# b) 10-, 100-, 1000- va 5000-qadamlarda w, b va cost ni chop qiling.
for step in range(5000):

    grad = gradient(w, b)

    w = w - lr1 * grad[0]
    b = b - lr1 * grad[1]

    if step + 1 == 10 or step + 1 == 100 or step + 1 == 1000 or step + 1 == 5000:
        print(f"step {step+1}: lr={lr1}: w={w:.4f}: b={b:.4f}: loss={loss(w, b):.4f}")

w = 0
b = 0

print("\n" + "-" * 30)
for step in range(5000):

    grad = gradient(w, b)

    w = w - lr2 * grad[0]
    b = b - lr2 * grad[1]

    if step + 1 == 10 or step + 1 == 100 or step + 1 == 1000 or step + 1 == 5000:
        print(f"step {step+1}: lr={lr2}: w={w:.4f}: b={b:.4f}: loss={loss(w, b):.4f}")

w = 0
b = 0

print("\n" + "-" * 30)

for step in range(5000):

    grad = gradient(w, b)

    w = w - lr3 * grad[0]
    b = b - lr3 * grad[1]

    if step + 1 == 10 or step + 1 == 100 or step + 1 == 1000 or step + 1 == 5000:
        print(f"step {step+1}: lr={lr3}: w={w:.4f}: b={b:.4f}: loss={loss(w, b):.4f}")


# c) lr = 0.03 da nima bo'ldi? Nega?
#  lr = 0.03 da bizda portlash yuz berdi chunki qadam kattalish ketdi

# d) w tez o'rnashadi, b esa sekin. Nima uchun?
# w tezroq o'rnashadi, chunki uning gradientida
# m qiymatlari (4, 5, 6, 7, 8) qatnashadi.
# b esa faqat qo'shiladi, shuning uchun uning gradienti kichikroq.

# e) Natijani np.polyfit(maydon, narx, 1) bilan solishtiring.

print("\n" + "-" * 30)

coef = np.polyfit(m, y, 1)

w_polyfit = coef[0]
b_polyfit = coef[1]

print("np.polyfit:")
print(f"w = {w_polyfit:.4f}")
print(f"b = {b_polyfit:.4f}")

# np.polyfit() to'g'ridan-to'g'ri eng yaxshi chiziqni hisoblaydi.
# Gradient Descent esa w va b ni bosqichma-bosqich yangilab,
# shu optimal qiymatlarga yaqinlashadi.
# Shuning uchun ikkala usulning natijalari bir-biriga yaqin chiqyabdi.

# f) 65 m² kvartira narxini bashorat qiling.
print("\n" + "-" * 30)

w = 7
b = 1.2

m_new = 6.5
predicted_price = w * m_new + b

print(f"65 m² kvartira narxi: {predicted_price:.2f} ming $")
