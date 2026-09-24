import numpy as np
import sympy as sp

# Ko'p o'zgaruvchili funksiya → qisman hosila → gradient vektor → ko'p parametrli gradient
# descent. 1–3-masalalarni qo'lda bajaring, keyin kod bilan tekshiring. 4–5-masalalar uchun
# Python (NumPy) kerak.

# Kafe foydasi
# Kafe kuniga x yuz dona somsa va y yuz porsiya lag'mon sotadi. Kunlik foyda (ming
# so'm):

# a) ∂P/∂x va ∂P/∂y ni qo'lda toping, keyin SymPy bilan tekshiring.

# ============================================================
# manual calculation
# ============================================================

# dp_dx = 60 + 0 - 2*2*x - 0 - y = -4*x - y + 60
# dp_dy = 0 + 50 - 0 - 2*y - x = -x - 2*y + 50

# ============================================================
# with sympy
# ============================================================

x, y = sp.symbols("x, y")

p = 60 * x + 50 * y - 2 * x**2 - y**2 - x * y

dp_dx = sp.diff(p, x)
dp_dy = sp.diff(p, y)

print(f"dp_dx  ->  {dp_dx} \ndp_dy  ->  {dp_dy}")

# b) Hozir kafe x = 5, y = 10 sotmoqda. Shu nuqtada qisman hosilalarni hisoblang.
# Foydani oshirish uchun nimani ko'paytirish kerak?

# ============================================================
# manual calculation
# ============================================================

# dp_dx = 60 + 0 - 2*2*x - 0 - y = -4*x - y + 60 = -4*5 - 10 + 60 = 30
# dp_dy = 0 + 50 - 0 - 2*y - x = -x - 2*y + 50 = -5 - 2*10 + 50 = 25

# ============================================================
# with sympy
# ============================================================

dx_value = dp_dx.subs({x: 5, y: 10})
dy_value = dp_dy.subs({x: 5, y: 10})

print(f"dx value  ->  {dx_value} \ndy value  ->  {dy_value}")

# ============================================================
# Somsa (x) va lagmon (y) sotuvini ikkalasini ham ko‘paytirish kerak.
# Shu nuqtada somsa (x=30, y=25, x>y) foydaga kuchliroq ta’sir qilmoqda.
# ============================================================


# c) ∂P/∂x = 0 va ∂P/∂y = 0 tenglamalarini yechib, foyda eng katta bo'ladigan (x, y) ni
# toping. Maksimal foyda qancha?

# ============================================================
# manual calculation
# ============================================================

# -4*x - y + 60 = 0 -> 4x+y = 60 / *2 -> 8x+2y = 120 -> 7x = 70    -> x=10
# -x - 2*y + 50 = 0 -> x+2y = 50       -> x+2y = 50  -> 10+2y = 50 -> y=20
# P(10, 20) = 60*10 + 50*20 - 2 * 10**2 -20**2 - 10*20 = 800
# maksimal foyda = 800 ming so'm

# ============================================================
# with sympy
# ============================================================

x_value = 10
y_value = 20

p_value = p.subs({x: x_value, y: y_value})

print(f"maksimal foyda = {p_value}")


# d) Nega −x**2 va −y**2 hadlari bor? Real hayotda ular nimani ifodalaydi?

# buni sababi ko‘proq somsa tayyorlash uchun koproq xarajat qilinadi 60x bizda foyda bolsa -x**2
# harajatlar uchun chiqib ketadi. Lagmon uchun ham huddi shunday boladi

# ========================================================================================================================

# Reklama byudjeti
# Kompaniya TV va Instagram reklamasiga pul sarflaydi (million so'm). Oylik sotuv:
# S(tv, insta) = 20·√tv + 30·√insta
# Hozirgi byudjet: tv = 100, insta = 36.

# a) Hozirgi sotuvni hisoblang.

# ============================================================
# manual calculation
# ============================================================

# S(tv, insta) = 20·√tv + 30·√insta
# tv = 100
# insta = 36
# S(100, 36) = 20·√100 + 30·√36 = 20*10+30*6 = 200+180 = 380 million so'm

# ============================================================
# with function
# ============================================================


def s(tv, insta):
    return 20 * np.sqrt(tv) + 30 * np.sqrt(insta)


tv = 100
insta = 36

print(f"Hozirgi sotuv  ->  {s(tv, insta)} million so'm")

# b) Darsdagi gradient(x, y) shablonini ishlatib, ∇S ni numerik hisoblang.


def gradient(x, y):
    h = 0.0001
    dx = (s(x + h, y) - s(x, y)) / h
    dy = (s(x, y + h) - s(x, y)) / h
    return np.array([dx, dy])


grad = gradient(tv, insta)

print(f" ∇S   ->  {grad}")
print(f"|∇S|  ->  {np.linalg.norm(grad):.4f}")


# c) Qo'lda tekshiring (maslahat: √t = t**0.5).

# S(tv, insta) = 20 * tv**0.5 + 30 * insta**0.5
# d_tv = 20*0.5 * tv**(-0.5) + 0 = 10/√t
# d_insta = 0 + 30*0.5 * insta**(-0.5) = 15/√insta

# tv = 100
# insta = 36

# d_tv    ->  10/√t = 10/10 = 1
# d_insta ->  15/√insta = 15/6 = 2.5
# ∇S=(1, 2.5)

# d) Qo'shimcha 1 million so'm bor. Uni qaysi kanalga sarflash foydaliroq? Gradient buni
# qanday ko'rsatadi?

# Bu yerda 2.5 > 1, yani hozirgi nuqtada Instagram byudjetini 1 million so‘mga oshirish
# sotuvga TV byudjetini 1 millionga oshirishdan kora kattaroq tasir beradi.

# e) Nega TV da har bir qo'shimcha so'mning samarasi pastro

# TV byudjeti oshgani sari har bir qo‘shimcha so‘mning samarasi kamayadi,
# chunki √tv funksiyasida mavjud va bu maxrajda tv budjed qancha oshsa samarasi kamayadi.

# ========================================================================================================================

# 3. Imtihon ballini bashorat qilish
# Model: ball = w·soat + b. Talaba 3 soat o'qib, 75 ball oldi. Shu talaba uchun loss:
# L(w, b) = (w·3 + b − 75)2
# Hozirgi parametrlar: w = 10, b = 20.

# a) Model qancha ball bashorat qiladi? Loss qancha?

w = 10
b = 20


def loss_func(w, b):
    pred = w * 3 + b
    return (pred - 75) ** 2


prediction = w * 3 + b

print("Prediction:", prediction)
print("Loss:", loss_func(w, b))


# b) ∂L/∂w va ∂L/∂b ni qo'lda hisoblang.

# L(w, b) = (w * 3 + b - 75)**2

# d_w = 2 * (w*3 + b - 75) * 3
# d_b = 2 * (w*3 + b - 75) * 1

# w = 10
# b = 20

# d_w -> 2 * (10*3 + 20 - 75) * 3
#      -> 2 * (-25) * 3
#      -> -150

# d_b -> 2 * (10*3 + 20 - 75) * 1
#      -> 2 * (-25) * 1
#      -> -50

# ∇L = (-150, -50)


# c) lr = 0.01 bilan bitta gradient descent qadamini bajaring. Yangi w, b va yangi loss ni toping.


w = 10
b = 20
lr = 0.01


def gradient_loss(w, b):
    h = 0.0001

    dw = (loss_func(w + h, b) - loss_func(w, b)) / h
    db = (loss_func(w, b + h) - loss_func(w, b)) / h

    return np.array([dw, db])


grad_loss = gradient_loss(w, b)

for step in range(1):

    w = w - lr * grad_loss[0]
    b = b - lr * grad_loss[1]

    print(f"step {step+1}:  w={w:.4f};  b={b:.4f};  loss={loss_func(w, b):.4f}")


# d) Nega ∂L/∂w ning moduli ∂L/∂b nikidan 3 marta katta?
# ∂L/∂w ning moduli ∂L/∂b nikidan 3 marta katta,
# chunki w modelda 3 ga ko‘paytirilgan.

# ========================================================================================================================
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

for step in range(5000):

    grad = gradient(w, b)

    w = w - lr2 * grad[0]
    b = b - lr2 * grad[1]

    if step + 1 == 10 or step + 1 == 100 or step + 1 == 1000 or step + 1 == 5000:
        print(f"step {step+1}: lr={lr2}: w={w:.4f}: b={b:.4f}: loss={loss(w, b):.4f}")

w = 0
b = 0

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

w = 7
b = 1.2

m_new = 6.5
predicted_price = w * m_new + b

print(f"65 m² kvartira narxi: {predicted_price:.2f} ming $")


# ========================================================================================================================
