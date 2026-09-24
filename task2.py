import numpy as np

# 2. Reklama byudjeti
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
