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
