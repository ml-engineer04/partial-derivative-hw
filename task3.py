import numpy as np

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
