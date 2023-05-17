from layer_naiv import MulLayer

def apple(number=2, price=100, tax=1.1):
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    apple_price = mul_apple_layer.forward(price, number)
    tax_price = mul_tax_layer.forward(apple_price, tax)

    print(tax_price)

    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, apple_num = mul_apple_layer.backward(dapple_price)

    print(dapple, apple_num, dtax)

if __name__ == '__main__':
    apple()