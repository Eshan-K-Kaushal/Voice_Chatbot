from T5_FT_CKPT import *
while True:
    question = str(input('User:'))
    print(generate_answer_custom(question))

    if question == 'EXIT_NOW':
        break
