import re

def no_three_letters_in_row(word):
    fixed_word = re.sub(r"(\w)\1{2,}", r'\1' * 2, word)
    return fixed_word

if __name__ == "__main__":
    print('just one character occurs more than three times consecutively')
    print(no_three_letters_in_row('shooooot'))
    print('testing more than one character occurs more than three times consecutively')
    print(no_three_letters_in_row('shhhhhhhiiiiiiiit'))
    print('one character occurs exactly three times consecutively')
    print(no_three_letters_in_row('shooot'))
