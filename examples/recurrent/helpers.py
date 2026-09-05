import itertools
import math
import numpy as np

"""Data Generation"""

ALPHABET = ["A", "B", "D", "O", "E", "K"]
TERMS = ["AB", "ED", "OK"]

def generate_valid_seq(length: int) -> str:
    out = ""
    ri = np.random.randint(0, len(TERMS), length, dtype=int)
    for i in ri:
        out = f"{out}{TERMS[i]}"

    return out

def generate_valid_seqs(row: int, col: int) -> list[str]:
    outs = []
    for _ in range(row):
        seq = generate_valid_seq(col)
        outs.append(seq)
    
    return outs

def generate_invalid_seq(length: int) -> str:
    out = ""
    ri = np.random.randint(0, len(ALPHABET), length, dtype=int)
    for i in ri:     
        out = f"{out}{ALPHABET[i]}"

    return out

def generate_invalid_seqs(row: int, col: int) -> list[str]:
    outs = []
    for _ in range(row):
        seq = generate_invalid_seq(col)
        outs.append(seq)
    
    return outs

"""Tokenizer"""

VOWELS = {"A", "E", "O"}

def tokenize(input: str | list[str] | list[list[str]]) -> np.ndarray:
    # list of strings / list of lists of strings: recurse elementwise and stack
    if isinstance(input, list):
        return np.stack([tokenize(elem) for elem in input])

    # multi-character string: recurse letter by letter (none of them stand alone)
    if isinstance(input, str) and len(input) > 1:
        return np.stack([tokenize(letter) for letter in input])

    # base case: a single character
    is_vowel = input in VOWELS
    is_consonant = input in ALPHABET and not is_vowel
    position = ALPHABET.index(input) + 1 if input in ALPHABET else 0

    return np.array([is_vowel, is_consonant, position], dtype=np.float32)

VECT_TERMS = {term: tokenize(term) for term in ALPHABET}

"""Parser"""

def parse_vect(input: np.ndarray) -> bool:
    assert len(input.shape) == 2

    prev = None
    for c in input:
        if np.array_equal(c, VECT_TERMS['A']):
            prev = 'A'
        elif np.array_equal(c, VECT_TERMS['B']):
            if prev != 'A': return False
            prev = ''
        elif np.array_equal(c, VECT_TERMS['O']):
            prev = 'O'
        elif np.array_equal(c, VECT_TERMS['D']):
            if prev != 'E': return False
            prev = ''
        elif np.array_equal(c, VECT_TERMS['E']):
            prev = 'E'
        elif np.array_equal(c, VECT_TERMS['K']):
            if prev != 'O': return False
            prev = ''
        else:
            return False

    # a sequence that ends on a dangling opener (A/E/O with no matching
    # closer) is an incomplete term, not a valid one
    if prev in ('A', 'E', 'O'):
        return False

    return True


"""Matplotlib"""
import matplotlib.pyplot as plt

def plot_metrics(title:str, metrics: dict, epochs: int, eval_step: int):
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    loss_ax.plot(range(epochs), metrics['train_loss'], label="train")
    loss_ax.plot(range(0, epochs, eval_step), metrics['eval_loss'], label="test")
    loss_ax.set_title("Losses")
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("loss")
    loss_ax.legend()

    accuracy_ax.set_title("Accuracy on test sequence")
    accuracy_ax.plot(range(0, epochs, eval_step), metrics['accuracy'], label="accuracy")
    accuracy_ax.set_xlabel("epoch")
    accuracy_ax.set_ylabel("accuracy")
    accuracy_ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
