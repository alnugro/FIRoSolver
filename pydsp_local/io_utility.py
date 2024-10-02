import random


def to_fixed_point(value, fractional_bits):
    return value / 2**fractional_bits


def from_fixed_point(values, fractional_bits):
    return [int(value * 2**fractional_bits) for value in values]


def get_random_values(min_val, max_val, how_many):
    return [random.randint(min_val, max_val) for _ in range(how_many)]