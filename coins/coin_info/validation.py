def coin_3_validation(sequence):
    correct = ["H", "T"] * (len(sequence) // 2) + ["H"] * (len(sequence) % 2)
    return correct == sequence

def coin_4_validation(sequence):
    for i,flip in enumerate(sequence[:-1]):
        if flip == "T" and sequence[i+1] != "H":
            return False 
    return True 

def coin_5_validation(sequence):
    pattern = ["H", "T", "T"]
    for i, flip in enumerate(sequence):
        if flip != pattern[i % 3]:
            return False 
    return True

def coin_6_validation(sequence):
    for i, flip in enumerate(sequence[:-2]):
        if flip == "T" and sequence[i + 1] == "T" and sequence[i+2] != "H":
            return False 
    return True

def validation(coin, sequence):
    mapping = {
        1: True,
        2: True ,
        3: coin_3_validation,
        4: coin_4_validation,
        5: coin_5_validation,
        6: coin_6_validation,
    }

    return mapping[coin](sequence)