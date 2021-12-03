from sequence_info.sequences import sequence_to_number

def raw_coin_to_number(raw_coin):
    return int(raw_coin.split(":")[0][-1])

def raw_sequence_to_number(raw_sequence):
    sequence = raw_sequence.split(": ")[-1].strip()
    return sequence_to_number[sequence]

def google_form_question_to_coin_sequence(question):
    raw_sequence, raw_coin = question.split("[")
    coin = raw_coin_to_number(raw_coin)
    sequence = raw_sequence_to_number(raw_sequence)

    return (coin, sequence)