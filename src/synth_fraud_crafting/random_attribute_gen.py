from datetime import timedelta
import random
import string
import hashlib
from preprocessing import const


def round_dt_to_second(dt_object):
    """
    Rounds a datetime object to the nearest second

    :param dt_object: A datetime to be rounded
    :return: A datetime rounded to the nearest second
    """
    new_dt = dt_object
    if dt_object.microsecond >= 500000:
        new_dt = new_dt + timedelta(seconds=1)
    return new_dt.replace(microsecond=0)


def generate_random_hashed_string(string_length):
    """
    Generates a random hashed (md5) string

    :param string_length: String length in input to the hash function
    :return: A random hashed string
    """
    char_allowed = string.ascii_letters + string.digits
    random_string = "".join(random.choice(char_allowed) for _ in range(string_length))
    return hashlib.md5(random_string.encode()).hexdigest()


def generate_attacker_iban_attack():
    iban = generate_random_hashed_string(34)
    int_choice = random.randint(1, 100)
    if int_choice <= 50:
        iban_cc = "IT"
    else:
        iban_cc = random.choice(const.EUROPEAN_CC)
    return iban, iban_cc


def generate_attacker_iban():
    """
    Randomly generates an attacker iban
    Generates an iban_cc based on approximation of the real fraud iban_cc
    distribution

    :return: A string representing the iban and a string of a country code
    """
    iban = generate_random_hashed_string(34)
    int_choice = random.randint(1, 100)
    if int_choice <= 40:
        iban_cc = "IT"
    elif int_choice <= 80:
        most_probable = ["FR", "DE", "GB", "RO", "ES"]
        iban_cc = random.choice(most_probable)
    else:
        # small probability we still select RO, FR, DE, GB, but that's ok
        iban_cc = random.choice(const.EUROPEAN_CC)
    return iban, iban_cc


def generate_random_hashed_IP():
    """
    Generates a random IP and then hashes it
    Considering a fraudster usually uses a VPN, randomly generating it doesn't
    really matter

    :return: An hashed (md5) IP
    """
    ip_32 = []
    for _ in range(4):
        ip_32.append(
            str(int("".join([random.choice(["0", "1"]) for _ in range(8)]), 2))
        )
    ip_32 = ".".join(ip_32)
    return hashlib.md5(ip_32.encode()).hexdigest()


def generate_ASN_CC_2012_13(data):
    """
    Randomly generate a ASN_CC, trying to replicate the real ASN_CC of the
    dataset. Fraudsters usually use VPN and/or proxy, so we can't really
    generate them better than this.

    :param data: The dataframe of all transactions, year 2012_13
    :return: A generated CC_ASN
    """

    # We try to replicate the distribution of real ASN_CC (from the 2013 set)
    n = 30
    cc_asn_list = data["CC_ASN"].value_counts()[:n].index.tolist()
    cc_asn_list.remove("n./d.,n./d.")  # remove this undisclosed value
    cc_asn_choice = random.randint(1, 100)
    if cc_asn_choice <= 42:  # 248703/584198
        cc_asn = cc_asn_list[0]
    elif cc_asn_choice <= 56:  # 82696/584198
        cc_asn = cc_asn_list[1]
    elif cc_asn_choice <= 74:  # 52419/584198 & 50558/584198
        cc_asn = random.choice(cc_asn_list[2:4])
    elif cc_asn_choice <= 82:  # 12419, 11631, 11422, 10010
        cc_asn = random.choice(cc_asn_list[4:8])
    else:
        cc_asn = random.choice(cc_asn_list[8:30])
    return cc_asn


def generate_ASN_CC_2014_15(data):
    """
    Randomly generate a ASN_CC, trying to replicate the real ASN_CC of the
    dataset. Fraudsters usually use VPN and/or proxy, so we can't really
    generate them better than this.

    :param data: The dataframe of all transactions, year 2014_15
    :return: A generated CC_ASN
    """
    n = 30
    cc_asn_list = data["CC_ASN"].value_counts()[:n].index.tolist()
    cc_asn_list.remove("n./d.,n./d.")  # remove this undisclosed value
    # 482909
    cc_asn_choice = random.randint(1, 100)
    if cc_asn_choice <= 42:  # 202199/482909
        cc_asn = cc_asn_list[0]
    elif cc_asn_choice <= 56:  # 70159/482909
        cc_asn = cc_asn_list[1]
    elif cc_asn_choice <= 67:  # 53428/482909
        cc_asn = cc_asn_list[2]
    elif cc_asn_choice <= 75:  # raw_df 39352/482909
        cc_asn = cc_asn_list[3]
    elif cc_asn_choice <= 81:  # 29251/482909
        cc_asn = random.choice(cc_asn_list[4:7])
    elif cc_asn_choice <= 85:
        cc_asn = random.choice(cc_asn_list[7:10])  # 20460/482909
    else:
        cc_asn = random.choice(cc_asn_list[10:30])
    return cc_asn


def generate_random_SessionID():
    """
    Generates a random SessionID

    :return: an hashed random SessionID
    """
    return generate_random_hashed_string(16)


def generate_random_TransactionID():
    """
    Generates a random TransactionID

    :return: an hashed random TransactionID
    """
    return generate_random_hashed_string(16)
    

def generate_num_conferma():
    """
    Randomly generates the field "confirm_SMS", using the real data
    distribution.
    Can be used for both 2012_13 and 2014_15, as they have a very similar
    distribution for this field.

    :return: 'No' or 'Yes'
    """
    # Si: 499264, No: 84934 (85,5%)
    if random.randint(1, 100) > 85:
        num_conferma = "No"
    else:
        num_conferma = "Si"
    return num_conferma


def generate_random_starting_ts(user_trans, db_finish_time):
    """
    Generates a random timestamp for the start of an attack against an User

    :param user_trans: A dataframe of user transactions
    :param db_finish_time: the finishing time of the transaction dataset
    :return: A random timestamp between the starting and the ending of
    transactions
    """
    td_hours = (db_finish_time - user_trans["Timestamp"].min()).days * 24
    return round_dt_to_second(
        user_trans.Timestamp.min()
        + timedelta(hours=round(random.uniform(0, td_hours), 2))
    )


def generate_ts_hour(starting_ts):
    """
    Replaces the hours of a timestamp with a random one, with probability
    distribution.
    approximating the one of a real fraud (Stripe report, 2016)

    :param starting_ts: A timestamp
    :return: The input timestamp with the hour modified
    """
    int_choice = random.randint(0, 100)
    if int_choice <= 25:  # 1 or 2 A.M.
        hour = random.randint(1, 2)
    elif int_choice <= 45:
        hour = random.choice([0, 3, 22, 23])
    elif int_choice <= 80:
        hour = random.choice([4, 11, 12, 13, 14, 15, 18, 19, 20, 21])
    else:
        hour = random.choice([5, 6, 7, 8, 9, 10, 16, 17])
    return starting_ts.replace(hour=hour)


def generate_ts_hour_working_hour(starting_ts):
    """
    Replaces the hours of a timestamp with a random one, with probability
    distribution approximating the one of a real transaction
    (Stripe report, 2016)

    :param starting_ts: A timestamp
    :return: The input timestamp with the hour modified
    """
    int_choice = random.randint(0, 100)
    if int_choice <= 70:
        hour = random.randint(8, 19)  # between 8 and 19
    elif int_choice <= 85:
        hour = random.choice([6, 7, 20, 21, 22])
    else:
        hour = random.choice([0, 1, 2, 3, 4, 5, 23])
    return starting_ts.replace(hour=hour)

