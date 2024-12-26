import re

def create_pseudo(word):
    if re.compile("[0-9][0-9]").fullmatch(word):
        return "twoDigitYear"
    if re.compile("[0-2][0-9][0-9][0-9]").fullmatch(word):
        return "fourDigitYear"
    if re.compile("[0-3]?[0-9]\/[1-2]?[0-9]").fullmatch(word):
        return "dateShort"
    if re.compile("[0-3]?[0-9]\/[1-2]?[0-9]\/[0-9][0-9]").fullmatch(word):
        return "dateLong"
    if re.compile("[0-9]+\,?[0-9]*\.?[0-9]+").fullmatch(word):
        return "MonetaryAmount"
    if re.compile("[0-9]?[0-9]\.[0-9]+").fullmatch(word):
        return "MonetaryAmountOrPercentage"
    if re.compile("([0-9]+\.)+[0-9]+").match(word):
        return "versionCode"
    if re.compile("\$[0-9]+(\,[0-9]+)*\.?[0-9]*").match(word):
        return "price"
    if re.compile("\$[012][0-9]?[0-9][0-9]").match(word):
        return "time"
    if re.compile("[0-9]+(\,[0-9]+)*\.?[0-9]*").match(word):
        return "otherNumber"
    if re.compile("[A-Z]+$").fullmatch(word):
        return "allCaps"
    if re.compile("[A-Z]\.").fullmatch(word):
        return "capPeriod"
    if re.compile("[A-Za-z]+\.").fullmatch(word):
        return "shortForm"
    if re.compile("([A-Za-z]\.)+[A-Za-z]\.?").match(word):
        return "initials"
    if re.compile(".*\$.*").match(word):
        return "containsDollarSign"
    if re.compile(".*'s|.*'").fullmatch(word):
        return "belongTo"
    if re.compile(".*'d|.*'ll|.*'t|.*'re|.*ve").match(word):
        return "shortWriting"
    if re.compile("([A-Za-z0-9]+\-)+[A-Za-z0-9]+").match(word):
        return "phrase"

def pseudo_set(train_set):
    pseudo = {}

    freq = {}

    for sentence in train_set:
        for word, tag in sentence:
            freq[word] = freq.get(word, 0) + 1
    
    threshold = 4

    for sentence in train_set:
        for word, tag in sentence:
            if freq[word] < threshold:
                pseudo_word = create_pseudo(word)
                pseudo[word] = pseudo_word
            else:
                pseudo[word] = word
    return pseudo