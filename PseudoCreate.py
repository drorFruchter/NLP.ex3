import re

def create_pseudo(word):
    if re.compile("[0-9][0-9]").fullmatch(word):
        return "shortYear"
    if re.compile("[0-2][0-9][0-9][0-9]").fullmatch(word):
        return "longYear"
    if re.compile("[0-3]?[0-9]\/[1-2]?[0-9]").fullmatch(word):
        return "shortDate"
    if re.compile("[0-3]?[0-9]\/[1-2]?[0-9]\/[0-9][0-9]").fullmatch(word):
        return "longDate"
    if re.compile("[0-9]+\,?[0-9]*\.?[0-9]+").fullmatch(word):
        return "amount"
    if re.compile("[0-9]?[0-9]\.[0-9]+").fullmatch(word):
        return "amountOrPrecent"
    if re.compile("([0-9]+\.)+[0-9]+").match(word):
        return "version"
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
    return "rare"

def pseudo_set(train_set):
    pseudo = {}

    freq = {}

    for sentence in train_set:
        for word, tag in sentence:
            freq[word] = freq.get(word, 0) + 1
    
    threshold = 3
    for sentence in train_set:
        for word, tag in sentence:
            if freq[word] < threshold:
                pseudo[word] =  create_pseudo(word)
            else:
                pseudo[word] = word
    return pseudo