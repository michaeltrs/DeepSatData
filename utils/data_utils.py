import re


def find_number(text, c, single=True):
    val = re.findall(r'%s(\d+)' % c, text)
    if single:
        val = val[0]
    return val