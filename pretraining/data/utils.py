import re

"""
Need to clean deid for both III and IV
"""

# def clean_note(x):
#     y = re.sub(r'\[\*\*(.*?)\*\*\]', '', x)
#     y = re.sub('[0-9]+\.', '', y) 
#     y = re.sub('--|___|__|==', '', y) 
#     y = re.sub('\n', ' ', y)
#     y = y.strip()
#     return y

def _clean_sent(sent):
    return sent.strip().replace('\n', ' ')

def _clean_deid(text):
    text = re.sub(r'\[\*\*(.*?)\*\*\]', _clean_matched, text)
    text = re.sub('--|___|__|==', '', text)
    return text 

def _clean_matched(matched):
    """
    applied to re.sub to further clean phi placeholders
    
    e.g.: 
        [**Last Name (NamePattern4) 1604**] --> Last Name
        [**MD Number(1) 1605**] --> MD Number
        [**2101-11-5**] --> 2101-11-5
    
    """
    phi = matched.group(1)
    phi = phi.strip()
    if phi == '':
        return phi.strip()
    
    # remove final id
    if ' ' in phi:
        pl = phi.split()
        if pl[-1].isnumeric():
            phi = ' '.join(pl[:-1])
        else:
            phi = ' '.join(pl)
    
    # remove (Name Pattern) etc
    phi = re.sub(r'\(.*?\)', '', phi)
    phi = phi.strip()
    return phi
