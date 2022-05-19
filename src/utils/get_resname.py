
def get_resname(res):

    if res == 0.01:
        resname = 'res001'
    elif res == 0.05:
        resname = 'res005'
    elif res == 0.10:
        resname = 'res010'
    elif res == 0.25:
        resname = 'res025'
    elif res == 1:
        resname = 'res100'
    else:
        raise ValueError('Undefined resolution, must be 0.01/0.05/0.25/1')

    return resname
