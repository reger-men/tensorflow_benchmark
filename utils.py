import sys



def print_msg(msg, typ=None, onLine=False):
    """Print msg in specific format by it type"""
    
    end_ = '\n'
    TXTEND = '\033[0m'
    TXTBOLD = '\033[1m'
    if typ == 'info':
        msg = TXTBOLD + '\033[95m' + msg + TXTEND
    elif typ == 'succ':
        msg = TXTBOLD + '\033[92m' + msg + TXTEND
    elif typ == 'warn':
        msg = TXTBOLD + '\033[93m' + msg + TXTEND
    elif typ == 'err':
        msg = TXTBOLD + '\033[91m' + msg + TXTEND
    else:
        msg = msg
    
    if onLine:
        end_ = '\r'

    print(msg, end=end_)

