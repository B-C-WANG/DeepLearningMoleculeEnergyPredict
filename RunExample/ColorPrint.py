
def cp(strPrint, front_color="",back_color=""):

    fd = {
        "BLACK":30,
        'RED' : 31,
    'GREEN' : 32,
    'BROWN' : 33,
    'BLUE' : 34,
    'PURPLE' : 35,
    'CYAN' : 36,
    'WHITE' : 37,
    'UNDERLINE_ON'  : 38,
    'UNDERLINE_OFF' : 39,
    "":""}

    bd = {
        "BLACK":40,
        'RED': 41,
        'GREEN': 42,
        'BROWN': 43,
        'BLUE': 44,
        'PURPLE': 45,
        'CYAN': 46,
        'WHITE': 47,
        "":""

    }

    strPrefix = "\033[0;%s;%sm" % (fd[str.upper(front_color)], bd[str.upper(back_color)])

    strSuffix = "\033[0m"
    strMsg = strPrefix + strPrint + strSuffix

    return strMsg


