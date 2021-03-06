import argparse
import numpy as np
import pandas as pd
import time
import copy
from sklearn.model_selection import KFold

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y, words=[]):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        self.w = np.zeros(len(xFeat[0, 1:]))

        for epoch in range(self.mEpoch):
            wrong = 0
            y_pred = self.predict(xFeat)
            for i in range(len(y_pred)):
                if y_pred[i] != y[i, 1]:
                    wrong += 1
                    if y[i, 1] > 0:
                        self.w += xFeat[i, 1:]
                    else:
                        self.w -= xFeat[i, 1:]
            
            if wrong == 0:
                return stats

            stats[epoch] = wrong

        temp = copy.copy(self.w)
        largest = np.zeros(15, int)
        for i in range(15):
            idx = np.argmax(temp)
            largest[i]=idx
            temp[idx]=0

        temp = copy.copy(self.w)
        smallest = np.zeros(15, int)
        for i in range(15):
            idx = np.argmax(temp)
            smallest[i]=idx
            temp[idx]=0

        i = 0
        positive = []
        for word in words:
            if i in largest:
                positive.append(word)
            i += 1

        i = 0
        negative = []
        for word in words:
            if i in smallest:
                negative.append(word)
            i += 1

        if words != []:
            print("Positive:", positive)
            print("Negative:", negative)

        return stats

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        yHat = []
        for i in range(len(xFeat)):
            if np.sum(np.matmul(self.w, xFeat[i, 1:])) >= 0:
                yHat.append(1)
            else:
                yHat.append(0)
        return yHat


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """

    wrong = 0
    for i in range(len(yHat)):
        if yHat[i] != yTrue[i, 1]:
            wrong += 1

    return wrong


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    if True:
        words = {'the': 4171, 'to': 4156, 'number': 4055, 'a': 3874, 'and': 3764, 'httpaddr': 3603, 'of': 3560, 'is': 3422, 'in': 3395, 'for': 3385, 'on': 3358, 'thi': 3127, 'it': 3062, 'you': 3018, 'that': 2958, 'be': 2717, 'i': 2685, 's': 2629, 'from': 2576, 'with': 2555, 'have': 2457, 'list': 2434, 'us': 2337, 'not': 2326, 'emailaddr': 2325, 'or': 2297, 'ar': 2279, 't': 2239, 'at': 2213, 'if': 2205, 'your': 2171, 'mail': 2148, 'by': 2120, 'can': 2028, 'as': 1999, 'all': 1982, 'but': 1826, 'an': 1767, 'get': 1747, 'do': 1700, 'we': 1634, 'will': 1623, 'more': 1575, 'email': 1545, 'so': 1542, 'no': 1525, 'just': 1511, 'here': 1510, 'out': 1506, 'time': 1450, 'new': 1427, 'up': 1414, 'there': 1398, 'my': 1391, 'ha': 1363, 'now': 1357, 'like': 1355, 'wa': 1330, 'ani': 1324, 'what': 1317, 'about': 1316, 'inform': 1301, 'onli': 1289, 'our': 1284, 'messag': 1277, 'thei': 1265, 'user': 1195, 'don': 1193, 'work': 1186, 'would': 1177, 'other': 1174, 'free': 1150, 'some': 1124, 'make': 1114, 'wrote': 1113, 'which': 1110, 'pleas': 1094, 'me': 1055, 'm': 1054, 'when': 1049, 'than': 1042, 'how': 1041, 'been': 1026, 'want': 1016, 'need': 1015, 'click': 1014, 'remov': 1002, 'go': 992, 'e': 972, 'then': 961, 'who': 953, 'receiv': 930, 'group': 922, 're': 912, 'also': 911, 'know': 908, 'see': 907, 'wai': 896, 'their': 894, 'look': 893, 'into': 883, 'date': 874, 'linux': 837, 'becaus': 836, 'them': 830, 'peopl': 812, 'over': 812, 'right': 808, 'think': 803, 'mai': 799, 'address': 791, 'net': 791, 'even': 783, 'send': 776, 've': 775, 'year': 773, 'take': 773, 'doe': 766, 'system': 753, 'should': 750, 'dai': 750, 'these': 749, 'help': 743, 'subject': 743, 'find': 741, 'type': 736, 'most': 732, 'could': 731, 'includ': 731, 'd': 718, 'call': 715, 'well': 714, 'much': 710, 'chang': 709, 'first': 702, 'good': 696, 'problem': 692, 'subscript': 690, 'dollarnumb': 687, 'sai': 679, 'name': 664, 'world': 664, 'll': 661, 'servic': 655, 'sponsor': 654, 'thing': 653, 'same': 652, 'busi': 651, 'line': 648, 'text': 641, 'hi': 641, 'file': 640, 'nbsp': 639, 'give': 637, 'content': 636, 'run': 633, 'thank': 633, 'todai': 633, 'had': 631, 'veri': 625, 'best': 621, 'maintain': 619, 'back': 618, 'offer': 617, 'home': 614, 'compani': 607, 'where': 604, 'url': 603, 'still': 602, 'come': 594, 'check': 594, 'link': 590, 'start': 589, 'un': 585, 'try': 578, 'am': 576, 'mani': 576, 'were': 574, 'those': 572, 'after': 572, 'set': 564, 'part': 561, 'phone': 558, 'web': 558, 'through': 558, 'site': 553, 'own': 552, 'irish': 551, 'end': 551, 'sent': 549, 'interest': 548, 'sf': 546, 'befor': 544, 'internet': 543, 'com': 540, 'unsubscrib': 538, 'while': 534, 'state': 533, 'product': 532, 'read': 528, 'sinc': 525, 'provid': 525, 'person': 525, 'why': 524, 'softwar': 524, 'without': 521, 'follow': 520, 'avail': 516, 'form': 515, 'quot': 513, 'last': 512, 'everi': 510, 'off': 510, 'realli': 507, 'such': 501, 'write': 499, 'someth': 499, 'never': 497, 'version': 492, 'two': 492, 'old': 491, 'anyon': 489, 'seem': 484, 'too': 482, 'monei': 481, 'sure': 480, 'differ': 480, 'below': 479, 'he': 477, 'few': 476, 'current': 475, 'futur': 475, 'anoth': 474, 'order': 473, 'put': 472, 'gener': 472, 'comput': 471, 'market': 470, 'transfer': 469, 'commun': 469, 'actual': 466, 'program': 463, 'point': 460, 'month': 453, 'onc': 453, 'let': 453, 'real': 453, 'keep': 452, 'down': 447, 'said': 447, 'c': 447, 'base': 446, 'origin': 446, 'report': 445, 'manag': 444, 'wish': 443, 'better': 443, 'doesn': 442, 'code': 441, 'question': 440, 'price': 438, 'requir': 438, 'found': 437, 'again': 436, 'spam': 433, 'network': 432, 'ad': 431, 'support': 430, 'did': 429, 'got': 428, 'spamassassin': 427, 'repli': 425, 'easi': 425, 'onlin': 424, 'page': 421, 'great': 420, 'long': 418, 'contact': 418, 'charset': 417, 'made': 416, 'build': 415, 'week': 414, 'plain': 411, 'lot': 408, 'show': 407, 'visit': 407, 'talk': 406, 'place': 404, 'might': 403, 'result': 400, 'rate': 400, 'encod': 398, 'mean': 398, 'html': 397, 'instal': 396, 'open': 396, 'special': 395, 'each': 395, 'next': 393, 'life': 392, 'complet': 392, 'hour': 390, 'high': 389, 'allow': 387, 'server': 381, 'r': 381, 'secur': 381, 'case': 380, 'post': 380, 'window': 380, 'add': 378, 'full': 377, 'custom': 377, 'applic': 377, 'p': 376, 'reason': 376, 'both': 373, 'howev': 371, 'save': 370, 'less': 370, 'someon': 370, 'around': 368, 'welcom': 367, 'tell': 364, 'issu': 364, 'per': 362, 'cost': 362, 'million': 360, 'ever': 357, 'live': 356, 'possibl': 356, 'top': 355, 'idea': 353, 'probabl': 353, 'packag': 352, 'within': 352, 'stop': 352, 'regard': 350, 'ask': 350, 'ye': 349, 'format': 348, 'aug': 348, 'releas': 347, 'error': 347, 'must': 346, 'test': 345, 'sourc': 344, 'rpm': 343, 'access': 342, 'o': 338, 'alwai': 338, 'geek': 337, 'bui': 337, 'search': 337, 'pai': 337, 'though': 335, 'els': 332, 'creat': 332, 'fact': 330, 'anyth': 329, 'simpli': 329, 'power': 328, 'word': 326, 'note': 325, 'alreadi': 322, 'heaven': 322, 'etc': 321, 'bit': 320, 'second': 320, 'featur': 319, 'turn': 319, 'process': 318, 'under': 317, 'subscrib': 315, 'low': 314, 'sign': 313, 'x': 312, 'develop': 312, 'data': 311, 'exampl': 310, 'request': 310, 'card': 310, 'suppli': 308, 'instead': 307, 'box': 306, 'enough': 306, 'cours': 305, 'sell': 304, 'abl': 304, 'experi': 304, 'feel': 303, 'copi': 303, 'dollar': 301, 'littl': 300, 'simpl': 299, 'begin': 299, 'least': 298, 'copyright': 298, 'job': 298, 'n': 296, 'happen': 296, 'credit': 295, 'account': 295, 'believ': 295, 'share': 294, 'thinkgeek': 293, 'friend': 292, 'updat': 291, 'between': 291, 'websit': 291, 'plan': 289, 'detail': 289, 'learn': 288, 'abov': 287, 'done': 286, 'profession': 286, 'recent': 285, 'answer': 285, 'design': 284, 'import': 284, 'inc': 282, 'u': 282, 'notic': 282, 'noth': 281, 'mayb': 281, 'advertis': 281, 'download': 281, 'fill': 281, 'past': 280, 'numbertnumb': 279, 'countri': 279, 'minut': 278, 'either': 278, 'pm': 278, 'larg': 276, 'yet': 275, 'guarante': 275, 'further': 274, 'public': 274, 'tool': 274, 'isn': 272, 'effect': 272, 'non': 272, 'sever': 272, 'hard': 271, 'info': 271, 'option': 271, 'understand': 270, 'contain': 269, 'protect': 269, 'didn': 268, 'august': 268, 'discuss': 266, 'increas': 266, 'iso': 266, 'reserv': 265, 'comment': 265, 'deal': 265, 'technolog': 265, 'everyth': 264, 'mime': 263, 'until': 263, 'improv': 262, 'limit': 261, 'thought': 261, 'three': 261, 'client': 261, 'stuff': 260, 'local': 259, 'seen': 259, 'forward': 259, 'caus': 259, 'hope': 258, 'sep': 258, 'thu': 257, 'exist': 257, 'valu': 257, 'hand': 257, 'multi': 256, 'g': 256, 'control': 255, 'big': 254, 'small': 254, 'legal': 254, 'accept': 253, 'consid': 253, 'purchas': 253, 'b': 252, 'term': 251, 'respons': 250, 'www': 250, 'against': 250, 'center': 249, 'engin': 249, 'becom': 249, 'specif': 249, 'far': 249, 'return': 249, 'fix': 249, 'quit': 245, 'individu': 245, 'drive': 244, 'opportun': 244, 'l': 244, 'famili': 243, 'true': 243, 'rather': 243, 'yourself': 243, 'suggest': 242, 'level': 242, 'oper': 241, 'via': 240, 'won': 239, 'awai': 239, 'join': 238, 'enter': 238, 'wait': 238, 'move': 237, 'cd': 237, 'plai': 236, 'sale': 236, 'major': 234, 'select': 234, 'bill': 234, 'tri': 232, 'connect': 232, 'offic': 232, 'newslett': 232, 'wonder': 231, 'ago': 231, 'law': 230, 'immedi': 230, 'record': 229, 'prefer': 229, 'view': 228, 'fax': 228, 'kind': 227, 'financi': 227, 'nextpart': 227, 'grow': 227, 'sort': 224, 'thousand': 224, 'size': 224, 'direct': 224, 'wrong': 224, 'act': 221, 'area': 221, 'easili': 221, 'man': 220, 'appear': 220, 'govern': 220, 'expect': 219, 'dear': 219, 'matter': 219, 'qualiti': 219, 'stori': 218, 'singl': 218, 'ship': 218, 'appli': 218, 'hit': 218, 'addit': 218, 'regist': 217, 'close': 217, 'citi': 217, 'soon': 216, 'nice': 216, 'log': 215, 'miss': 215, 'unit': 215, 'lead': 214, 'fast': 214, 'hundr': 214, 'w': 214, 'love': 213, 'juli': 213, 'continu': 212, 'almost': 212, 'book': 212, 'pc': 211, 'databas': 211, 'bad': 210, 'standard': 210, 'whether': 210, 'razor': 209, 'success': 209, 'member': 209, 'parti': 209, 'total': 209, 'intern': 209, 'posit': 209, 'wed': 208, 'industri': 208, 'relat': 208, 'numberpm': 207, 'present': 207, 'kei': 207, 'plu': 206, 'john': 205, 'later': 205, 'printabl': 205, 'sound': 205, 'invest': 205, 'signatur': 204, 'final': 204, 'pretti': 204, 'yahoo': 203, 'instruct': 203, 'refer': 202, 'activ': 202, 'h': 202, 'whole': 202, 'charg': 201, 'entir': 201, 'machin': 201, 'mind': 200, 'trade': 200, 'distribut': 200, 'dure': 200, 'septemb': 200, 'vnumber': 200, 'risk': 199, 'ag': 199, 'nation': 199, 'mark': 198, 'decid': 198, 'id': 198, 'corpor': 198, 'often': 197, 'daili': 197, 'rule': 196, 'digit': 196, 'ok': 194, 'amount': 194, 'cannot': 194, 'similar': 194, 'pick': 193, 'worth': 193, 'directori': 193, 'f': 193, 'happi': 193, 'opt': 193, 'cell': 193, 'perform': 192, 'latest': 191, 'altern': 191, 'collect': 191, 'upon': 191, 'rememb': 190, 'claim': 190, 'everyon': 189, 'mention': 189, 'leav': 188, 'review': 188, 'bodi': 188, 'author': 186, 'univers': 185, 'guess': 185, 'load': 185, 'research': 185, 'known': 184, 'automat': 184, 'red': 184, 'jul': 183, 'document': 183, 'benefit': 182, 'project': 182, 'handl': 181, 'track': 180, 'watch': 179, 'gui': 179, 'incom': 178, 'platform': 178, 'quick': 177, 'exmh': 177, 'figur': 177, 'short': 177, 'suit': 176, 'anywai': 176, 'amp': 176, 'exactli': 175, 'cash': 175, 'natur': 174, 'absolut': 174, 'clean': 174, 'promot': 174, 'reach': 174, 'haven': 173, 'media': 173, 'meet': 173, 'discov': 173, 'given': 173, 'print': 172, 'left': 172, 'tech': 172, 'her': 172, 'recommend': 172, 'involv': 172, 'privat': 172, 'potenti': 172, 'american': 171, 'cheer': 171, 'polici': 171, 'tire': 171, 'solut': 171, 'fail': 170, 'oblig': 170, 'mon': 169, 'pass': 169, 'hold': 169, 'edit': 169, 'target': 169, 'along': 169, 'forc': 169, 'co': 168, 'perhap': 168, 'method': 167, 'function': 167, 'configur': 166, 'tue': 166, 'memori': 166, 'microsoft': 166, 'j': 166, 'event': 165, 'fine': 165, 'head': 165, 'street': 165, 'care': 165, 'earn': 164, 'bring': 164, 'usual': 163, 'longer': 163, 'delet': 163, 'choos': 163, 'cut': 163, 'electron': 162, 'half': 162, 'numberth': 162, 'numberbit': 161, 'partner': 161, 'profit': 161, 'except': 161, 'produc': 161, 'matthia': 160, 'hous': 160, 'sight': 160, 'choic': 160, 'upgrad': 159, 'late': 159, 'publish': 159, 'store': 159, 'organ': 159, 'lose': 159, 'side': 158, 'expens': 158, 'concern': 158, 'game': 158, 'due': 158, 'fall': 158, 'deliv': 158, 'submit': 157, 'readi': 157, 'imag': 157, 'basic': 156, 'hear': 156, 'color': 156, 'execut': 155, 'secret': 155, 'abil': 155, 'although': 155, 'third': 155, 'respect': 155, 'step': 154, 'interact': 154, 'fri': 154, 'normal': 154, 'ascii': 153, 'hardwar': 153, 'insur': 153, 'main': 153, 'perl': 153, 'clear': 153, 'bug': 153, 'oh': 152, 'paid': 152, 'sun': 151, 'win': 151, 'depend': 151, 'trust': 151, 'speed': 150, 'train': 150, 'action': 149, 'situat': 149, 'unless': 149, 'enabl': 149, 'practic': 148, 'script': 148, 'stock': 148, 'video': 148, 'human': 147, 'bank': 147, 'space': 147, 'assist': 147, 'replac': 146, 'popular': 146, 'agre': 146, 'whatev': 146, 'approv': 146, 'particular': 145, 'polit': 145, 'goe': 145, 'usa': 145, 'disk': 145, 'war': 145, 'effort': 145, 'pgp': 145, 'came': 144, 'him': 144, 'coupl': 144, 'agent': 144, 'five': 144, 'advantag': 144, 'none': 144, 'chri': 143, 'self': 143, 'face': 143, 'mondai': 143, 'fork': 143, 'america': 142, 'combin': 142, 'assum': 142, 'indic': 141, 'extra': 141, 'associ': 141, 'section': 141, 'gari': 140, 'purpos': 140, 'she': 140, 'locat': 140, 'articl': 140, 'taken': 140, 'directli': 139, 'displai': 139, 'remain': 139, 'devic': 139, 'util': 139, 'loss': 139, 'worker': 139, 'v': 138, 'break': 138, 'announc': 138, 'drop': 138, 'browser': 138, 'huge': 138, 'root': 138, 'wide': 137, 'mr': 137, 'lawrenc': 137, 'field': 137, 'os': 137, 'default': 136, 'cover': 136, 'letter': 136, 'folk': 136, 'heard': 136, 'kernel': 135, 'rest': 135, 'lower': 135, 'spend': 135, 'four': 135, 'sometim': 134, 'togeth': 134, 'setup': 134, 'itself': 134, 'class': 134, 'numberd': 134, 'variou': 134, 'host': 134, 'fastest': 134, 'worri': 133, 'went': 133, 'hat': 133, 'built': 133, 'sorri': 132, 'chanc': 132, 'accord': 132, 'certain': 132, 'screen': 131, 'explain': 130, 'de': 130, 'intend': 130, 'music': 130, 'advanc': 130, 'especi': 129, 'dvd': 129, 'lost': 129, 'privaci': 129, 'folder': 129, 'payment': 129, 'resourc': 129, 'school': 129, 'fee': 129, 'period': 129, 'propos': 129, 'header': 129, 'retail': 129, 'deathtospamdeathtospamdeathtospam': 128, 'dream': 127, 'social': 127, 'told': 127, 'command': 127, 'multipl': 127, 'global': 127, 'team': 127, 'debt': 127, 'numberam': 126, 'across': 126, 'independ': 126, 'averag': 126, 'licens': 126, 'contract': 126, 'consult': 126, 'shop': 126, 'y': 125, 'consum': 125, 'avoid': 125, 'speak': 125, 'osdn': 125, 'fund': 125, 'compil': 124, 'redhat': 124, 'gnu': 124, 'david': 124, 'night': 123, 'sens': 123, 'fun': 123, 'imagin': 123, 'common': 123, 'feder': 123, 'confirm': 123, 'easier': 122, 'behalf': 122, 'sender': 122, 'express': 122, 'insid': 121, 'murphi': 121, 'wouldn': 121, 'opinion': 121, 'anywher': 121, 'model': 121, 'quickli': 121, 'safe': 121, 'apt': 120, 'block': 120, 'port': 120, 'certainli': 120, 'correct': 120, 'tax': 120, 'ed': 119, 'extrem': 119, 'histori': 119, 'warn': 119, 'took': 119, 'lowest': 119, 'valid': 119, 'press': 119, 'rang': 119, 'integr': 119, 'domain': 119, 'attach': 118, 'saou': 118, 'confidenti': 118, 'apolog': 118, 'piec': 118, 'hello': 118, 'six': 118, 'im': 118, 'commerci': 118, 'owner': 117, 'compar': 117, 'attempt': 116, 'style': 116, 'enjoi': 116, 'car': 116, 'mortgag': 116, 'women': 116, 'filter': 116, 'knowledg': 116, 'seriou': 115, 'board': 115, 'editor': 115, 'offici': 115, 'tuesdai': 115, 'troubl': 114, 'languag': 114, 'mass': 114, 'burn': 114, 'match': 113, 'statement': 113, 'promis': 113, 'moment': 113, 'implement': 113, 'modifi': 113, 'demand': 113, 'decis': 113, 'near': 112, 'cool': 112, 'pictur': 112, 'numbera': 112, 'technic': 112, 'k': 111, 'org': 111, 'pre': 111, 'driver': 111, 'studi': 111, 'iii': 111, 'ignor': 111, 'qualifi': 111, 'robert': 111, 'suppos': 111, 'sincer': 111, 'movi': 110, 'men': 110, 'bottom': 110, 'billion': 110, 'administr': 110, 'aren': 109, 'dave': 109, 'gain': 109, 'basenumb': 109, 'ca': 109, 'definit': 109, 'necessari': 109, 'transact': 109, 'excel': 109, 'item': 109, 'stand': 109, 'desir': 108, 'written': 108, 'wasn': 108, 'object': 108, 'appar': 108, 'gnupg': 108, 'loan': 108, 'describ': 108, 'pro': 108, 'respond': 108, 'introduc': 108, 'fulli': 108, 'amaz': 107, 'initi': 107, 'separ': 107, 'count': 107, 'patch': 106, 'stai': 106, 'strong': 106, 'myself': 106, 'prepar': 106, 'cc': 106, 'materi': 106, 'deserv': 105, 'perfect': 105, 'educ': 105, 'tag': 105, 'cv': 104, 'switch': 104, 'attack': 104, 'faster': 104, 'expert': 104, 'kill': 104, 'earli': 104, 'guid': 104, 'reduc': 104, 'appreci': 104, 'ie': 103, 'previous': 103, 'director': 103, 'paul': 103, 'fortun': 103, 'front': 103, 'central': 103, 'mobil': 103, 'anumb': 102, 'uniqu': 102, 'health': 102, 'button': 102, 'weight': 101, 'app': 101, 'manual': 101, 'capabl': 101, 'titl': 101, 'behind': 100, 'court': 100, 'deliveri': 100, 'kid': 99, 'hot': 99, 'player': 99, 'sa': 99, 'wireless': 99, 'thursdai': 99, 'properti': 99, 'tv': 99, 'toward': 99, 'presid': 99, 'seek': 99, 'disposit': 98, 'honor': 98, 'di': 98, 'held': 98, 'useless': 98, 'sold': 98, 'identifi': 97, 'fridai': 97, 'confer': 97, 'path': 97, 'tip': 97, 'among': 97, 'determin': 97, 'wednesdai': 97, 'approach': 97, 'lock': 96, 'obtain': 96, 'black': 96, 'proven': 96, 'congress': 96, 'themselv': 96, 'monthli': 96, 'york': 96, 'kevin': 96, 'outsid': 96, 'death': 95, 'prevent': 95, 'voic': 95, 'faq': 95, 'equip': 95, 'ms': 95, 'confus': 95, 'paper': 95, 'statu': 95, 'san': 94, 'vs': 94, 'fair': 94, 'fals': 94, 'adam': 94, 'predict': 94, 'launch': 94, 'regular': 94, 'slow': 94, 'energi': 94, 'resid': 94, 'brand': 93, 'fit': 93, 'interfac': 93, 'realiz': 93, 'forget': 93, 'photo': 93, 'extens': 93, 'michael': 92, 'verifi': 92, 'modul': 92, 'telephon': 92, 'virtual': 92, 'valuabl': 92, 'repres': 92, 'confid': 92, 'commit': 92, 'argument': 91, 'archiv': 91, 'tom': 91, 'econom': 91, 'refin': 91, 'brian': 91, 'score': 90, 'pack': 90, 'reader': 90, 'conveni': 90, 'mine': 90, 'enhanc': 90, 'otherwis': 90, 'firm': 90, 'travel': 90, 'mac': 90, 'desktop': 90, 'occur': 89, 'william': 89, 'solv': 89, 'awar': 89, 'gone': 89, 'higher': 89, 'art': 89, 'rel': 89, 'traffic': 89, 'core': 89, 'competit': 89, 'attent': 89, 'seri': 88, 'permiss': 88, 'sequenc': 88, 'ten': 88, 'entri': 87, 'surpris': 87, 'justin': 87, 'jabber': 87, 'difficult': 87, 'ey': 86, 'impress': 86, 'lib': 86, 'exclus': 86, 'unfortun': 86, 'condit': 86, 'ma': 86, 'appl': 86, 'stream': 86, 'usr': 85, 'ident': 85, 'catch': 85, 'evil': 85, 'et': 85, 'exchang': 85, 'foreign': 85, 'award': 84, 'carri': 84, 'ensur': 84, 'skip': 84, 'discount': 84, 'graphic': 84, 'bonu': 84, 'mailer': 84, 'signific': 84, 'children': 84, 'medic': 84, 'road': 84, 'manufactur': 84, 'alon': 84, 'depart': 84, 'programm': 83, 'pull': 83, 'yeah': 83, 'zip': 83, 'girl': 83, 'societi': 83, 'largest': 83, 'strategi': 83, 'elimin': 83, 'mode': 83, 'feedback': 83, 'compet': 83, 'defin': 82, 'beat': 82, 'employ': 82, 'obvious': 82, 'relev': 82, 'compat': 82, 'commiss': 82, 'techniqu': 82, 'popul': 82, 'investig': 82, 'capit': 82, 'unix': 82, 'numberk': 82, 'listen': 82, 'toll': 82, 'neg': 82, 'sex': 81, 'pocket': 81, 'thread': 81, 'dead': 81, 'anybodi': 81, 'danger': 81, 'pain': 81, 'establish': 81, 'afford': 81, 'poor': 81, 'excit': 81, 'environ': 81, 'south': 80, 'larger': 80, 'air': 80, 'monitor': 80, 'arriv': 80, 'numberb': 80, 'recipi': 80, 'q': 80, 'career': 80, 'bar': 79, 'spec': 79, 'steve': 79, 'reveal': 79, 'white': 79, 'colleg': 79, 'therefor': 79, 'physic': 79, 'msg': 79, 'numberp': 79, 'europ': 79, 'grant': 79, 'correspond': 79, 'couldn': 79, 'investor': 79, 'hp': 79, 'uk': 78, 'light': 78, 'instant': 78, 'achiev': 78, 'extend': 78, 'agenc': 78, 'anti': 77, 'sat': 77, 'fire': 77, 'jame': 77, 'spammer': 77, 'rais': 77, 'boot': 77, 'affili': 77, 'numbercnumb': 77, 'dn': 77, 'lack': 77, 'land': 77, 'progress': 77, 'forev': 77, 'concept': 77, 'wife': 77, 'sit': 77, 'rom': 77, 'roll': 77, 'doubl': 76, 'freedom': 76, 'canada': 76, 'xp': 76, 'serv': 76, 'sampl': 76, 'isp': 76, 'doubt': 76, 'googl': 76, 'whose': 76, 'chat': 76, 'radio': 76, 'acquir': 76, 'goal': 76, 'ip': 76, 'nearli': 76, 'wall': 76, 'tabl': 75, 'panel': 75, 'st': 75, 'innov': 75, 'shot': 75, 'rock': 75, 'cnet': 75, 'disabl': 75, 'auto': 75, 'earlier': 75, 'input': 75, 'attract': 75, 'morn': 75, 'particip': 75, 'delai': 75, 'trick': 75, 'descript': 75, 'spain': 74, 'hei': 74, 'inumb': 74, 'relationship': 74, 'brain': 74, 'octob': 74, 'california': 74, 'regul': 74, 'blank': 74, 'trial': 74, 'enterpris': 73, 'evid': 73, 'teledynam': 73, 'picasso': 73, 'room': 73, 'unlik': 73, 'greater': 73, 'inlin': 73, 'blog': 73, 'revenu': 73, 'leader': 73, 'remot': 73, 'treat': 72, 'ultim': 72, 'audio': 72, 'bulk': 72, 'scienc': 72, 'highli': 72, 'mostli': 72, 'incred': 72, 'previou': 72, 'al': 72, 'critic': 72, 'inde': 72, 'numberst': 72, 'feed': 72, 'walk': 72, 'challeng': 72, 'dev': 72, 'odd': 71, 'la': 71, 'unsolicit': 71, 'particularli': 71, 'cheap': 71, 'former': 71, 'yesterdai': 71, 'detect': 71, 'java': 71, 'sundai': 71, 'perfectli': 71, 'washington': 71, 'backup': 71, 'nobodi': 71, 'mpnumber': 71, 'proper': 71, 'output': 70, 'laptop': 70, 'eventu': 70, 'touch': 70, 'plug': 70, 'comparison': 70, 'basi': 70, 'perman': 70, 'suspect': 70, 'nt': 70, 'cach': 70, 'north': 70, 'saw': 70, 'worldwid': 70, 'declin': 70, 'suffer': 70, 'interview': 70, 'nor': 70, 'paragraph': 70, 'financ': 70, 'west': 70, 'peter': 69, 'multipart': 69, 'profil': 69, 'tim': 69, 'fight': 69, 'lender': 69, 'librari': 69, 'scan': 69, 'beberg': 69, 'jim': 69, 'favorit': 69, 'drug': 69, 'skeptic': 69, 'weekli': 69, 'se': 69, 'summer': 68, 'gave': 68, 'affect': 68, 'despit': 68, 'numbermb': 68, 'economi': 68, 'employe': 68, 'rich': 68, 'teach': 68, 'explor': 68, 'password': 68, 'abus': 67, 'contribut': 67, 'shape': 67, 'june': 67, 'motiv': 67, 'campaign': 67, 'vendor': 67, 'middl': 67, 'task': 67, 'effici': 67, 'male': 67, 'devel': 67, 'expir': 67, 'hettinga': 67, 'jump': 67, 'wast': 67, 'growth': 67, 'printer': 67, 'earth': 67, 'repositori': 66, 'camera': 66, 'advic': 66, 'prompt': 66, 'storag': 66, 'boundari': 66, 'truli': 66, 'reliabl': 66, 'rh': 66, 'sum': 66, 'instanc': 66, 'cultur': 66, 'discoveri': 66, 'joe': 66, 'mistak': 66, 'magazin': 65, 'hack': 65, 'region': 65, 'comfort': 65, 'vacat': 65, 'en': 65, 'wi': 65, 'classifi': 65, 'anim': 65, 'xml': 65, 'clearli': 65, 'finish': 65, 'menu': 65, 'assur': 65, 'ah': 65, 'somewher': 65, 'consolid': 65, 'entertain': 65, 'string': 65, 'shell': 65, 'fat': 65, 'tel': 65, 'biggest': 65, 'charact': 65, 'flag': 65, 'font': 64, 'stupid': 64, 'conserv': 64, 'truth': 64, 'boston': 64, 'suck': 64, 'god': 64, 'sexual': 64, 'father': 64, 'garrigu': 64, 'measur': 64, 'tree': 64, 'elsewher': 64, 'tune': 64, 'spot': 64, 'empir': 64, 'super': 64, 'anymor': 64, 'mix': 64, 'tradit': 64, 'artist': 63, 'convert': 63, 'french': 63, 'split': 63, 'btw': 63, 'master': 63, 'dont': 63, 'franc': 63, 'smart': 63, 'coverag': 63, 'cabl': 63, 'tomorrow': 63, 'ireland': 63, 'kept': 63, 'knew': 62, 'boi': 62, 'fresh': 62, 'config': 62, 'enumb': 62, 'instantli': 62, 'numberc': 62, 'convers': 62, 'typic': 62, 'procmail': 62, 'dell': 62, 'damag': 62, 'roman': 62, 'varieti': 62, 'unlimit': 62, 'worst': 62, 'bunch': 61, 'round': 61, 'schedul': 61, 'eat': 61, 'ran': 61, 'cent': 61, 'canon': 61, 'membership': 61, 'mother': 61, 'edg': 61, 'ground': 61, 'tx': 61, 'heart': 61, 'equal': 61, 'obviou': 61, 'boss': 61, 'emploi': 61, 'hire': 61, 'spent': 61, 'consider': 61, 'focu': 61, 'institut': 61, 'sport': 61, 'door': 61, 'seven': 61, 'english': 60, 'oct': 60, 'pop': 60, 'creativ': 60, 'hate': 60, 'headlin': 60, 'pudg': 60, 'tend': 60, 'dan': 60, 'correctli': 60, 'emerg': 60, 'termin': 60, 'ftp': 60, 'exact': 60, 'broker': 60, 'serious': 60, 'vast': 60, 'revers': 60, 'scheme': 60, 'ii': 60, 'background': 60, 'greg': 60, 'bigger': 59, 'staff': 59, 'construct': 59, 'exercis': 59, 'prospect': 59, 'zero': 59, 'zdnet': 59, 'factor': 59, 'star': 59, 'plenti': 59, 'usag': 59, 'gift': 59, 'blue': 59, 'specialist': 59, 'keyboard': 59, 'hide': 59, 'balanc': 59, 'attornei': 59, 'toni': 59, 'hair': 59, 'ps': 59, 'planta': 58, 'beyond': 58, 'slightli': 58, 'snumber': 58, 'autom': 58, 'asset': 58, 'brought': 58, 'categori': 58, 'xnumber': 58, 'heavi': 58, 'estim': 58, 'analysi': 58, 'expand': 58, 'maker': 58, 'substanti': 58, 'stick': 58, 'european': 58, 'prove': 58, 'experienc': 58, 'complianc': 58, 'properli': 58, 'procedur': 58, 'empti': 57, 'cnumber': 57, 'funni': 57, 'skill': 57, 'evalu': 57, 'ga': 57, 'sole': 57, 'realiti': 57, 'sir': 57, 'minimum': 57, 'li': 57, 'bnumber': 57, 'consist': 57, 'insert': 57, 'fairli': 57, 'parent': 57, 'sleep': 57, 'mous': 57, 'essenti': 57, 'edward': 57, 'dr': 57, 'theori': 57, 'hell': 57, 'label': 57, 'symbol': 57, 'push': 57, 'student': 57, 'ceo': 57, 'annoi': 56, 'alert': 56, 'woman': 56, 'intel': 56, 'wors': 56, 'bought': 56, 'fnumber': 56, 'senior': 56, 'reli': 56, 'visa': 56, 'modern': 56, 'mailbox': 56, 'portion': 56, 'ibm': 56, 'map': 56, 'fear': 56, 'adapt': 56, 'edificio': 55, 'nort': 55, 'barcelona': 55, 'succe': 55, 'statist': 55, 'chip': 55, 'src': 55, 'ilug': 55, 'specifi': 55, 'strongli': 55, 'cooper': 55, 'mike': 55, 'rare': 55, 'percentag': 55, 'imposs': 55, 'antiqu': 55, 'inconveni': 55, 'aol': 55, 'topic': 55, 'histor': 55, 'journal': 55, 'meant': 55, 'restrict': 55, 'annual': 55, 'dig': 55, 'met': 55, 'underwrit': 55, 'behavior': 55, 'kate': 55, 'invit': 55, 'outlook': 55, 'straight': 55, 'conf': 54, 'numberenumb': 54, 'invent': 54, 'friendli': 54, 'pablo': 54, 'disappear': 54, 'china': 54, 'adult': 54, 'numberx': 54, 'bother': 54, 'dozen': 54, 'mason': 54, 'polic': 54, 'analyst': 54, 'strang': 54, 'impact': 54, 'complex': 54, 'scale': 54, 'spread': 54, 'frequent': 54, 'prohibit': 54, 'civil': 54, 'arm': 54, 'flow': 54, 'neither': 54, 'mile': 54, 'judg': 53, 'march': 53, 'hall': 53, 'hang': 53, 'young': 53, 'advis': 53, 'calcul': 53, 'dog': 53, 'lo': 53, 'convinc': 53, 'msn': 53, 'th': 53, 'degre': 53, 'somehow': 53, 'catalog': 53, 'beta': 53, 'wealth': 53, 'context': 53, 'gmt': 53, 'appropri': 53, 'failur': 53, 'mechan': 53, 'numberanumb': 53, 'throughout': 53, 'index': 52, 'filenam': 52, 'fl': 52, 'luck': 52, 'random': 52, 'restor': 52, 'vari': 52, 'aid': 52, 'shock': 52, 'broken': 52, 'deposit': 52, 'somebodi': 52, 'van': 52, 'volum': 52, 'angl': 52, 'distanc': 52, 'minor': 52, 'commerc': 52, 'urgent': 52, 'structur': 52, 'mh': 52, 'intellig': 52, 'byte': 52, 'remind': 52, 'legitim': 52, 'ticket': 52, 'damn': 52, 'inch': 52, 'buyer': 52, 'hasn': 51, 'saturdai': 51, 'child': 51, 'winner': 51, 'crimin': 51, 'processor': 51, 'tape': 51, 'le': 51, 'gordon': 51, 'deep': 51, 'honest': 51, 'grab': 51, 'austin': 51, 'pattern': 51, 'equiti': 51, 'token': 51, 'virus': 51, 'decad': 51, 'centuri': 51, 'bear': 51, 'bush': 51, 'ventur': 51, 'water': 51, 'gnome': 51, 'shouldn': 51, 'numberm': 51, 'budget': 51, 'channel': 51, 'exclud': 51, 'argu': 51, 'partit': 50, 'highlight': 50, 'il': 50, 'pa': 50, 'role': 50, 'cat': 50, 'deni': 50, 'british': 50, 'shown': 50, 'cross': 50, 'twice': 50, 'blood': 50, 'illeg': 50, 'chief': 50, 'henc': 50, 'icq': 50, 'onto': 50, 'bandwidth': 50, 'es': 50, 'fi': 49, 'planet': 49, 'massiv': 49, 'viru': 49, 'mess': 49, 'binari': 49, 'ne': 49, 'ham': 49, 'numberdnumb': 49, 'extract': 49, 'protocol': 49, 'clue': 49, 'encrypt': 49, 'whitelist': 49, 'anytim': 49, 'militari': 49, 'hidden': 49, 'bell': 49, 'liter': 49, 'percent': 49, 'warranti': 49, 'complain': 49, 'visual': 49, 'highest': 49, 'satisfi': 49, 'aspect': 49, 'club': 49, 'queri': 49, 'doc': 49, 'premium': 49, 'bind': 49, 'januari': 49, 'recogn': 48, 'serial': 48, 'strength': 48, 'everybodi': 48, 'roger': 48, 'soni': 48, 'mount': 48, 'caught': 48, 'gnumber': 48, 'freshrpm': 48, 'homeown': 48, 'resolv': 48, 'east': 48, 'dump': 48, 'rise': 48, 'ill': 48, 'estat': 48, 'prior': 48, 'primari': 48, 'firewal': 48, 'gold': 48, 'babi': 48, 'buck': 48, 'unseen': 48, 'england': 48, 'strictli': 48, 'weekend': 48, 'draw': 48, 'resel': 48, 'famou': 48, 'hacker': 48, 'boost': 48, 'alan': 48, 'twenti': 48, 'repeat': 47, 'republ': 47, 'inbox': 47, 'pnumber': 47, 'duncan': 47, 'z': 47, 'doctor': 47, 'freebsd': 47, 'hmm': 47, 'foundat': 47, 'quarter': 47, 'summari': 47, 'feet': 47, 'proprietari': 47, 'compon': 47, 'admin': 47, 'citizen': 47, 'proof': 47, 'admit': 47, 'bottl': 47, 'richard': 47, 'strip': 47, 'carefulli': 47, 'beauti': 47, 'easiest': 47, 'muscl': 47, 'hunt': 47, 'modem': 47, 'encourag': 47, 'bin': 47, 'superior': 47, 'older': 46, 'film': 46, 'translat': 46, 'dynam': 46, 'minim': 46, 'solid': 46, 'hint': 46, 'classic': 46, 'ce': 46, 'bb': 46, 'null': 46, 'remark': 46, 'mere': 46, 'merchant': 46, 'geeg': 46, 'throw': 46, 'cancel': 46, 'img': 46, 'brother': 46, 'extern': 46, 'tremend': 46, 'numer': 46, 'park': 46, 'column': 46, 'equival': 46, 'swap': 46, 'te': 46, 'variabl': 46, 'dice': 46, 'aim': 46, 'rob': 45, 'farquhar': 45, 'beach': 45, 'length': 45, 'favor': 45, 'presum': 45, 'doer': 45, 'suse': 45, 'retain': 45, 'accur': 45, 'adopt': 45, 'phrase': 45, 'hole': 45, 'agreement': 45, 'prioriti': 45, 'flash': 45, 'bearer': 45, 'session': 45, 'solari': 45, 'cold': 45, 'defend': 45, 'fundament': 45, 'island': 45, 'repair': 45, 'food': 45, 'tm': 45, 'oppos': 45, 'hash': 45, 'var': 45, 'optim': 44, 'permit': 44, 'ram': 44, 'frustrat': 44, 'pioneer': 44, 'architectur': 44, 'usb': 44, 'blame': 44, 'vircio': 44, 'warm': 44, 'speech': 44, 'led': 44, 'margin': 44, 'dramat': 44, 'writer': 44, 'crash': 44, 'batteri': 44, 'surviv': 44, 'er': 44, 'patent': 44, 'transmiss': 44, 'maximum': 44, 'palm': 44, 'arrang': 44, 'april': 44, 'apart': 43, 'mirror': 43, 'martin': 43, 'explan': 43, 'trip': 43, 'duplic': 43, 'vehicl': 43, 'ac': 43, 'cycl': 43, 'smtp': 43, 'http': 43, 'laugh': 43, 'stage': 43, 'femal': 43, 'carrier': 43, 'excess': 43, 'signal': 43, 'rick': 43, 'gibbon': 43, 'rapid': 43, 'embed': 43, 'alter': 43, 'logic': 43, 'biz': 43, 'constitut': 43, 'whom': 43, 'georg': 43, 'pr': 43, 'ben': 42, 'battl': 42, 'usdollarnumb': 42, 'defens': 42, 'rebuild': 42, 'conduct': 42, 'ratio': 42, 'numberf': 42, 'unabl': 42, 'unwant': 42, 'owen': 42, 'amend': 42, 'becam': 42, 'comprehens': 42, 'concentr': 42, 'privileg': 42, 'regardless': 42, 'agreeabl': 42, 'ourselv': 42, 'exce': 42, 'ideal': 42, 'llc': 42, 'encount': 42, 'focus': 42, 'audienc': 42, 'corner': 42, 'conclud': 42, 'threat': 42, 'strike': 42, 'niall': 42, 'approxim': 42, 'prescript': 42, 'finger': 42, 'gatewai': 42, 'stabl': 42, 'recal': 42, 'necessarili': 42, 'joseph': 42, 'vipul': 41, 'dark': 41, 'convent': 41, 'vote': 41, 'harm': 41, 'floppi': 41, 'disclaim': 41, 'giant': 41, 'ti': 41, 'bai': 41, 'nb': 41, 'gt': 41, 'satisfact': 41, 'crazi': 41, 'liber': 41, 'neighbor': 41, 'outstand': 41, 'nigeria': 41, 'station': 41, 'green': 41, 'pound': 41, 'safeti': 41, 'bet': 41, 'weapon': 41, 'hewlett': 41, 'oil': 41, 'distro': 41, 'fashion': 41, 'okai': 41, 'postal': 41, 'grand': 41, 'mainten': 41, 'solicit': 41, 'ex': 41, 'skin': 41, 'revok': 41, 'lift': 41, 'satellit': 41, 'anonym': 40, 'rose': 40, 'em': 40, 'shoot': 40, 'glad': 40, 'decor': 40, 'randomli': 40, 'pilot': 40, 'cloth': 40, 'season': 40, 'eugen': 40, 'evolv': 40, 'perlnumb': 40, 'elect': 40, 'registr': 40, 'mastercard': 40, 'town': 40, 'spamd': 40, 'crap': 40, 'greet': 40, 'packard': 40, 'numbernd': 40, 'penni': 40, 'reject': 40, 'ng': 40, 'heck': 40, 'congratul': 40, 'assumpt': 40, 'smaller': 40, 'revis': 40, 'flat': 40, 'imho': 40, 'son': 40, 'africa': 40, 'song': 40, 'infrastructur': 40, 'weird': 39, 'demonstr': 39, 'comp': 39, 'pentium': 39, 'notifi': 39, 'kit': 39, 'netscap': 39, 'attend': 39, 'silli': 39, 'sub': 39, 'ing': 39, 'ban': 39, 'precis': 39, 'valhalla': 39, 'overal': 39, 'exit': 39, 'brows': 39, 'retir': 39, 'belong': 39, 'ahead': 39, 'rout': 39, 'gather': 39, 'gotten': 39, 'london': 39, 'resum': 39, 'unknown': 39, 'banner': 39, 'televis': 39, 'rss': 39, 'whatsoev': 39, 'acknowledg': 39, 'cheaper': 39, 'pda': 39, 'successfulli': 39, 'domest': 39, 'euro': 39, 'enemi': 39, 'unusu': 39, 'debian': 39, 'ow': 39, 'victim': 39, 'presenc': 39, 'await': 39, 'corpu': 38, 'besid': 38, 'cio': 38, 'dublin': 38, 'fantasi': 38, 'ton': 38, 'numberrd': 38, 'distributor': 38, 'dnumber': 38, 'formula': 38, 'loos': 38, 'belief': 38, 'numberxnumb': 38, 'bitbitch': 38, 'brief': 38, 'latter': 38, 'observ': 38, 'portabl': 38, 'chicago': 38, 'blow': 38, 'compens': 38, 'daniel': 38, 'craig': 38, 'reward': 38, 'insist': 38, 'diet': 38, 'conclus': 38, 'depress': 38, 'union': 38, 'pill': 38, 'emot': 38, 'authent': 38, 'frame': 38, 'smoke': 38, 'mozilla': 38, 'ved': 37, 'newspap': 37, 'collector': 37, 'porn': 37, 'die': 37, 'lnumber': 37, 'da': 37, 'pleasur': 37, 'proxi': 37, 'harlei': 37, 'partnership': 37, 'india': 37, 'crack': 37, 'rush': 37, 'committe': 37, 'rapidli': 37, 'receipt': 37, 'wire': 37, 'reduct': 37, 'renam': 37, 'toner': 37, 'cartridg': 37, 'rank': 37, 'pursu': 37, 'si': 37, 'threaten': 37, 'border': 37, 'angel': 37, 'wild': 37, 'colleagu': 37, 'mo': 37, 'capac': 37, 'pure': 37, 'entiti': 37, 'holidai': 37, 'spin': 37, 'trend': 37, 'compel': 37, 'php': 36, 'fuck': 36, 'felt': 36, 'aggress': 36, 'advisor': 36, 'forum': 36, 'hook': 36, 'handi': 36, 'fan': 36, 'py': 36, 'algorithm': 36, 'raw': 36, 'layer': 36, 'bob': 36, 'bless': 36, 'shift': 36, 'shall': 36, 'width': 36, 'declar': 36, 'sector': 36, 'assembl': 36, 'adjust': 36, 'thoma': 36, 'faith': 36, 'mid': 36, 'hadn': 36, 'shut': 36, 'plugin': 36, 'relai': 36, 'fault': 36, 'zone': 36, 'album': 36, 'opposit': 36, 'bound': 36, 'crime': 36, 'nationwid': 36, 'race': 36, 'destruct': 36, 'razornumb': 36, 'dealer': 36, 'plant': 36, 'ebook': 36, 'bone': 36, 'husband': 35, 'king': 35, 'arrest': 35, 'junk': 35, 'captur': 35, 'gai': 35, 'orient': 35, 'debat': 35, 'scientist': 35, 'guido': 35, 'engag': 35, 'violat': 35, 'routin': 35, 'cf': 35, 'compress': 35, 'ebai': 35, 'smith': 35, 'happier': 35, 'expos': 35, 'inspir': 35, 'con': 35, 'suffici': 35, 'erect': 35, 'stabil': 35, 'resolut': 35, 'vision': 35, 'consequ': 35, 'familiar': 35, 'pressur': 35, 'fed': 35, 'destroi': 35, 'terror': 35, 'began': 35, 'everywher': 35, 'chart': 35, 'stuck': 35, 'pipe': 35, 'loop': 34, 'interpret': 34, 'brent': 34, 'export': 34, 'interrupt': 34, 'crucial': 34, 'startup': 34, 'pars': 34, 'pair': 34, 'kick': 34, 'complic': 34, 'britain': 34, 'surround': 34, 'joke': 34, 'vulner': 34, 'retriev': 34, 'texa': 34, 'truck': 34, 'lcd': 34, 'decent': 34, 'principl': 34, 'transmit': 34, 'branch': 34, 'entrepreneur': 34, 'preserv': 34, 'refund': 34, 'seed': 34, 'corrupt': 34, 'rent': 34, 'int': 34, 'steal': 34, 'virginia': 34, 'trace': 34, 'scratch': 34, 'heat': 34, 'tast': 34, 'wrinkl': 34, 'el': 34, 'numberi': 34, 'txt': 34, 'shanumb': 34, 'python': 34, 'po': 34, 'spring': 34, 'examin': 33, 'gibson': 33, 'fell': 33, 'lifetim': 33, 'fuel': 33, 'aa': 33, 'navig': 33, 'ext': 33, 'bed': 33, 'coast': 33, 'chosen': 33, 'tweak': 33, 'incorpor': 33, 'schuman': 33, 'su': 33, 'florida': 33, 'depth': 33, 'bankruptci': 33, 'francisco': 33, 'chain': 33, 'element': 33, 'newsgroup': 33, 'manner': 33, 'electr': 33, 'diseas': 33, 'cant': 33, 'creation': 33, 'meatspac': 33, 'draft': 33, 'domin': 33, 'bounc': 33, 'lawsuit': 33, 'est': 33, 'assign': 33, 'broadcast': 33, 'overnight': 33, 'occasion': 33, 'drink': 33, 'cancer': 33, 'lie': 33, 'mlm': 33, 'lai': 33, 'loonei': 33, 'reboot': 33, 'builder': 33, 'entitl': 33, 'elig': 33, 'trigger': 33, 'tcl': 33, 'mountain': 33, 'ct': 33, 'mnumber': 33, 'ugli': 33, 'echo': 32, 'debug': 32, 'rewrit': 32, 'trivial': 32, 'arial': 32, 'rid': 32, 'recov': 32, 'sweet': 32, 'attain': 32, 'ir': 32, 'dc': 32, 'harder': 32, 'constantli': 32, 'circumst': 32, 'reflect': 32, 'appeal': 32, 'nativ': 32, 'extraordinari': 32, 'ride': 32, 'wit': 32, 'inquiri': 32, 'auction': 32, 'novemb': 32, 'acquisit': 32, 'demo': 32, 'clock': 32, 'hotel': 32, 'epson': 32, 'kinda': 32, 'medicin': 32, 'perspect': 32, 'survei': 32, 'theft': 32, 'enforc': 32, 'rip': 32, 'maxim': 32, 'drag': 32, 'intellectu': 32, 'invok': 32, 'struggl': 32, 'dial': 32, 'yield': 32, 'lesson': 32, 'dedic': 32, 'lucki': 32, 'sendmail': 32, 'km': 32, 'va': 32, 'weather': 32, 'render': 32, 'primarili': 32, 'movement': 32, 'zealot': 32, 'dsl': 32, 'norton': 32, 'iirc': 31, 'protest': 31, 'shirt': 31, 'ic': 31, 'contest': 31, 'afternoon': 31, 'hal': 31, 'pt': 31, 'wipe': 31, 'av': 31, 'rohit': 31, 'destin': 31, 'transform': 31, 'chines': 31, 'forg': 31, 'asid': 31, 'rah': 31, 'western': 31, 'whenev': 31, 'refus': 31, 'alsa': 31, 'wisdom': 31, 'gif': 31, 'wow': 31, 'holder': 31, 'lawyer': 31, 'bu': 31, 'newest': 31, 'gpl': 31, 'freeli': 31, 'redistribut': 31, 'corp': 31, 'spirit': 31, 'magic': 31, 'apach': 31, 'till': 31, 'recruit': 31, 'scientif': 31, 'somewhat': 31, 'stephen': 31, 'remedi': 31, 'cb': 31, 'bond': 31, 'intent': 31, 'airlin': 31, 'dot': 31, 'hospit': 31, 'sh': 31, 'arrai': 31, 'reg': 31, 'hesit': 31, 'difficulti': 31, 'voyag': 31, 'bore': 31, 'kindli': 31, 'buri': 31, 'hopefulli': 31, 'exec': 31, 'modif': 30, 'killer': 30, 'influenc': 30, 'helvetica': 30, 'curiou': 30, 'recompil': 30, 'mutual': 30, 'exploit': 30, 'cpu': 30, 'matthew': 30, 'hmmm': 30, 'testimoni': 30, 'cook': 30, 'mathemat': 30, 'sea': 30, 'legisl': 30, 'indian': 30, 'bright': 30, 'democrat': 30, 'sophist': 30, 'relax': 30, 'fellow': 30, 'hill': 30, 'certifi': 30, 'medium': 30, 'significantli': 30, 'inkjet': 30, 'moral': 30, 'drunken': 30, 'sober': 30, 'settl': 30, 'creator': 30, 'payabl': 30, 'wave': 30, 'anthoni': 30, 'attribut': 30, 'webcam': 30, 'hugh': 30, 'lab': 30, 'hawaii': 30, 'mexico': 30, 'vice': 30, 'pic': 30, 'rnumber': 30, 'accomplish': 30, 'icon': 30, 'ball': 30, 'syntax': 30, 'verif': 30, 'agenda': 30, 'tmp': 30, 'incompat': 30, 'floor': 30, 'compaq': 30, 'forgotten': 30, 'larri': 30, 'shout': 30}

    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        # default="X_train_bin.csv",
                        default="X_train_count.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="y_train.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        # default="X_test_bin.csv",
                        default="X_test_count.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="y_test.csv",
                        help="filename for labels associated with the test data")
    # parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    if True:
        args.epoch=2000

        np.random.seed(args.seed)   
        model = Perceptron(args.epoch)
        trainStats = model.train(xTrain, yTrain, words)
        print(trainStats)
        yHat = model.predict(xTest)
        # print out the number of mistakes
        print("Number of mistakes on the test dataset")
        print(calc_mistakes(yHat, yTest))
    else:
        epochs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        for epoch in epochs:
            args.epoch=epoch

            X = np.concatenate((xTrain, xTest), axis=0)
            y = np.concatenate((yTrain, yTest), axis=0)
            
            mistakes = 0
            n_splits=5
            kf = KFold(n_splits=n_splits)
            for train_index, test_index in kf.split(X):
                xTrain, xTest = X[train_index], X[test_index]
                yTrain, yTest = y[train_index], y[test_index]

                np.random.seed(args.seed)   
                model = Perceptron(args.epoch)
                trainStats = model.train(xTrain, yTrain)
                # print(trainStats)
                yHat = model.predict(xTest)
                # # print out the number of mistakes
                # print("Number of mistakes on the test dataset")
                mistakes += calc_mistakes(yHat, yTest)
            print("Epoch:", epoch, " Mistakes:", mistakes / n_splits)

if __name__ == "__main__":
    main()

# COUNT
# Epoch: 1  Mistakes: 374.4
# Epoch: 2  Mistakes: 368.8
# Epoch: 5  Mistakes: 220.2
# Epoch: 10  Mistakes: 187.2
# Epoch: 20  Mistakes: 234.4
# Epoch: 50  Mistakes: 75.2
# Epoch: 100  Mistakes: 82.2
# Epoch: 200  Mistakes: 50.8
# Epoch: 500  Mistakes: 34.2
# Epoch: 1000  Mistakes: 31.4
# Epoch: 2000  Mistakes: 29.0
# Epoch: 5000  Mistakes: 29.8
# Epoch: 10000  Mistakes: 29.8

# Epoch: 2000 => 3 Errors on Training and 27 Errors on Testing
# Positive: ['will', 'remov', 'call', 'dollarnumb', 'name', 'busi', 'compani', 'site', 'monei', 'below', 'order', 'report', 'guarante', 'size', 'numberc']
# Negative: ['your', 'our', 'free', 'click', 'send', 'year', 'market', 'month', 'profession', 'insur', 'face', 'anumb', 'numberb', 'enumb', 'bnumber']



# BINARY
# Epoch: 1  Mistakes: 374.4
# Epoch: 2  Mistakes: 374.2
# Epoch: 5  Mistakes: 71.8
# Epoch: 10  Mistakes: 92.8
# Epoch: 20  Mistakes: 30.6
# Epoch: 50  Mistakes: 24.4
# Epoch: 100  Mistakes: 21.8
# Epoch: 200  Mistakes: 21.6
# Epoch: 500  Mistakes: 22.0
# Epoch: 1000  Mistakes: 21.8
# Epoch: 2000  Mistakes: 21.8

# Epoch: 200 => 2 Erros on Training and 15 Errors on Testing
# Positive: ['your', 'we', 'will', 'email', 'here', 'our', 'pleas', 'click', 'remov', 'dollarnumb', 'form', 'monei', 'below', 'guarante', 'sight']
# Negative: ['no', 'inform', 'free', 'name', 'nbsp', 'offer', 'market', 'life', 'hour', 'within', 'pai', 'dollar', 'credit', 'fill' 'deathtospamdeathtospamdeathtospam']


