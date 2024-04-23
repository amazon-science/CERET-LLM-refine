import re
import string


def strip_punctuation(txt: str) -> str:
    txt = txt.replace(".", " ")
    txt = txt.replace("-", " ")

    # strip punctuation
    pattern = '[!"$%&()*+,\:;=?@\^_`{|}#]'
    txt = re.sub(pattern, "", txt)

    return txt


def proc_tags(txt: str) -> str:
    """remove different brackets"""
    txt = re.sub("\<.*?\>", "", txt)
    txt = re.sub("\(\(.*?\)\)", "", txt)
    txt = re.sub("\(.*?\)", "", txt)
    txt = re.sub("\[.*?\]", "", txt)
    txt = re.sub("।", "", txt)
    return txt


def norm_text(txt: str) -> str:
    txt = strip_punctuation(txt)
    txt = proc_tags(txt)
    # lower case
    txt = txt.lower()
    # remove duplicate spaces
    txt = re.sub(" +", " ", txt)
    return txt


def norm_text_v2(txt: str) -> str:
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    return txt.lower()


def norm_text_v3(txt: str) -> str:
    # remove punc
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = txt.lower()
    # remove duplicate spaces and leading/trailing spaces
    txt = re.sub(" +", " ", txt).strip()
    return txt


def test():
    norm_text("who will work on zero-shot learning? I'm doing it. A.F.L. [New] ")
    # "who will work on zero shot learning i'm doing it a f l "
    norm_text_v2("who will work on zero-shot learning? I'm doing it. A.F.L. [New] ")
    # "who will work on zeroshot learning im doing it afl new "
    norm_text_v3("  who will  work on zero-shot learning? I'm doing it. A.F.L. [New] ")
    # 'who will work on zeroshot learning im doing it afl new'
