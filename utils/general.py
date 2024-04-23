import logging
import os
import sys
import time
import re
import pickle
import argparse
from collections import OrderedDict
from dataclasses import fields
from typing import Tuple, Any

#############################################################################
# python general


# main
def main():
    pass


if __name__ == "__main__":
    main()


def logger2file(log_path="train.log"):
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )


def set_logger_file_n_console(log_path="train.log"):
    """Log to console and file"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def logger2(log_path="train.log"):
    """Log to console and file"""

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(log_format))
    ch.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger(__name__)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def sysStdin():
    arr = []
    for line in sys.stdin:
        if line[-1] == "\n":
            line = line[:-1]
        arr.append(line)


def write_lines2file(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def read_f_lines(path, strip_nline=True):
    lines = []
    with open(path) as fp:
        lines = fp.readlines()
    if strip_nline:
        lines = [line.rstrip("\n") for line in lines]
    return lines


def string_to_bool(v: str):
    """
    Can be used for argparse E.g.
    parser.add_argument("--bool_flag", type=string_to_bool, required=False, default="false")
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise TypeError


def timer():
    t_start = time.time()
    print(f"Time elapsed: {time.time() - t_start:.1f} s")


def pickle_dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def flatten_2d_list(input_list):
    res_list = []
    for sublist in input_list:
        for item in sublist:
            res_list.append(item)

    return res_list


def get_sorted_file_path_list(folder_path):
    file_path_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # append the file name to the list
            file_path_list.append(os.path.join(root, file))
    return sorted(file_path_list)


# group functions into a class with staticmethod
class BunchFn(object):
    def __init__(self):
        pass

    @staticmethod
    def echo(t):
        return t


class PowerDict(OrderedDict):
    """
    Adapted from huggingface ModelOutput.
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that
    allows indexing by integer or slice (like a tuple) or strings (like a
    dictionary) that will ignore the ``None`` attributes. Otherwise behaves
    like a regular python dictionary.

    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have > 1 required field."

        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        string = self.__class__.__name__
        raise Exception(f"Cannot use ``__delitem__`` on a {string} instance.")

    def setdefault(self, *args, **kwargs):
        string = self.__class__.__name__
        raise Exception(f"Cannot use ``setdefault`` on a {string} instance.")

    def pop(self, *args, **kwargs):
        string = self.__class__.__name__
        raise Exception(f"Cannot use ``pop`` on a {string} instance.")

    def update(self, *args, **kwargs):
        string = self.__class__.__name__
        raise Exception(f"Cannot use ``update`` on a {string} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


#############################################################################
# get_args


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2list(v):
    """convert str into list, e.g., "1,2,3" -> [1,2,3]"""
    # format check
    res = re.findall(r"[^0-9,]", v)  # only allow , and digits
    assert len(res) == 0
    res = re.findall(r",,+", v)
    assert len(res) == 0

    ls = v.split(",")
    ls = sorted([int(x) for x in ls])
    assert len(ls) > 0
    assert len(ls) == len(set(ls))  # repeat element
    assert min(ls) >= 0 and max(ls) <= 11
    return ls


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--mode", type=str, required=True, help="")
    parser.add_argument("--seed", type=int, default=1122)
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--disable_display", type=str2bool, default=False, help="display progress bar or not")
    args = parser.parse_args()
    print(args)

    return args


#############################################################################
# multiprocessing


def multiprocessing_with_argparse():
    from argparse import ArgumentParser
    from multiprocessing import Pool, Manager, set_start_method

    def f(i, args_dict):
        """Worker function"""
        print(args_dict["n_workers"])
        for x in range(2**16):
            i += x
        return i

    def main():
        # parse args
        parser = ArgumentParser()
        parser.add_argument("--data_path", type=str, default="cool")
        parser.add_argument("--n_workers", type=int, default=5)
        args = get_args()

        # mp
        set_start_method("spawn")
        with Manager() as manager:
            args_dict = manager.dict(vars(args))
            pool_tuple = [(i, args_dict) for i in range(1, args.n_workers + 10)]
            with Pool(args.n_workers) as p:
                res = p.starmap(f, pool_tuple)
            print(res)

    if __name__ == "__main__":
        main()


def multiprocessing():
    from multiprocessing import Pool

    n_workers = 30
    t_start = time.time()

    def f1(i):
        """Worker function"""
        for x in range(2**23):
            i += x
        return i

    with Pool(n_workers) as p:
        results = p.map(f1, list(range(n_workers)))

    print(f"Time elapsed: {time.time() - t_start:.1f} s")
