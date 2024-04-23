import logging
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import nltk
from rouge import Rouge
from bert_score import score as bert_score

from utils.data_science import get_freq_and_dist, get_ls_mean
from utils.text_norm import norm_text_v3


class Metrics(object):
    @staticmethod
    def get_len(t):
        return len(nltk.word_tokenize(t))

    @staticmethod
    def avg_len(texts):
        # texts_tokenized = [nltk.word_tokenize(t) for t in texts]
        texts_tokenized = [t.split() for t in texts]
        t_lengths = [len(t) for t in texts_tokenized]
        return sum(t_lengths) / len(t_lengths)

    @staticmethod
    def rouge_score(refs: List[str], hyps: List[str]):
        """
        https://github.com/pltrdy/rouge

        hyps = ["This implementation are independant from the official script", "No good metrics."]
        refs = ["This implementation is independant from the official script", "No more metrics."]

        rouge_1, rouge_2, rouge_l, rouge_1_ls, rouge_2_ls, rouge_l_ls = Metrics.rouge_score(refs, hyps)
        print(rouge_1, rouge_2, rouge_l, rouge_1_ls, rouge_2_ls, rouge_l_ls)

        hyps = ["This implementation is independant from the official script", "No more metrics."]
        refs = ["This implementation is independant from the official script", "No more metrics."]

        rouge_1, rouge_2, rouge_l, rouge_1_ls, rouge_2_ls, rouge_l_ls = Metrics.rouge_score(refs, hyps)
        print(rouge_1, rouge_2, rouge_l, rouge_1_ls, rouge_2_ls, rouge_l_ls)

        # 77.08333283333334 35.714285464285716 77.08333283333334 [87.4999995, 66.66666616666667] [71.42857092857143, 0.0] [87.4999995, 66.66666616666667]
        # 99.9999995 99.9999995 99.9999995 [99.9999995, 99.9999995] [99.9999995, 99.9999995] [99.9999995, 99.9999995]
        """
        rouge = Rouge()

        scores = rouge.get_scores(hyps, refs, avg=False)
        rouge_1_ls = [di["rouge-1"]["f"] * 100 for di in scores]
        rouge_2_ls = [di["rouge-2"]["f"] * 100 for di in scores]
        rouge_l_ls = [di["rouge-l"]["f"] * 100 for di in scores]

        rouge_1, rouge_2, rouge_l = get_ls_mean(rouge_1_ls), get_ls_mean(rouge_2_ls), get_ls_mean(rouge_l_ls)

        return rouge_1, rouge_2, rouge_l, rouge_1_ls, rouge_2_ls, rouge_l_ls

    @staticmethod
    def bert_score(
        refs: List[str],
        hyps: List[str],
        lang: str = "en",
        verbose: bool = False,
        rescale_with_baseline: bool = False,
    ):
        """

        https://github.com/Tiiiger/bert_score/

        hyps = ["This implementation are independant from the official script", "No good metrics."]
        refs = ["This implementation is independant from the official script", "No more metrics."]

        P, R, F1, P_ls, R_ls, F1_ls = Metrics.bert_score(refs, hyps)
        print(P, R, F1, P_ls, R_ls, F1_ls)

        hyps = ["This implementation is independant from the official script", "No more metrics."]
        refs = ["This implementation is independant from the official script", "No more metrics."]

        P, R, F1, P_ls, R_ls, F1_ls = Metrics.bert_score(refs, hyps)
        print(P, R, F1, P_ls, R_ls, F1_ls)

        # 95.74147415161133 95.78520202636719 95.76332473754883 [98.09466552734375, 93.3882827758789] [98.09466552734375, 93.47573852539062] [98.09466552734375, 93.4319839477539]
        # 99.99999237060547 99.99999237060547 99.99999237060547 [99.99999237060547, 99.99999237060547] [99.99999237060547, 99.99999237060547] [99.99999237060547, 99.99999237060547]

        hyps = ["This implementation is new", "book"]
        refs = ["This implementation is independant", "No more metrics."]

        bert_score(hyps, refs,  lang="en", rescale_with_baseline=True)
        # (tensor([0.6485, 0.1473]), tensor([0.4786, 0.1566]), tensor([0.5629, 0.1533]))


        """
        P_raw, R_raw, F1_raw = bert_score(
            hyps, refs, lang=lang, verbose=verbose, rescale_with_baseline=rescale_with_baseline
        )

        P_ls, R_ls, F1_ls = (P_raw * 100).tolist(), (R_raw * 100).tolist(), (F1_raw * 100).tolist()
        P, R, F1 = get_ls_mean(P_ls), get_ls_mean(R_ls), get_ls_mean(F1_ls)

        return P, R, F1, P_ls, R_ls, F1_ls

    @staticmethod
    def hit_rate(hyp_ls: List[str], tgts_ls: List[List[str]]):
        """
        hyp_ls = ["car", "cake"]
        tgts_ls = [["car", "computer"], ["air", "sky", "tea"]]
        print(hit_rate(hyp_ls, tgts_ls))
        # (50.0, [1, 0])
        """

        assert len(hyp_ls) == len(tgts_ls)
        hit_ls: List[int] = []
        for i, hyp in enumerate(hyp_ls):
            tgts = tgts_ls[i]
            if hyp in tgts:
                hit_ls += [1]
            else:
                hit_ls += [0]
        hr = get_ls_mean(hit_ls) * 100

        return hr, hit_ls


class Evaluator(object):
    """
    Evaluate a prediction df.
    if eval_mode == "gen":
        df should contain columns ["src", "tgt", "hyp"]
    elif eval_mode == "hit":
        df should contain columns ["tgts", "hyp"]
    """

    # these attributes may be added to data samples after eval
    EVAL_ATTRS = ("rouge_1", "rouge_2", "rouge_l", "bert_f1", "hit")

    def eval_gen(self, df: pd.DataFrame = None, bertscore_rescale: bool = True, verbose: bool = True):
        """
        Evaluate with typical metrics for generation: rouge, BERTscore etc.
        df: contains columns ["src", "tgt", "hyp"]
        """
        # well-formed targets
        df.src = df.src.apply(norm_text_v3)
        df.tgt = df.tgt.apply(norm_text_v3)
        df.hyp = df.hyp.apply(norm_text_v3)
        src_ls = list(df.src)
        tgt_ls = list(df.tgt)
        hyp_ls = list(df.hyp)

        src_len = Metrics.avg_len(src_ls)
        ref_len = Metrics.avg_len(tgt_ls)
        hyp_len = Metrics.avg_len(hyp_ls)
        if verbose:
            logging.info(f"src_len: {src_len}")
            logging.info(f"ref_len: {ref_len}")
            logging.info(f"hyp_len: {hyp_len}")

        # special hendlding of hyps: if empty, add a placeholder
        hyp_ls = [hyp if len(hyp.split()) > 0 else "None" for hyp in hyp_ls]

        # rouge
        rouge_1, rouge_2, rouge_l, rouge_1_ls, rouge_2_ls, rouge_l_ls = Metrics.rouge_score(tgt_ls, hyp_ls)
        df["rouge_1"] = rouge_1_ls
        df["rouge_2"] = rouge_2_ls
        df["rouge_l"] = rouge_l_ls
        if verbose:
            logging.info(f"rouge_1: {rouge_1:.3f}")
            logging.info(f"rouge_2: {rouge_2:.3f}")
            logging.info(f"rouge_l: {rouge_l:.3f}")

        # BERTscore
        P, R, F1, P_ls, R_ls, F1_ls = Metrics.bert_score(
            tgt_ls, hyp_ls, rescale_with_baseline=bertscore_rescale
        )
        df["bert_f1"] = F1_ls
        if verbose:
            logging.info(f"bert_score, P: {P:.3f}")
            logging.info(f"bert_score, R: {R:.3f}")
            logging.info(f"bert_score, F1: {F1:.3f}")

        # sentiment distribution
        if "hyp_sentiment" in df.columns:
            data_list = list(df.hyp_sentiment)
            frequency, distribution = get_freq_and_dist(data_list)
            frequency = dict(frequency)
            distribution = {k: round(v, 2) for k, v in distribution.items()}
            if verbose:
                logging.info(f"hyp_sentiment stats: frequency: {frequency}. distribution: {distribution}.")

        # output CSV string
        if verbose:
            logging.info(f"rouge_1/rouge_2/rouge_l/bert_score_F1/hyp_len:")
            logging.info(f"{rouge_1:.1f}, {rouge_2:.1f}, {rouge_l:.1f}, {F1:.1f}, {hyp_len:.1f}")

    def eval_hit(self, df: pd.DataFrame = None, verbose: bool = True):
        """
        Evaluate with typical metrics for generation: rouge, BERTscore etc.
        df: contains columns ["tgts", "hyp"]
        """

        df.tgts = df.tgts.apply(lambda ls: [norm_text_v3(txt) for txt in ls])
        df.hyp = df.hyp.apply(norm_text_v3)

        tgts_ls = list(df.tgts)  # 2d list
        hyp_ls = list(df.hyp)
        hit_rate, hit_ls = Metrics.hit_rate(hyp_ls, tgts_ls)
        df["hit"] = hit_ls

        if verbose:
            logging.info(f"hit_rate: {hit_rate:.3f}")

    def work(
        self,
        df: pd.DataFrame = None,
        bertscore_rescale: bool = True,
        eval_mode: str = "gen",
        verbose: bool = True,
    ):
        try:
            if eval_mode == "gen":
                self.eval_gen(df, bertscore_rescale=bertscore_rescale, verbose=verbose)
            elif eval_mode == "hit":
                self.eval_hit(df, verbose=verbose)

        except Exception as error:
            if verbose:
                logging.info("An exception occurred:", error)

            # tgt may contain NaN, as certain datasets lack annotation
            df.src = df.src.apply(norm_text_v3)
            df.hyp = df.hyp.apply(norm_text_v3)

            src_len = Metrics.avg_len(list(df.src))
            hyp_len = Metrics.avg_len(list(df.hyp))
            if verbose:
                logging.info(f"src_len: {src_len}")
                logging.info(f"hyp_len: {hyp_len}")
