"""Refine LLM outputs"""

from abc import ABC
import argparse
from collections import defaultdict
from dataclasses import dataclass
import math
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Tuple, Union
import time
import logging
import os
import re
import json

import numpy as np
import pandas as pd
from scipy.spatial import distance
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from eval.metrics import Evaluator
from utils.general import set_logger_file_n_console, str2bool, flatten_2d_list
from utils.args import get_args

@dataclass
class HypRecord:
    hyp_id: str = None
    hyp: str = None
    overall_score: float = None
    stability_score: float = None
    entailment_score: float = None
    uncertainty_score: float = None


def recover_sample_id_from_hyp_id(hyp_id: str = None):
    """
    "hyp_id" has the form f"{sample_id}_{index_in_group}"
    """
    sample_id = "_".join(hyp_id.split("_")[:-1])
    return sample_id


class TextsScorer(ABC):
    def score_hyp_2dls(self, hyp_2dls: List[List[HypRecord]]):
        """
        score a 2d list of HypRecord's
        """
        raise NotImplementedError


class SemanticScorer(TextsScorer):
    """Util class handling semantic embedding generation and I/O"""

    def __init__(
        self,
        tokenizer_name_or_path: str = "roberta-base",
        model_name_or_path: str = "roberta-base",
        batch_size: int = 16,
        torch_device: str = "cpu",
        **kwargs,
    ):
        self.batch_size = batch_size
        self.torch_device = torch_device
        self.combined_emb_array = None

        # init model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def gen_embds(self, text_ls: List[str]):
        """generate embeddings with PLM"""
        model = self.model
        tokenizer = self.tokenizer
        if self.torch_device == "cuda":
            model = nn.DataParallel(model)
            model = model.cuda()

        emb_ls = []
        iterator = tqdm(range(0, len(text_ls), self.batch_size))
        for batch_idx in iterator:
            texts = text_ls[batch_idx : batch_idx + self.batch_size]
            input = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

            if self.torch_device == "cuda":
                for key, value in input.items():
                    input[key] = input[key].cuda()
            output = model(**input)
            cls_repr = output.last_hidden_state[:, 0, :]  # shape (batch_size, hidden_dimension)
            emb_ls += [cls_repr.detach().cpu().numpy()]
        combined_emb_array = np.concatenate(emb_ls, axis=0)

        assert len(text_ls) == combined_emb_array.shape[0]

        return combined_emb_array

    def get_emb_io(self, text_ls: List[str], emb_save_path: str = None):
        """Either load precomputed embs, or generate new embs and save."""

        # get embeddings
        combined_emb_array = None
        if emb_save_path is not None and os.path.isfile(emb_save_path):
            combined_emb_array = np.load(emb_save_path)

        elif emb_save_path is not None and not os.path.isfile(emb_save_path):
            combined_emb_array = self.gen_embds(text_ls)
            np.save(emb_save_path, combined_emb_array)

        elif emb_save_path is None:
            combined_emb_array = self.gen_embds(text_ls)

        self.combined_emb_array = combined_emb_array

        return combined_emb_array


class SemanticStabilityScorer(SemanticScorer):
    """Semantic stability"""

    def score_hyp_2dls(self, hyp_2dls: List[List[HypRecord]], emb_save_path: str = None):
        """score a 2d list of HypRecord's"""
        # flatten 2d list. [We input 2d list to preseve <sample, hyps> structure]
        hyp_records_flat = flatten_2d_list(hyp_2dls)
        hyp_ids = [hyp_record.hyp_id for hyp_record in hyp_records_flat]
        text_ls = [hyp_record.hyp for hyp_record in hyp_records_flat]
        hyp_id2idx_dict = {
            hyp_id: idx for idx, hyp_id in enumerate(hyp_ids)
        }  # used for mapping hyp_id to embed

        # combined_emb_array preserves the order of flattened hyp_2dls
        combined_emb_array = self.get_emb_io(text_ls, emb_save_path)

        # get scores
        for hyp_group in hyp_2dls:
            # get emb_group
            emb_group = []
            for hyp_record in hyp_group:
                idx = hyp_id2idx_dict[hyp_record.hyp_id]
                emb_group += [combined_emb_array[idx]]

            mean_emb = np.mean(np.array(emb_group), axis=0)
            for j, hyp_record in enumerate(hyp_group):
                dst = distance.euclidean(mean_emb, emb_group[j])
                hyp_record.stability_score = -dst

        return hyp_2dls


class EntailmentScorer(TextsScorer):
    """Entailment score"""

    def __init__(
        self,
        tokenizer_name_or_path: str = "MoritzLaurer/DeBERTa-v3-base-mnli",
        model_name_or_path: str = "MoritzLaurer/DeBERTa-v3-base-mnli",
        batch_size: int = 16,
        torch_device: str = "cpu",
        **kwargs,
    ):
        self.batch_size = batch_size
        self.torch_device = torch_device
        self.nli_score_records = None

        # init model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

    def parse_ent_declarative(self, text: str):
        return f"X is/are {text}"

    def compute_nli_score(self, score_records: List[list], ent_declarative: bool = False):
        """
        compute NLI scores with PLM
        input: score_records has format [[hyp_id_i, hyp_i, hyp_id_j, hyp_j, score_dict], ..., [...] ]
        """
        model = self.model
        tokenizer = self.tokenizer
        if self.torch_device == "cuda":
            model = nn.DataParallel(model)
            model = model.cuda()

        iterator = tqdm(range(0, len(score_records), self.batch_size))
        for batch_idx in iterator:
            # prep inputs
            score_records_batch = score_records[batch_idx : batch_idx + self.batch_size]
            texts1 = [score_record[1] for score_record in score_records_batch]
            texts2 = [score_record[3] for score_record in score_records_batch]

            if ent_declarative:
                texts1 = [self.parse_ent_declarative(txt) for txt in texts1]
                texts2 = [self.parse_ent_declarative(txt) for txt in texts2]
                if batch_idx == 0:
                    # logging two examples
                    logging.info(f"Declarative parsing for noun phrases entailment is enabled.")
                    logging.info(f"E.g. {texts1[0]}")
                    logging.info(f"E.g. {texts2[0]}")

            input_batch = tokenizer(texts1, texts2, truncation=True, padding=True, return_tensors="pt")
            if self.torch_device == "cuda":
                for key, value in input_batch.items():
                    input_batch[key] = input_batch[key].cuda()

            # inference
            output = model(**input_batch)

            # parse prediction
            softmaxes = torch.softmax(output["logits"], -1).tolist()
            label_names = ["entailment", "neutral", "contradiction"]
            predictions = [
                {name: round(float(pred) * 100, 1) for pred, name in zip(sm_prob, label_names)}
                for sm_prob in softmaxes
            ]

            assert len(score_records_batch) == len(predictions)
            for i in range(len(score_records_batch)):
                score_records[batch_idx : batch_idx + self.batch_size][i][4] = predictions[i]

        return score_records

    def score_hyp_2dls(
        self, hyp_2dls: List[List[HypRecord]], nli_save_path: str = None, ent_declarative: bool = False
    ):
        """score a 2d list of HypRecord's"""
        # score_records has format [[hyp_id_i, hyp_i, hyp_id_j, hyp_j, score_dict], ..., [...] ]
        score_records = []

        # gather score_records
        for hyp_records_ls in hyp_2dls:
            hyp_ids = [hyp_record.hyp_id for hyp_record in hyp_records_ls]
            text_ls = [hyp_record.hyp for hyp_record in hyp_records_ls]

            for i, hyp_id_i in enumerate(hyp_ids):
                hyp_i = text_ls[i]
                for j, hyp_id_j in enumerate(hyp_ids):
                    hyp_j = text_ls[j]
                    if i != j:
                        score_records += [[hyp_id_i, hyp_i, hyp_id_j, hyp_j, None]]

        # get pairwise NLI predictions: s^i_j = NLI(y_i, y_j)
        if nli_save_path is not None and os.path.isfile(nli_save_path):
            nli_score_df = pd.read_json(nli_save_path, lines=True)
            nli_score_df = nli_score_df.astype({"hyp_id_i": str, "hyp_i": str, "hyp_id_j": str, "hyp_j": str})
            score_records = nli_score_df.values.tolist()

        elif nli_save_path is not None and not os.path.isfile(nli_save_path):
            score_records = self.compute_nli_score(score_records, ent_declarative=ent_declarative)
            nli_score_df = pd.DataFrame(
                score_records, columns=["hyp_id_i", "hyp_i", "hyp_id_j", "hyp_j", "score_dict"]
            )
            nli_score_df.to_json(nli_save_path, orient="records", lines=True)

        elif nli_save_path is None:
            score_records = self.compute_nli_score(score_records, ent_declarative=ent_declarative)

        self.nli_score_records = score_records

        # get aggregated scores
        aggregated_scores = {}  # format: {hyp_id_i: entailment_score}
        for hyp_id_i, hyp_i, hyp_id_j, hyp_j, score_dict in score_records:
            if hyp_id_i in aggregated_scores:
                aggregated_scores[hyp_id_i] += score_dict["entailment"]
            else:
                aggregated_scores[hyp_id_i] = score_dict["entailment"]

        # store entailment scores
        for hyp_group in hyp_2dls:
            for hyp_record in hyp_group:
                hyp_record.entailment_score = aggregated_scores[hyp_record.hyp_id]

        return hyp_2dls


class UncertaintyScorer(SemanticScorer):
    """Inter-sample Uncertainty Scoring"""

    def score_hyp_2dls(
        self,
        hyp_2dls: List[List[HypRecord]],
        nearest_k: int = 30,
        emb_save_path: str = None,
        unc_neib_save_path: str = None,
    ):
        """score a 2d list of HypRecord's"""
        # flatten 2d list. [We input 2d list to preseve <sample, hyps> structure]
        hyp_records_flat = flatten_2d_list(hyp_2dls)
        hyp_ids = [hyp_record.hyp_id for hyp_record in hyp_records_flat]
        text_ls = [hyp_record.hyp for hyp_record in hyp_records_flat]
        hyp_id2idx_dict = {
            hyp_id: idx for idx, hyp_id in enumerate(hyp_ids)
        }  # used for mapping hyp_id to embed

        # combined_emb_array preserves the order of flattened hyp_2dls
        combined_emb_array = self.get_emb_io(text_ls, emb_save_path)

        # get scores

        # 1. compute pairwise Euclidean distance of all predictions
        hyp_id_pair2dist = {}  # a dict of form {(hyp_id_i, hyp_id_j): distance}
        for hyp_id_i in hyp_ids:
            for hyp_id_j in hyp_ids:
                if hyp_id_i != hyp_id_j and (hyp_id_i, hyp_id_j) not in hyp_id_pair2dist:
                    emb_i = combined_emb_array[hyp_id2idx_dict[hyp_id_i]]
                    emb_j = combined_emb_array[hyp_id2idx_dict[hyp_id_j]]
                    dist = distance.euclidean(emb_i, emb_j)
                    hyp_id_pair2dist[(hyp_id_i, hyp_id_j)] = dist
                    hyp_id_pair2dist[(hyp_id_j, hyp_id_i)] = dist

        # 2. construct neighbor lists ranked by distance

        # hyp_id2neib_ls: dict of form {hyp_id: neib_ls}
        # neib_ls: list of form [(hyp_id, distance)]
        hyp_id2neib_ls = {}
        for hyp_id_i in hyp_ids:
            neib_ls_i = []
            for hyp_id_j in hyp_ids:
                if hyp_id_i != hyp_id_j:
                    dist = hyp_id_pair2dist[(hyp_id_j, hyp_id_i)]
                    neib_ls_i += [(hyp_id_j, dist)]

            # sort neib_ls, so nearest neibs can be obtained easily
            neib_ls_i.sort(key=lambda tup: tup[1], reverse=False)

            hyp_id2neib_ls[hyp_id_i] = neib_ls_i

        record_ls = list(hyp_id2neib_ls.items())
        hyp_id2neibs_df = pd.DataFrame(record_ls, columns=["hyp_id", "neib_ls"])
        if unc_neib_save_path is not None:
            hyp_id2neibs_df.to_json(unc_neib_save_path, orient="records", lines=True)

        # convert to a nearest neib version
        for hyp_id in hyp_id2neib_ls:
            hyp_id2neib_ls[hyp_id] = hyp_id2neib_ls[hyp_id][:nearest_k]

        # 3. compute uncertainty scores

        for hyp_group in hyp_2dls:
            for hyp_record in hyp_group:
                hyp_id_i = hyp_record.hyp_id
                neibs = hyp_id2neib_ls[hyp_id_i]
                score = 0
                for hyp_id_j, dist in neibs:
                    sample_id_i = recover_sample_id_from_hyp_id(hyp_id_i)
                    sample_id_j = recover_sample_id_from_hyp_id(hyp_id_j)
                    if sample_id_i != sample_id_j:
                        score += 1 / (1 + dist)

                # Negative sign is to ensure that a higher score is better.
                hyp_record.uncertainty_score = -score

        return hyp_2dls


class RefinerDriver:
    @staticmethod
    def get_ref_metric_col(eval_mode: str = "gen") -> str:
        if eval_mode == "gen":
            ref_metric_col = "rouge_1"
        elif eval_mode == "hit":
            ref_metric_col = "hit"
        else:
            raise NotImplementedError

        return ref_metric_col

    @staticmethod
    def transform_hyp_2dls(hyp_2dls: List[List[HypRecord]] = None)-> None:

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
        
        u1, u2, u3 = 3.0, 0.01, 1.0
        for hyp_group in hyp_2dls:
            for hyp_record in hyp_group:
                hyp_record.stability_score = sigmoid(hyp_record.stability_score * u1)
                hyp_record.entailment_score = sigmoid(hyp_record.entailment_score * u2)
                hyp_record.uncertainty_score  = sigmoid(hyp_record.uncertainty_score * u3)

    def get_overall_score(
        self,
        hyp_2dls: List[List[HypRecord]] = None,
        coeff0: float = 0.33,
        coeff1: float = 0.33,
        coeff2: float = 0.33,
    ) -> None:
        for hyp_group in hyp_2dls:
            for hyp_record in hyp_group:
                score_ls = [
                    hyp_record.stability_score,
                    hyp_record.entailment_score,
                    hyp_record.uncertainty_score,
                ]
                # handle None
                score_ls = [score if score is not None else 0 for score in score_ls]
                # multiply by coeff
                score_ls2 = [coeff0 * score_ls[0], coeff1 * score_ls[1], coeff2 * score_ls[2]]
                hyp_record.overall_score = sum(score_ls2)

    def get_oracle(
        self,
        hyp_2dls: List[List[HypRecord]] = None,
        topN_df: pd.DataFrame = None,
        eval_mode: str = "gen",
        verbose: bool = True,
    ) -> float:
        """
        Oracle compuation concerns hyp group structure. Hence it's not inlcuded in Evaluator().
        Assume each row in topN_df already has a number in ref_metric_col.
        """
        ref_metric_col = self.get_ref_metric_col(eval_mode)

        # Select the best hyps, according to ref_metric
        hyp_id2ref_metric = {}
        for ind, r in topN_df.iterrows():
            hyp_id2ref_metric[r["hyp_id"]] = r[ref_metric_col]

        best_hyp_ids = []
        for hyp_group in hyp_2dls:
            ref_metric_ls = [hyp_id2ref_metric[hyp_record.hyp_id] for hyp_record in hyp_group]
            index_max = np.argmax(ref_metric_ls)
            best_hyp_ids += [hyp_group[index_max].hyp_id]

        best_hyp_ids_set = set(best_hyp_ids)
        oracle_df = topN_df.loc[topN_df["hyp_id"].isin(best_hyp_ids_set)]

        # compute oracle
        oracle_ref_metric = oracle_df[ref_metric_col].mean()
        if verbose:
            logging.info(f"Oracle performance -- {ref_metric_col}: {oracle_ref_metric}")

        return oracle_ref_metric

    def tune_coefficient(
        self,
        hyp_2dls: List[List[HypRecord]] = None,
        topN_df: pd.DataFrame = None,
        eval_mode: str = "gen",
        # bertscore_rescale: bool = True,
    ):
        """Assume each row in topN_df already has a number in ref_metric_col."""

        ref_metric_col = self.get_ref_metric_col(eval_mode)

        tune_coeff_res_ls: List(tuple) = []  # of form [(<coeff0>,<coeff1>,<coeff2>,<ref_metric>)]

        # search
        for coeff0 in [0.1 * i for i in range(0, 11)]:
            for coeff1 in [0.1 * i for i in range(0, 11)]:
                coeff2 = 1 - coeff0 - coeff1
                if 0.0 <= coeff2 <= 1.0:
                    # Compute overall_score
                    self.get_overall_score(hyp_2dls, coeff0=coeff0, coeff1=coeff1, coeff2=coeff2)

                    # Select the best hyps
                    for hyp_group in hyp_2dls:
                        hyp_group.sort(key=lambda hyp_record: hyp_record.overall_score, reverse=True)
                    best_hyp_ids = set([hyp_group[0].hyp_id for hyp_group in hyp_2dls])

                    # aggregation and log auto-eval results: compute average ref_metric of the refined df
                    refined_df = topN_df.loc[topN_df["hyp_id"].isin(best_hyp_ids)]
                    ref_metric = refined_df[ref_metric_col].mean()

                    tune_coeff_res_ls += [(coeff0, coeff1, coeff2, ref_metric)]
        tune_coeff_res_ls.sort(key=lambda tup: tup[3], reverse=True)
        best_combo, best_performance = tune_coeff_res_ls[0][:3], tune_coeff_res_ls[0][3]

        # output
        logging.info(
            f"After tune_coefficient(), best combo: (coeff0, coeff1, coeff2) = {best_combo}, ref_metric({ref_metric_col}) = {best_performance}"
        )
        tune_coeff_res_df = pd.DataFrame(
            tune_coeff_res_ls, columns=["coeff0", "coeff1", "coeff2", ref_metric_col]
        )

        return tune_coeff_res_df

    def evaluate_with_certain_coeff(
        self,
        hyp_2dls: List[List[HypRecord]] = None,
        topN_df: pd.DataFrame = None,
        coeff0: float = 0.33,
        coeff1: float = 0.33,
        coeff2: float = 0.33,
    ):
        # Overall
        self.get_overall_score(hyp_2dls, coeff0=coeff0, coeff1=coeff1, coeff2=coeff2)

        # Select the best hyps
        # re-ranking
        for hyp_group in hyp_2dls:
            hyp_group.sort(key=lambda hyp_record: hyp_record.overall_score, reverse=True)
        best_hyp_ids = set([hyp_group[0].hyp_id for hyp_group in hyp_2dls])
        # select
        refined_df = topN_df.loc[topN_df["hyp_id"].isin(best_hyp_ids)]
        # Evaluate
        evaluator = Evaluator()
        evaluator.work(refined_df, bertscore_rescale=args.bertscore_rescale, eval_mode=args.eval_mode)

        return refined_df

    def work(self, args) -> None:
        # Read grouped hyps
        # grouped_hyps_df has cols ["sample_id", "hyp_records"]
        topN_df = pd.read_json(args.in_path, lines=True)
        grouped_hyps_df = pd.read_json(args.in_grouped_hyps_path, lines=True)
        # when read from json, HypRecord becomes dict. Need to recover the HypRecord structure
        grouped_hyps = list(grouped_hyps_df.hyp_records)
        hyp_2dls = []
        for hyp_group in grouped_hyps:
            hyp_2dls += [[HypRecord(**hyp_record_dict) for hyp_record_dict in hyp_group]]
        logging.info("Loading hyp_2dls completed.")

        # Score

        coeff1 = args.score_coeff1
        coeff2 = args.score_coeff2
        coeff0 = 1 - coeff1 - coeff2

        logging.info(f"coeff0 = {coeff0}, coeff1 = {coeff1}, coeff2 = {coeff2}")

        # Prepare refinement scorers first, so scoring latency can be more easily calculated.
        semanticStabilityScorer = SemanticStabilityScorer()
        entailmentScorer = EntailmentScorer()
        uncertaintyScorer = UncertaintyScorer()
        logging.info(f"Finished preparing refinement scorers.")

        # 0: Semantic
        semanticStabilityScorer.score_hyp_2dls(hyp_2dls=hyp_2dls, emb_save_path=args.emb_save_path)
        logging.info("SemanticStabilityScorer() work completed.")

        # 1: Entailment
        entailmentScorer.score_hyp_2dls(
            hyp_2dls=hyp_2dls, nli_save_path=args.nli_save_path, ent_declarative=args.ent_declarative
        )
        logging.info("EntailmentScorer() work completed.")

        # 2: Uncertainty
        uncertaintyScorer.score_hyp_2dls(
            hyp_2dls=hyp_2dls,
            nearest_k=args.nearest_k,
            emb_save_path=args.emb_save_path,
            unc_neib_save_path=args.unc_neib_save_path,
        )
        logging.info("UncertaintyScorer() work completed.")
        
        # Transform scores in hyp_2dls
        self.transform_hyp_2dls(hyp_2dls)
        
        # Preparation: auto-eval all samples, for oracle and potentially coefficient_tuning
        evaluator = Evaluator()
        evaluator.work(
            topN_df, bertscore_rescale=args.bertscore_rescale, eval_mode=args.eval_mode, verbose=False
        )

        # do_coefficient_tuning
        if args.do_coefficient_tuning:
            tune_coeff_res_df = self.tune_coefficient(
                hyp_2dls=hyp_2dls,
                topN_df=topN_df,
                eval_mode=args.eval_mode,
            )
            tune_coeff_res_df.to_json(args.tune_coeff_res_path, orient="records", lines=True)
            best_combo = list(tune_coeff_res_df.loc[0, ["coeff0", "coeff1", "coeff2"]])
            logging.info("### Evaluating with best coefficient combo.")
            refined_df = self.evaluate_with_certain_coeff(
                hyp_2dls=hyp_2dls,
                topN_df=topN_df,
                coeff0=best_combo[0],
                coeff1=best_combo[1],
                coeff2=best_combo[2],
            )
        oracle_ref_metric = self.get_oracle(
            hyp_2dls=hyp_2dls,
            topN_df=topN_df,
            eval_mode=args.eval_mode,
            verbose=True,
        )

        logging.info("### Evaluating with input coefficient combo.")
        refined_df = self.evaluate_with_certain_coeff(
            hyp_2dls=hyp_2dls, topN_df=topN_df, coeff0=coeff0, coeff1=coeff1, coeff2=coeff2
        )
        # Save
        grouped_hyps_df.hyp_records = hyp_2dls
        grouped_hyps_df.to_json(args.out_grouped_hyps_path, orient="records", lines=True)

        refined_df.to_json(args.out_path, orient="records", lines=True)


if __name__ == "__main__":
    args = get_args()
    set_logger_file_n_console(args.log_outpath)
    logging.info(f"Args: {args}")

    driver = RefinerDriver()
    driver.work(args)
