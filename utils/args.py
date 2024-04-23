import argparse
from utils.general import str2bool


def get_args():
    parser = argparse.ArgumentParser()

    ########################################################################
    group = parser.add_argument_group("File input/output")
    group.add_argument("--in_path", type=str, default=None)
    group.add_argument("--out_path", type=str, default=None)
    group.add_argument("--log_outpath", type=str, default=None)
    group.add_argument("--in_grouped_hyps_path", type=str, default=None)
    group.add_argument("--out_grouped_hyps_path", type=str, default=None)
    group.add_argument("--emb_save_path", type=str, default=None)
    group.add_argument("--nli_save_path", type=str, default=None)
    group.add_argument("--unc_neib_save_path", type=str, default=None)
    group.add_argument("--tune_coeff_res_path", type=str, default=None)
    group.add_argument("--do_overwriting", type=str2bool, default="false")

    ########################################################################
    group = parser.add_argument_group("LLM generation")
    group.add_argument("--prompt_template", type=str, default="please do XYZ:")
    group.add_argument("--template_id", type=str, default=None)
    group.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    group.add_argument(
        "--temperature", type=float, default=0.0, help="Lower temperatures mean less random generations."
    )
    group.add_argument(
        "--topP",
        type=float,
        default=1,
        help="The smallest set of tokens with probabilities that add up to top_p or higher are kept.",
    )
    group.add_argument("--topK", type=int, default=250, help="Penalty applied to previously present tokens")
    group.add_argument("--topN", type=int, default=1, help="Top n predictions for each input")
    group.add_argument("--model_type", type=str, default="amazon.titan-tg1-large")
    group.add_argument("--request_interval_sec", type=int, default=-1)
    # these are for local inference:
    group.add_argument("--batch_size", type=int, default=12)
    group.add_argument("--torch_device", type=str, default="cuda")
    group.add_argument("--dp", type=str2bool, default="false", help="nn.DataParallel")

    ########################################################################
    group = parser.add_argument_group("Parsing")
    group.add_argument("--trunc_dialogue", type=int, default=-1, help="By word count")
    group.add_argument("--trunc_dialogue_tok", type=int, default=-1, help="By token count.")
    group.add_argument("--post_process_max_len", type=int, default=-1)
    group.add_argument("--field_ops", type=str, default="")
    group.add_argument("--hyp_field_json", type=str, default=None)
    group.add_argument("--post_process_ops", type=str, default="")

    ########################################################################
    group = parser.add_argument_group("Evaluation")
    group.add_argument("--eval_mode", default="gen", const="gen", nargs="?", choices=["gen", "hit"])
    group.add_argument("--bertscore_rescale", type=str2bool, default="true")

    ########################################################################
    group = parser.add_argument_group("Refinement")
    group.add_argument("--score_coeff1", type=float, default=0.3333, help="")
    group.add_argument("--score_coeff2", type=float, default=0.3333, help="")
    group.add_argument(
        "--ent_declarative",
        type=str2bool,
        default="false",
        help="Declarative parsing for noun phrases entailment. E.g. George Washington -> X is/are George Washington",
    )
    group.add_argument("--nearest_k", type=int, default=3, help="")
    group.add_argument("--do_coefficient_tuning", type=str2bool, default="false")

    args = parser.parse_args()
    return args
