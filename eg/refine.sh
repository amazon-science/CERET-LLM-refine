# Example cmd for refining llm outputs. 
# Inputs are stored in $stage1_dir. Outputs are stored in $stage2_dir.

code_dir="/PATH/TO/CERET-LLM-refine"
cd $code_dir
# conda activate <YOUR_CONDA_ENV>

stage1_dir="${code_dir}/exp/dialog_sum/vicuna13b_dialogsum_test_1107"
stage2_dir="${stage1_dir}_refine1"

eval_mode="gen"
mkdir -p $stage2_dir
export PYTHONPATH=$code_dir

python eval/refiner.py \
    --in_path "${stage1_dir}/data.json" \
    --in_grouped_hyps_path  "${stage1_dir}/grouped_hyps.json" \
    --out_path "${stage2_dir}/data.json" \
    --out_grouped_hyps_path "${stage2_dir}/grouped_hyps.json" \
    --tune_coeff_res_path "${stage2_dir}/tune_coeff_res.json" \
    --log_outpath "${stage2_dir}/log.txt" \
    --emb_save_path "${stage2_dir}/semantic_emb.npy" \
    --nli_save_path "${stage2_dir}/nli_save_res.json" \
    --eval_mode $eval_mode \
    --score_coeff1 0.3333 \
    --score_coeff2 0.3333 \
    --do_coefficient_tuning true
