sample_dir="evals/siglip_32k" 
torchrun \
--nproc_per_node=... \
--master_addr=... --master_port=... \
tokenizer/reconstruction_vq_ddp.py \
--sample-dir=$sample_dir \
--codebook-size=32768 \
--image-size 384 \
--image-size-eval 384 \
--per-proc-batch-size 8 \
--teacher "siglip_384" \
--vq-ckpt="path/to/tokenflow_siglip_32k.pt" &&


src_dir1=$sample_dir/samples/ &&
src_dir2=$sample_dir/gts/ &&
python3 -m evaluations.vq.pytorch_fid $src_dir1 $src_dir2 --device cuda:0 --results_path $sample_dir


set +x
target_folder_path=$sample_dir/demos/
mkdir -p "$target_folder_path"
files=()
for file in $(ls "$src_dir1"); do
    if [[ "$file" == *.png ]]; then
        files+=("$file")
    fi
done

for ((i=0; i<5 && i<${#files[@]}; i++)); do
    selected_file="${files[$i]}"
    cp "$src_dir1/$selected_file" "$target_folder_path"
done

total_files=${#files[@]}
for ((i=total_files-5; i<total_files && i>=0; i++)); do
    selected_file="${files[$i]}"
    cp "$src_dir1/$selected_file" "$target_folder_path"
done

echo "Selected files have been copied to $target_folder_path"