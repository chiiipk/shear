from huggingface_hub import upload_folder

repo_id = "minhchuxuan/bge_pruned_298"

# Upload experiments_high → HF under subfolder exp_high/
upload_folder(
    repo_id=repo_id,
    folder_path="./experiments_bad/production_hf",
    path_in_repo="exp_bad",
    commit_message="Upload experiments_high model"
)

# hf
# _IFFcrUmIo
# IhnochFGkPwVHPfnQthVSOpfv

#pip install huggingface_hub
#huggingface-cli login