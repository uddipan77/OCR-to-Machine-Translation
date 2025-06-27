from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj=r"pytorch_model.bin",  # or pytorch_model.bin
    path_in_repo="pytorch_model.bin",                   # How it will be named on HF
    repo_id="Uddipan107/ocr-layoutlmv3-base-t5-small",
    repo_type="model"
)
