python main.py \
    --dataset_type syn1 \
    --dataset \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1b/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1c/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1d/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1e/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1f/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1g/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1h/" \
    --dataset_path \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1b/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1c/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1d/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1e/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1f/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1g/" \
    "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/data/v0_1h/" \
    --cls_token \
    --compress_class 2,3 \
    --compress_ratio 4 \
    --compress_shuffle \
    --pad_input \
    --smooth_class 6 \
    --smooth 10 \
    --vqvae_dir "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/stellar-yogurt-334/" \
    --vqvae_refl_dir "/home/randycm/Documents/Research/Transformers/testing/ElasticGPT/celestial-cloud-127/" \
    --patience 10 \
    --wandb_log \