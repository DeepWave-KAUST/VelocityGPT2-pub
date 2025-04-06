import argparse
import os
import torch
import wandb
from velocitygpt.utils import set_seed, setup, get_git_info, count_parameters, save_all, set_mpl_params
from velocitygpt.pipeline import *
from velocitygpt.datasets import *
from velocitygpt.modules import *
from velocitygpt.quantizer import *
from velocitygpt.train import run_velenc, run_velgen, run_velup
from velocitygpt.vis import plot_example, plot_example2, plot_example3
from pathlib import Path

def parse_list_of_lists(arg_value):
    # Assume the input is given in a specific format, like "1,2,3;4,5,6;7,8"
    return [list(map(int, group.split(','))) for group in arg_value.split(';')]

def parse_list_of_lists2(arg_value):
    # Assume the input is given in a specific format, like "1,2,3;4,5,6;7,8"
    return [list(map(str, group.split(','))) for group in arg_value.split(';')]

def parse_range(value):
    range_list = []
    for part in value.split(','):
        if '-' in part:
            start_end = part.split('-')
            start = int(start_end[0])
            if ':' in start_end[1]:
                end, step = map(int, start_end[1].split(':'))
            else:
                end = int(start_end[1])
                step = 1
            range_list.extend(range(start, end + 1, step))
        else:
            range_list.append(int(part))
    return range_list

def parse_args():
    parser = argparse.ArgumentParser(description="StorSeismic Fine-tune Denoising")
    # Data parameters
    parser.add_argument('--dataset', type=parse_list_of_lists2, required=True)
    parser.add_argument('--prop', type=float, default=1)
    parser.add_argument('--train_prop', type=float, default=None)
    parser.add_argument('--dataset_path', type=parse_list_of_lists2, required=True)
    parser.add_argument('--dataset_type', type=str, default='syn1')
    parser.add_argument('--image_size', type=int, nargs=2, default=[64, 64])
    parser.add_argument('--orig_image_size', type=int, nargs=2, default=[64, 64])
    parser.add_argument('--patch_size', type=int, nargs=2, default=[4, 4])
    parser.add_argument('--stride', type=int, nargs=2, default=[4, 4])
    parser.add_argument('--input_type', type=str, default='img')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--norm_mode', type=str, nargs='+', default=['independent'])
    parser.add_argument('--norm_const', type=float, nargs='+', default=[4500])
    parser.add_argument('--norm_level', type=str, default='sample')
    parser.add_argument('--dt0', type=float, default=0.016)
    parser.add_argument('--freq', type=float, default=20)
    parser.add_argument('--nt0', type=int, default=64)
    parser.add_argument('--ntwav', type=int, default=63)
    parser.add_argument('--transform_gaussian_kernel', type=int, nargs=2, default=[0, 0])
    parser.add_argument('--transform_gaussian_sigma', type=float, nargs='+', default=[0, 0])

    # VQVAE Model parameters
    parser.add_argument('--vq_type', type=str, default='vqvae')
    parser.add_argument('--quantizer', type=str, default='vq')
    parser.add_argument('--quantizer_entropy_weight', type=float, default=0.1)
    parser.add_argument('--quantizer_levels', type=int, nargs='+', default=[8, 6, 5])
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--input_rand_mask', action='store_true')
    parser.add_argument('--intermediate_dim', type=int, default=64)
    parser.add_argument('--K', type=int, default=128)
    parser.add_argument('--add_input_latent_noise', action='store_true')
    parser.add_argument('--input_noise_factor', type=float, default=0.1)
    parser.add_argument('--latent_noise_factor', type=float, default=0.1)

    # GPT Model parameters
    parser.add_argument('--latent_dim', type=int, nargs=2, default=[16, 16])
    parser.add_argument('--vocab_size', type=int, default=128)
    parser.add_argument('--refl_vocab_size', type=int, default=128)
    parser.add_argument('--type_vocab_size', type=int, default=2)
    parser.add_argument('--attn_type', type=str, default='default')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--intermediate_size', type=int, default=1024)
    parser.add_argument('--num_hidden_ffn', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--pre_ln', action='store_true', default=True)
    parser.add_argument('--cls_token', action='store_true')
    parser.add_argument('--max_position_embeddings', type=int, default=274)
    parser.add_argument('--position_embedding_type', type=str, default='learnable')
    parser.add_argument('--fixed_slopes', action='store_true')
    parser.add_argument('--embedding_type', type=str, default='none')
    parser.add_argument('--n_concat_token', type=int, default=1)
    parser.add_argument('--double_pos', action='store_true')
    parser.add_argument('--output_scores', action='store_true')
    parser.add_argument('--output_attentions', action='store_true')
    parser.add_argument('--output_hidden_states', action='store_true')
    parser.add_argument('--flip_train', action='store_true')
    parser.add_argument('--flip_train_inv', action='store_true')
    parser.add_argument('--use_dip', action='store_true')
    parser.add_argument('--use_dip_prob', type=float, default=0)
    parser.add_argument('--glob_pos_mode', type=str, default=None)
    parser.add_argument('--use_refl_prob', type=float, default=0.9)
    parser.add_argument('--well_cond_prob', type=float, default=0.9)
    parser.add_argument('--add_dip_to_well', action='store_true')
    parser.add_argument('--prepend_refl', action='store_true')
    parser.add_argument('--use_init_prob', type=float, default=0)

    # SR model parameters
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--n_resblocks', type=int, default=8)
    parser.add_argument('--n_feats', type=int, default=128)
    parser.add_argument('--res_scale', type=int, default=1)

    # Training parameters
    parser.add_argument('--training_stage', type=str, required=True)
    parser.add_argument('--parent_dir', type=str, default='.', help="Saving directory")
    parser.add_argument('--cont_dir', type=str, default=None, help="Directory to continue training from")
    parser.add_argument('--lr', type=float, default=0.0025)
    parser.add_argument('--lr_min', type=float, default=0.000001)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='none')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--warmup', type=str, default='none')
    parser.add_argument('--warmup_period', type=int, default=20)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--label_smoothing', type=float, default=0)
    parser.add_argument('--scheduled_sampling_c', type=float, default=0.005)
    parser.add_argument('--scheduled_sampling_k', type=int, default=1)
    parser.add_argument('--scheduled_sampling_decay', type=str, default='linear')
    parser.add_argument('--scheduled_sampling_limit', type=int, default=0)
    parser.add_argument('--sampling_type', type=str, default='teacher_forcing')
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--loss', type=str, default='crossentropy')
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--recon_loss_fn', type=str, default='l2')
    parser.add_argument('--use_focal_loss', action='store_true')

    # Misc parameters
    parser.add_argument('--device', type=str, default='cuda')
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        cuda_idx = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        cuda_idx = 0
    parser.add_argument('--cuda_idx', default=cuda_idx)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--aug_flip', action='store_true')
    parser.add_argument('--revert', action='store_true')
    parser.add_argument('--pad_input', action='store_true')
    parser.add_argument('--add_pos_first', action='store_true')
    parser.add_argument('--compress_shuffle', action='store_true')
    parser.add_argument('--compress_class', type=parse_list_of_lists, default=[])
    parser.add_argument('--compress_ratio', nargs='+', type=int, default=[])
    parser.add_argument('--smooth_class', type=parse_list_of_lists, default=[])
    parser.add_argument('--smooth', nargs='+', type=float, default=[])
    parser.add_argument('--dip_bins', type=parse_range, default=[])
    parser.add_argument('--scaler2', type=int, nargs='+', default=[2])
    parser.add_argument('--scaler3', type=float, nargs='+', default=[0.5])
    parser.add_argument('--vqvae_dir', type=str, default=None)
    parser.add_argument('--vqvae_refl_dir', type=None, default=None)
    parser.add_argument('--anti_aliasing', action='store_true')
    parser.add_argument('--order', type=int, default=0)

    # WandB parameters
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--wandb_notes', default=None)
    parser.add_argument('--wandb_job_type', default=None)
    parser.add_argument('--wandb_group', default=None)
    parser.add_argument('--wandb_tags', default=None)
    parser.add_argument('--wandb_id', type=str, default=None)
    
    return parser.parse_args()

def main(args):
    set_mpl_params()
    setup(args)
    train_data, test_data, scaler1, pad = load_and_prep(args)
    print("No. of training data samples: ", len(train_data))
    print("No. of test data samples: ", len(test_data))
    train_dataloader, test_dataloader = build_dataloader(args, train_data, test_data)
    if "vqvae" not in args.training_stage and "sr" not in args.training_stage:
        vqvae_model = load_model(args)
        if args.vqvae_refl_dir is not None:
            vqvae_refl_model = load_model(args, model_type="refl")
        else:
            vqvae_refl_model = None
    model = build_model(args)
    if args.wandb_log:
        wandb.watch(model, log_freq=1)
    total_params = count_parameters(model)
    print(total_params)
    optim = build_optimizer(args, model)
    warmup, scheduler = build_warmup_and_scheduler(args, optim)
    loss_fn = build_loss_fn(args)

    if "vqvae" in args.training_stage:
        model, avg_train_loss, avg_valid_loss, time_per_epoch = \
            run_velenc(model, optim, warmup, scheduler, loss_fn, train_dataloader, test_dataloader, scaler1,
                    args, verbose=False)
        idx_dict = {'syn1': [0], 'fld1': [0], 'fld2': [100], 'syn2': [0]}
        plot_example2(model, train_data, scaler1[0], pad, args, idx_dict[args.dataset_type], log=args.wandb_log, 
                      prefix=1)
        plot_example2(model, test_data, scaler1[1], pad, args, idx_dict[args.dataset_type], log=args.wandb_log, 
                      prefix=2)
    elif "sr" in args.training_stage:
        model, avg_train_loss, avg_valid_loss, time_per_epoch = \
            run_velup(model, optim, warmup, scheduler, loss_fn, train_dataloader, test_dataloader, scaler1,
                    args, verbose=False)
        plot_example3(model, train_data, scaler1[0], pad, args, [0], log=args.wandb_log, prefix=1)
        plot_example3(model, test_data, scaler1[1], pad, args, [0], log=args.wandb_log, prefix=2)
    else:
        model, avg_train_loss, avg_valid_loss, time_per_epoch = \
            run_velgen(model, vqvae_model, vqvae_refl_model, optim, warmup, scheduler, loss_fn, train_dataloader, 
                    test_dataloader, scaler1, args, verbose=False)
    
        plot_example(vqvae_model, vqvae_refl_model, model, train_data, scaler1[0], pad, args, [0], 
                        idx_gen=[35], log=args.wandb_log, prefix=1)
        plot_example(vqvae_model, vqvae_refl_model, model, test_data, scaler1[1], pad, args, [0], 
                        idx_gen=[35], log=args.wandb_log, prefix=2)
                        
    if args.wandb_log:
        wandb.log({"total_params" : total_params})
        wandb.unwatch()
    save_all(model, avg_train_loss, avg_valid_loss, time_per_epoch, args, optim, scheduler)

if __name__ == "__main__":
    args = parse_args()
    args.max_position_embeddings = args.max_length + \
                                (args.latent_dim[1] + 1) * math.ceil(args.well_cond_prob) + \
                                args.use_dip + \
                                int(0 if args.vqvae_refl_dir is None else 1) + \
                                int(args.max_length if args.prepend_refl else 0) + \
                                ((args.max_length + 1) if args.use_init_prob else 0)
                                
    if args.dataset_type not in ['fld2', 'syn2']:
        args.dataset = args.dataset[0]
        args.dataset_path = args.dataset_path[0]
        args.scaler2 = args.scaler2[0]
        args.scaler3 = args.scaler3[0]
        args.norm_const = args.norm_const[0]
    if args.wandb_log:
        wandb.login()
        wandb.init(config=args, project='ElasticGPT', 
                   notes=args.wandb_notes, 
                   job_type=args.wandb_job_type, 
                   group=args.wandb_group,
                   tags=args.wandb_tags,
                   id=args.wandb_id,
                   resume='allow' if args.wandb_id else None)
        args.parent_dir = os.path.join(args.parent_dir, wandb.run.name)

        # Get Git commit and branch
        git_commit, git_branch = get_git_info()
        # Log Git information to W&B
        wandb.config.update({
            "git_commit": git_commit,
            "git_branch": git_branch
        }, allow_val_change=True)
    if args.parent_dir:
        Path(args.parent_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    if args.wandb_log:
        wandb.finish()