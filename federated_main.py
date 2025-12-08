import warnings
warnings.filterwarnings(
    "ignore",
    message="The default value of the antialias parameter",
    category=UserWarning,
    module="torchvision.transforms.functional"
)


import argparse
import torch
import os
from Dassl.dassl.utils import setup_logger, set_random_seed, collect_env_info
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
import time

import copy
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.fed_utils import average_weights, count_parameters

def _normalize_ctx_payload(prompt_learner, payload):
    """
    Accepts: dict with 'ctx', raw tensor, list/tuple (possibly nested)
    Returns: dict with a proper Tensor under 'ctx' on correct device/dtype/shape,
             or None if the input payload resolves to an empty tensor/list.
    """
    device = prompt_learner.ctx.device
    dtype  = prompt_learner.ctx.dtype
    n_ctx  = prompt_learner.n_ctx  # Prompt length
    
    # Handle case where prompt_learner.ctx might not be initialized yet
    try:
        dim = prompt_learner.ctx.shape[-1] # Get dim from the learner
    except AttributeError:
        # Fallback or error if ctx is not a tensor (e.g., None)
        print("Error: prompt_learner.ctx is not initialized or not a tensor.")
        return None

    ctx = None # Initialize ctx

    # --- Extract potential tensor/list from payload ---
    if isinstance(payload, torch.Tensor):
        if payload.numel() > 0: # Check if tensor is not empty
            ctx = payload.to(device=device, dtype=dtype)
    elif isinstance(payload, dict):
        ctx_data = payload.get('ctx', None)
        if isinstance(ctx_data, torch.Tensor):
            if ctx_data.numel() > 0:
                 ctx = ctx_data.to(device=device, dtype=dtype)
        elif isinstance(ctx_data, (list, tuple)) and ctx_data: # Check if list/tuple is not empty
            try:
                ctx = torch.tensor(ctx_data, dtype=dtype, device=device)
                if ctx.numel() == 0: # Double-check tensor creation didn't result in empty
                    ctx = None
            except Exception as e:
                print(f"Error converting list/tuple to tensor in _normalize_ctx_payload: {e}")
                return None # Return None on conversion error
        # else: ctx remains None if ctx_data is None, empty list/tuple, or other type
    elif isinstance(payload, (list, tuple)) and payload: # Check if list/tuple is not empty
        try:
            ctx = torch.tensor(payload, dtype=dtype, device=device)
            if ctx.numel() == 0: # Double-check
                 ctx = None
        except Exception as e:
            print(f"Error converting list/tuple payload to tensor in _normalize_ctx_payload: {e}")
            return None # Return None on conversion error
    # else: ctx remains None if payload is None, empty list/tuple, or other type

    # --- If ctx is None after extraction, return None ---
    if ctx is None:
        print("Warning: _normalize_ctx_payload received or resulted in empty ctx.")
        return None

    # --- Reshape if necessary ---
    try:
        expected_shape = (n_ctx, dim)
        if ctx.shape == expected_shape:
             pass # Shape is already correct
        elif ctx.dim() == 1:
            # assume flattened; infer dim from current parameter
            ctx = ctx.view(n_ctx, dim)
        elif ctx.dim() == 3 and ctx.shape[0] == 1:
            # e.g., (1, n_ctx, dim) -> (n_ctx, dim)
            ctx = ctx.squeeze(0)
        
        # Final shape check
        if ctx.shape != expected_shape:
             raise ValueError(f"Final ctx shape {tuple(ctx.shape)} does not match expected ({n_ctx}, {dim})")

    except Exception as e:
        print(f"Error during reshaping in _normalize_ctx_payload: {e}. Ctx shape was {tuple(ctx.shape) if ctx is not None else 'None'}")
        return None # Return None if reshaping fails

    return {'ctx': ctx}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pFedMoAP", help="model of aggregation, choose from: pFedMoAP (used with pFedMoAP), fedavg, fedprox, local(The last three are used with PromptFL)")
    parser.add_argument("--trainer", type=str, default="PFEDMOAP", help="name of trainer, choose from: CLIP (used with fedavg), PromptFL, PFEDMOAP")
    parser.add_argument('--round', type=int, default=10, help="number of communication round")
    parser.add_argument('--local_epochs', type=int, default=5, help="number of local epochs")
    parser.add_argument('--num_users', type=int, default=10, help="number of users")
    parser.add_argument('--frac', type=float, default=1.0, help='the client sample ratio: C')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma of single_step (learning rate decay multiplier)')
    parser.add_argument('--train_batch_size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=100, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument('--mu', type=float, default=0.5, help='The parameter for fedprox')

    # caltech101, oxford_flowers, oxford_pets, food101 and dtd
    parser.add_argument('--iid', default=False, help="is iid, control the iid of caltech101, oxford_flowers, oxford_pets, food101 and dtd")
    parser.add_argument('--num_shots', type=int, default=16, help="number of shots in few shot setting")
    parser.add_argument('--useall', default=False, help="is useall, True for training all samples, False for few shot learning")
    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir100', help='the data partitioning strategy of cifar10 and cifar100, select from "homo, noniid-labeluni, noniid-labeldir,noniid-labeldir100"')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    # domainnet, office
    parser.add_argument('--imbalance_train', default=False, help="is adding label skew to feature skew datasets")
    parser.add_argument('--split_client', default=False, help="is adding label skew to feature skew datasets and split one domain to multi clients")
    parser.add_argument('--num_domain', type=int, default=4, help="number of domain")

    # parameters of learnable prompts
    parser.add_argument('--n_ctx', type=int, default=16, help="number of text encoder of text prompts")
    parser.add_argument('--ctx_init', default=False, help="is using the ctx init, set True for CLIP")

    # parameters of pFedMoAP
    parser.add_argument("--num_experts", type=int, default=10, help="number of experts")
    parser.add_argument("--sparse_selection", type=str, default="nearest", choices=["nearest", "random"], help="type of expert selection, choose between random and nearest")
    parser.add_argument("--gating_heads", type=int, default=8, help="number of heads in gating network")
    parser.add_argument("--gating_embed_dim", type=int, default=128, help="number of heads in gating network")
    parser.add_argument("--lmbda", type=float, default=0.5, help="the coefficient of the local output loss")
    parser.add_argument("--scaling", type=float, default=10.0, help="the scaling factor in attention dot product for attention weights")

    # parameters of path
    parser.add_argument("--logdir", type=str, required=False, default="./logs/", help="Log directory path")
    parser.add_argument("--root", type=str, default="./dataset/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="./output/food101/pFedMoAP/rn50_16shots/prompt16/10users_10rounds_seed1", help="output directory")
    parser.add_argument("--config-file", type=str, default="./configs/trainers/PFEDMOAP/rn50.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="./configs/datasets/food101.yaml", help="path to config file for dataset setup")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")

    args = parser.parse_args()
    return args


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.round:
        cfg.OPTIM.ROUND = args.round # global round
        
    if args.local_epochs:
        cfg.OPTIM.MAX_EPOCH = args.local_epochs

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    """
    Add new config variables.
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.PROMPTFL.CSC = False  # class-specific context
    cfg.TRAINER.PROMPTFL.CTX_INIT = args.ctx_init  # initialization words
    cfg.TRAINER.PROMPTFL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # Config for PFEDMOAP
    cfg.TRAINER.PFEDMOAP = CN()
    cfg.TRAINER.PFEDMOAP.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.PFEDMOAP.CSC = False  # class-specific context
    cfg.TRAINER.PFEDMOAP.CTX_INIT = args.ctx_init  # initialization words
    cfg.TRAINER.PFEDMOAP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PFEDMOAP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PFEDMOAP.NUM_EXPERTS = min(args.num_experts, args.num_users) # number of experts
    cfg.TRAINER.PFEDMOAP.GATING_HEADS = args.gating_heads # number of heads in gating network
    cfg.TRAINER.PFEDMOAP.GATING_EMBED_DIM = args.gating_embed_dim # embedding dimension in gating network
    cfg.TRAINER.PFEDMOAP.LMBDA = args.lmbda # number of heads in gating network
    cfg.TRAINER.PFEDMOAP.SCALING = args.scaling # scaling of the distance matrix

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.USERS = args.num_users  # number of clients
    cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.PARTITION = args.partition
    cfg.DATASET.USEALL = args.useall # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0 # repeat rate on each client
    cfg.DATASET.IMBALANCE_TRAIN = args.imbalance_train # is adding label skew to feature skew datasets
    cfg.DATASET.SPLIT_CLIENT = args.split_client # is adding label skew to feature skew datasets and split one domain to multi clientss

    cfg.DATALOADER.TRAIN_X.N_DOMAIN = args.num_domain # number of domain
    
    cfg.OPTIM.ROUND = args.round # global round
    cfg.OPTIM.MAX_EPOCH = args.local_epochs # local epoch
    cfg.OPTIM.GAMMA = args.gamma # gamma of single-step (learning rate decay multiplier)
    cfg.OPTIM.LR = args.lr #learning rate

    # cfg.OPTIMGATING = cfg.OPTIM.clone() # optimizer for pFedMoAP

    cfg.MODEL.BACKBONE.PRETRAINED = True

    cfg.TEST.NO_TEST = True



def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    cfg.set_new_allowed(True)
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    
    # --- NEW: Ensure CHECKPOINT_FREQ is set from opts ---
    if "TRAIN.CHECKPOINT_FREQ" in args.opts:
        cfg.TRAIN.CHECKPOINT_FREQ = int(args.opts[args.opts.index("TRAIN.CHECKPOINT_FREQ") + 1])


    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        # print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)

    local_weights = [[] for i in range(args.num_users)]
    local_gatings = [{} for i in range(args.num_users)]
    local_prompts = [[] for i in range(args.num_users)]
    global_prompt = None

    # local_trainer = build_trainer(cfg)
    local_trainer = build_trainer(cfg)
    if torch.cuda.device_count() > 1:
        print(f"🔁 Using DataParallel on {torch.cuda.device_count()} GPUs")
        local_trainer.model = torch.nn.DataParallel(local_trainer.model)
    local_trainer.fed_before_train()
    count_parameters(local_trainer.model,"prompt_learner")
    count_parameters(local_trainer.model, "image_encoder")
    count_parameters(local_trainer.model, "text_encoder")
    count_parameters(local_trainer.model, "gating")

    datanumber_client = []
    if args.trainer == 'CLIP':
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        for net_i in range(cfg.DATASET.USERS):
            datanumber_client.append(len(local_trainer.fed_train_loader_x_dict[net_i].dataset))
        global_weights = copy.deepcopy(local_trainer.model.state_dict())

    # Training
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    # global_trainer.before_train()
    global_test_acc_list = []
    global_test_error_list = []
    global_test_f1_list = []
    global_epoch_list = []
    global_time_list = []
    start = time.time()

    # --- LOAD CHECKPOINT LOGIC ---
    if cfg.RESUME:
        checkpoint_path = os.path.join(cfg.RESUME, 'checkpoint_latest.pth.tar')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                # --- FIX: ADD weights_only=False ---
                checkpoint = torch.load(checkpoint_path, weights_only=False) 
                
                start_epoch = checkpoint['epoch'] + 1
                local_prompts = checkpoint['local_prompts']
                local_gatings = checkpoint['local_gatings']
                global_prompt = checkpoint['global_prompt']
                global_test_acc_list = checkpoint.get('global_test_acc_list', [])
                global_test_error_list = checkpoint.get('global_test_error_list', [])
                global_test_f1_list = checkpoint.get('global_test_f1_list', [])
                global_epoch_list = checkpoint.get('global_epoch_list', [])
                global_time_list = checkpoint.get('global_time_list', [])
                start = time.time() - global_time_list[-1] if global_time_list else time.time()
                print(f"Resuming training from round {start_epoch}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                start_epoch = 0
        else:
            print(f"Warning: --resume specified but checkpoint_latest.pth.tar not found in {cfg.RESUME}. Starting from scratch.")
    # --- END NEW LOGIC ---

    def evaluate_trainer(results, mode="CLIP"):
        nonlocal global_time_list, global_test_acc_list, global_test_error_list, global_test_f1_list, global_epoch_list
        nonlocal start, max_epoch
        nonlocal cfg

        if mode == "CLIP" or mode == "local":
            condition = (epoch == max_epoch - 1)
        else:
            condition = (epoch >= 2)

        global_test_acc = []
        global_test_error = []
        global_test_f1 = []
        # --- FIX: Ensure results[k] is not None before indexing ---
        for k in range(len(results)):
            if results[k] is not None and len(results[k]) >= 3:
                global_test_acc.append(results[k][0])
                global_test_error.append(results[k][1])
                global_test_f1.append(results[k][2])
            else:
                # Handle cases where a client might not have results (e.g., skipped test)
                print(f"Warning: No valid results for client {k} in evaluate_trainer.")
                
        # --- FIX: Avoid division by zero if no clients had results ---
        if not global_test_acc:
            print("Warning: No client results to evaluate.")
            avg_acc = 0.0
            avg_error = 100.0
            avg_f1 = 0.0
        else:
            avg_acc = sum(global_test_acc) / len(global_test_acc)
            avg_error = sum(global_test_error) / len(global_test_error)
            avg_f1 = sum(global_test_f1) / len(global_test_f1)

        global_time_list.append(time.time() - start)
        global_test_acc_list.append(avg_acc)
        global_test_error_list.append(avg_error)
        global_test_f1_list.append(avg_f1)
        global_epoch_list.append(epoch)
        
        print(f"Global test acc: {avg_acc}")
        print(f"Global test error: {avg_error}")
        print(f"Global test macro_f1: {avg_f1}")

        if (cfg.DATASET.NAME == "DomainNet" or cfg.DATASET.NAME == "Office") and condition and args.split_client:
            domains = {"DomainNet":["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
                       "Office":["amazon", "caltech", "dslr", "webcam"]}
            num_domains = len(domains[cfg.DATASET.NAME])
            num_clients_per_domain = args.num_users // num_domains
            print("Test acc of clients:", global_test_acc)
            for i in range(num_domains):
                accs = global_test_acc[i*num_clients_per_domain:(i+1)*num_clients_per_domain]
                if accs: # Avoid errors on empty lists
                    print("Test acc of", domains[cfg.DATASET.NAME][i], np.mean(accs), "±", np.std(accs))
            if global_test_acc: # Avoid errors on empty lists
                print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
        print("------------local test finish-------------")

    # # --- NEW: FILTER TEST SET TO MATCH TRAIN CLASSES ---
    # print("\n--- FIXING PARTITION: Filtering Test Sets to match Train Classes ---")
    # try:
    #     for client_idx in range(cfg.DATASET.USERS):
    #         # 1. Identify Train Classes
    #         train_loader = local_trainer.fed_train_loader_x_dict[client_idx]
    #         train_labels = []
    #         for batch in train_loader:
    #              if isinstance(batch, dict) and "label" in batch:
    #                  train_labels.extend(batch["label"].tolist())
    #              else:
    #                  train_labels.extend(batch[1].tolist())
    #         valid_classes = set(train_labels)

    #         # 2. Filter Test Dataset
    #         test_loader = local_trainer.fed_test_loader_x_dict[client_idx]
    #         dataset = test_loader.dataset
            
    #         if hasattr(dataset, 'data_source'):
    #             original_count = len(dataset.data_source)
    #             # Filter the list (datum.label or item.label)
    #             new_data_source = []
    #             for item in dataset.data_source:
    #                 # Check label attribute (Datum object) or use item if it's simple
    #                 label = item.label if hasattr(item, 'label') else item
    #                 # Only keep if in valid classes
    #                 if label in valid_classes:
    #                     new_data_source.append(item)
                
    #             # Apply filter
    #             dataset.data_source = new_data_source
    #             new_count = len(dataset.data_source)
                
    #             if original_count != new_count:
    #                 print(f"Client {client_idx}: Removed {original_count - new_count} unseen test samples (Now {new_count}).")
    #         else:
    #             pass # No data_source to filter
    # except Exception as e:
    #     print(f"Partition fix failed: {e}")
    # print("--------------------------------------------------------------------\n")
    # # --- END FIX ---

    # # --- SANITY CHECK FOR TRAIN/TEST CLASSES (ALL CLIENTS) ---
    # print("\n--- STARTING SANITY CHECK: Data Distribution for ALL Clients ---")
    # clients_with_issues = []
    # try:
    #     # Iterate over all clients
    #     for check_client in range(cfg.DATASET.USERS):
    #         train_loader = local_trainer.fed_train_loader_x_dict[check_client]
    #         test_loader = local_trainer.fed_test_loader_x_dict[check_client]

    #         train_labels = []
    #         for batch in train_loader:
    #             if isinstance(batch, dict) and "label" in batch:
    #                 train_labels.extend(batch["label"].tolist())
    #             else:
    #                 # Fallback for standard torch loaders, assume tuple (img, label)
    #                 train_labels.extend(batch[1].tolist())

    #         test_labels = []
    #         for batch in test_loader:
    #             if isinstance(batch, dict) and "label" in batch:
    #                 test_labels.extend(batch["label"].tolist())
    #             else:
    #                 test_labels.extend(batch[1].tolist())

    #         train_classes = set(train_labels)
    #         test_classes = set(test_labels)
            
    #         # Check for classes in Test that are NOT in Train
    #         unseen = test_classes - train_classes
            
    #         if unseen:
    #             print(f"⚠️ Client {check_client}: Has {len(unseen)} UNSEEN test classes: {sorted(list(unseen))}")
    #             clients_with_issues.append(check_client)
            
    #         # Optional: Print progress every 10 clients to show it's working
    #         if check_client % 10 == 0:
    #             print(f"Checked Client {check_client}...")

    #     if not clients_with_issues:
    #         print("✅ SANITY CHECK PASSED: All clients are tested ONLY on classes present in their training set.")
    #     else:
    #         print(f"❌ SANITY CHECK FAILED: {len(clients_with_issues)} clients have data leakage/mismatch.")

    # except Exception as e:
    #     print(f"Sanity Check crashed: {e}")
    # print("------------------------------------------------------------\n")
    # # --- END SANITY CHECK ---

    for epoch in range(start_epoch, max_epoch):

        if args.trainer == 'CLIP':
            print("------------local test start-------------")
            results = []
            idxs_users = list(range(0, cfg.DATASET.USERS))
            local_trainer.model.load_state_dict(global_weights)
            for idx in idxs_users:
                # --- CHANGE: Pass new args ---
                results.append(local_trainer.test(idx=idx, epoch=epoch, output_dir=cfg.OUTPUT_DIR))
            evaluate_trainer(results, mode=args.trainer)
            print("Round on server :", epoch)
            break

        elif args.model == "fedavg":
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights, strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights[idx] = copy.deepcopy(local_weight)
            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights, idxs_users, datanumber_client)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))
            local_trainer.model.load_state_dict(global_weights, strict=False)
            local_weights = [[] for i in range(args.num_users)] 
            for idx in all_users:
                # --- CHANGE: Pass new args ---
                results.append(local_trainer.test(idx=idx, epoch=epoch, output_dir=cfg.OUTPUT_DIR))
            evaluate_trainer(results, mode=args.model)
            print("Round on server :", epoch)

        elif args.model == "fedprox":
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights, strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True, global_weight=global_weights, fedprox=True, mu=args.mu)
                local_weight = local_trainer.model.state_dict()
                local_weights[idx] = copy.deepcopy(local_weight)
            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights, idxs_users, datanumber_client)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))
            local_trainer.model.load_state_dict(global_weights, strict=False)
            local_weights = [[] for i in range(args.num_users)] 
            for idx in all_users:
                # --- CHANGE: Pass new args ---
                results.append(local_trainer.test(idx=idx, epoch=epoch, output_dir=cfg.OUTPUT_DIR))
            evaluate_trainer(results, mode=args.model)
            print("Round on server :", epoch)

        elif args.model == "pFedMoAP":
            num_client_selected = int(args.frac * args.num_users)
            m = max(num_client_selected, 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            
            results = [None for _ in range(cfg.DATASET.USERS)]
            all_client_preds_lists = [None] * cfg.DATASET.USERS
            all_client_labels_lists = [None] * cfg.DATASET.USERS

            all_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")

            for idx in idxs_users:
                # download
                if epoch == 0 and global_prompt is None: 
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    if local_prompts[idx] != []:
                        if local_gatings[idx] != {}: 
                            local_trainer.model.load_state_dict(local_gatings[idx], strict=False)
                        else:
                            local_trainer.model.load_state_dict(global_weights, strict=False)
                        
                        if local_prompts[idx] != [] or epoch > 0: 
                            selected_experts_indices = local_trainer.sparse_selection(idx, local_prompts)
                            print(f"Client {idx} selected experts: {selected_experts_indices}")
                            if selected_experts_indices: 
                                local_trainer.download_nonlocal_ctx([local_prompts[expert_idx] for expert_idx in selected_experts_indices])
                            else: 
                                local_trainer.download_nonlocal_ctx([])
                    else:
                        local_trainer.model.load_state_dict(global_weights, strict=False)
                    
                    if global_prompt is not None:
                        global_prompt_payload = _normalize_ctx_payload(local_trainer.model.prompt_learner, global_prompt)
                        if global_prompt_payload is not None:
                            local_trainer.model.load_state_dict({"prompt_learner.ctx": global_prompt_payload['ctx']}, strict=False)
                        else:
                            print(f"Warning: Could not normalize and load global_prompt for client {idx}")
                    elif local_prompts[idx] is not None and local_prompts[idx] != []:
                        local_prompt_payload = _normalize_ctx_payload(local_trainer.model.prompt_learner, local_prompts[idx])
                        if local_prompt_payload is not None:
                            local_trainer.model.load_state_dict({"prompt_learner.ctx": local_prompt_payload['ctx']}, strict=False)

                # train
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)

                # test selected clients for this round
                # --- CHANGE: Pass new args ---
                test_output = local_trainer.test(idx=idx, epoch=epoch, output_dir=cfg.OUTPUT_DIR)
                results[idx] = test_output[:3] # (acc, loss, f1)
                all_client_preds_lists[idx] = test_output[3] # preds
                all_client_labels_lists[idx] = test_output[4] # labels

                if results[idx] is not None:
                    current_accuracy = results[idx][0] / 100.0

                # upload
                local_weight = local_trainer.model.state_dict()
                if 'prompt_learner.ctx' in local_weight:
                    local_gatings[idx] = {name: copy.deepcopy(local_weight[name]) for name in local_weight if 'gating' in name}  # gating dict
                    local_prompts[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'])  # prompts
                else:
                    print(f"Warning: 'prompt_learner.ctx' not in local_weight for client {idx}. Skipping upload.")


            print("------------local train finish epoch:", epoch, "-------------")
            local_trainer.update_lr(["gating"])
            local_trainer.reset_distance_cache(update_indices=idxs_users)
            global_prompt = average_weights(local_prompts, idxs_users, datanumber_client, islist=True)

            print("------------local test start-------------")
            
            # test clients that are not selected
            for idx in all_users:
                if results[idx] is not None:
                    continue

                payload_dict = None 
                
                if local_gatings[idx] != {}: 
                    local_trainer.model.load_state_dict(local_gatings[idx], strict=False)
                    selected_experts = local_trainer.sparse_selection(idx, local_prompts)
                    if selected_experts:
                        local_trainer.download_nonlocal_ctx([local_prompts[iii] for iii in selected_experts])
                    else:
                        local_trainer.download_nonlocal_ctx([])

                    payload = local_prompts[idx]
                    payload_dict = _normalize_ctx_payload(local_trainer.model.prompt_learner, payload)

                    if payload_dict is not None:
                        local_trainer.model.load_ctx(payload_dict['ctx'])
                        # --- CHANGE: Pass new args ---
                        test_output = local_trainer.test(idx=idx, epoch=epoch, output_dir=cfg.OUTPUT_DIR)
                        results[idx] = test_output[:3]
                        all_client_preds_lists[idx] = test_output[3]
                        all_client_labels_lists[idx] = test_output[4]
                    else:
                        print(f"Skipping test for trained client {idx}: prompt normalization failed.")
                        results[idx] = [0.0, 100.0, 0.0, np.array([]), np.array([])]

                elif local_prompts[idx] != []: 
                    payload = local_prompts[idx]
                    payload_dict = _normalize_ctx_payload(local_trainer.model.prompt_learner, payload)

                    if payload_dict is not None:
                        local_trainer.model.load_ctx(payload_dict['ctx'])
                        local_trainer.download_nonlocal_ctx([])
                        # --- CHANGE: Pass new args ---
                        test_output = local_trainer.test(idx=idx, epoch=epoch, output_dir=cfg.OUTPUT_DIR)
                        results[idx] = test_output[:3]
                        all_client_preds_lists[idx] = test_output[3]
                        all_client_labels_lists[idx] = test_output[4]
                    else:
                        print(f"Skipping test for trained client {idx}: prompt normalization failed.")
                        results[idx] = [0.0, 100.0, 0.0, np.array([]), np.array([])]

                else: 
                    if global_prompt is not None: 
                        payload = global_prompt
                        payload_dict = _normalize_ctx_payload(local_trainer.model.prompt_learner, payload)

                    if payload_dict is not None:
                        local_trainer.model.load_ctx(payload_dict['ctx'])
                        local_trainer.download_nonlocal_ctx([])
                        # --- CHANGE: Pass new args ---
                        test_output = local_trainer.test(idx=idx, epoch=epoch, output_dir=cfg.OUTPUT_DIR)
                        results[idx] = test_output[:3]
                        all_client_preds_lists[idx] = test_output[3]
                        all_client_labels_lists[idx] = test_output[4]
                    else:
                        print(f"Skipping test for untrained client {idx}: global_prompt unavailable or invalid.")
                        results[idx] = [0.0, 100.0, 0.0, np.array([]), np.array([])]

            
            evaluate_trainer(results, mode=args.model)
            
            # --- GLOBAL CM LOGIC (with file saving) ---
            if epoch % 2 == 0:
                print(f"\n--- Calculating Global Confusion Matrix at Round: {epoch} ---")
                global_all_preds = []
                global_all_labels = []
                # Use all_users to get predictions from everyone
                for i in all_users:
                    if all_client_preds_lists[i] is not None and all_client_labels_lists[i] is not None:
                        global_all_preds.append(all_client_preds_lists[i])
                        global_all_labels.append(all_client_labels_lists[i])
                
                if global_all_preds:
                    try:
                        global_all_preds = np.concatenate(global_all_preds)
                        global_all_labels = np.concatenate(global_all_labels)
                        
                        # --- FIX: Ensure Global CM is 100x100 ---
                        # Get all class names from dataset to know total count
                        all_classnames = local_trainer.dm.dataset.classnames
                        num_classes = len(all_classnames)
                        all_possible_labels = np.arange(num_classes)
                        
                        global_cm = confusion_matrix(global_all_labels, global_all_preds, labels=all_possible_labels)
                        
                        # --- NEW: Save Global CM to file ---
                        try:
                            cm_dir = os.path.join(cfg.OUTPUT_DIR, 'confusion_matrices', 'global')
                            os.makedirs(cm_dir, exist_ok=True)
                            
                            cm_filename = os.path.join(cm_dir, f'global_cm_round_{epoch}.npy')
                            np.save(cm_filename, global_cm)
                            
                            # Save labels too (though they are just 0..99)
                            labels_filename = os.path.join(cm_dir, f'global_labels_round_{epoch}.npy')
                            np.save(labels_filename, np.array(all_classnames))

                            print(f"--- GLOBAL CM (Round {epoch}): Saved to {cm_filename} (Shape: {global_cm.shape}) ---")
                            print("-------------------------------------")
                        except Exception as e:
                            print(f"--- GLOBAL CM (Round {epoch}): FAILED to save. Error: {e} ---")
                            print("-------------------------------------")
                        # --- END NEW SAVE LOGIC ---
                        
                    except Exception as e:
                        print(f"Could not compute global confusion matrix: {e}")
                else:
                    print("No predictions collected for global confusion matrix.")
            # --- END GLOBAL CM LOGIC ---
            
            # --- NEW: SAVE CHECKPOINT LOGIC ---
            if cfg.TRAIN.CHECKPOINT_FREQ > 0 and (epoch + 1) % cfg.TRAIN.CHECKPOINT_FREQ == 0:
                state = {
                    'epoch': epoch,
                    'local_prompts': local_prompts,
                    'local_gatings': local_gatings,
                    'global_prompt': global_prompt,
                    'global_test_acc_list': global_test_acc_list,
                    'global_test_error_list': global_test_error_list,
                    'global_test_f1_list': global_test_f1_list,
                    'global_epoch_list': global_epoch_list,
                    'global_time_list': global_time_list
                }
                
                # Save a round-specific checkpoint
                save_path = os.path.join(cfg.OUTPUT_DIR, f'checkpoint_round_{epoch}.pth.tar')
                torch.save(state, save_path)
                print(f"Saved checkpoint: {save_path}")
                
                # Overwrite the 'latest' checkpoint for easy resume
                latest_path = os.path.join(cfg.OUTPUT_DIR, 'checkpoint_latest.pth.tar')
                torch.save(state, latest_path)
                print(f"Updated latest checkpoint: {latest_path}")
            # --- END NEW LOGIC ---

            print("Round on server :", epoch)

        elif args.model == "local":
            idxs_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            results = []
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                # --- CHANGE: Pass new args ---
                results.append(local_trainer.test(idx=idx, epoch=epoch, output_dir=cfg.OUTPUT_DIR))
            evaluate_trainer(results, mode=args.model)
            break
        
        else:
            raise NotImplementedError(f"Model '{args.model}' is not implemented.")
    
    if 'idxs_users' in locals():
        for idx in idxs_users:
            local_trainer.fed_after_train()
    else:
        print("Skipping final fed_after_train(), idxs_users not defined.")

    print("global_test_acc_list:",global_test_acc_list)
    if global_test_acc_list: 
        print("maximum test acc:", max(global_test_acc_list))
        print("mean of acc:",np.mean(global_test_acc_list[-5:]))
        print("std of acc:",np.std(global_test_acc_list[-5:]))

if __name__ == "__main__":
    args = get_args()
    main(args)