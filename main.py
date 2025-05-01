import argparse
from models import create_vision_transformer
from datasets import build_continual_dataloader
from engine import *
import warnings

warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')


def main(args):
    # fix the seed for reproducibility
    seed = args.seed
    jt.set_global_seed(seed)

    data_loader, class_mask = build_continual_dataloader(args)

    jt.flags.use_cuda = 1

    print(f"Jittor use_cuda flag set to: {jt.flags.use_cuda}")

    if jt.flags.use_cuda:
        print("CUDA is enabled. Proceeding with GPU operations.")

    print(f"Creating original model: {args.model}")
    original_model = create_vision_transformer(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
    )

    print(f"Creating model: {args.model}")
    model = create_vision_transformer(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        prompt_len=args.length,
        prompt_init=args.prompt_key_init,
        use_prompt_pool=args.prompt_pool,
        use_prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
    )

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.stop_grad()

        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.stop_grad()
            else:
                print(n)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.jt'.format(task_id + 1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = jt.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader,
                                  task_id, class_mask, acc_matrix, args=args, )


            print("Current accuracy matrix after evaluating task {}:".format(task_id + 1))
            print(acc_matrix)

            save_path = os.path.join(args.output_dir, f'accuracy_matrix_after_task_{task_id + 1}.npy')
            try:
                np.save(save_path, acc_matrix)
                print(f"Accuracy matrix snapshot saved to: {save_path}")
            except Exception as e:
                print(f"Error saving accuracy matrix to {save_path}: {e}")

        return

    n_parameters = sum(p.numel() for p in model.parameters() if not p.is_stop_grad())
    n_parameters += sum(p.numel() for p in original_model.parameters() if not p.is_stop_grad())
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    optimizer = jt.optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = None

    criterion = jt.nn.CrossEntropyLoss()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, original_model,
                       criterion, data_loader, optimizer, lr_scheduler,
                       class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('L2P training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser

        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 L2P configs')
    # elif config == 'five_datasets_l2p':
    #     from configs.five_datasets_l2p import get_args_parser
    #
    #     config_parser = subparser.add_parser('five_datasets_l2p', help='5-Datasets L2P configs')
    else:
        raise NotImplementedError

    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)
