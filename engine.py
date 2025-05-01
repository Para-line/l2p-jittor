import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
import time

import jittor
import jittor as jt

import numpy as np

from jt_extension import accuracy, index_fill_inplace
from torch.utils.tensorboard import SummaryWriter

import utils


def train_one_epoch(model: jt.nn.Module, original_model: jt.nn.Module,
                    criterion, data_loader: Iterable, optimizer: jt.optim.Optimizer,
                    epoch: int, max_norm: float = 0, task_id=-1, class_mask=None, writer: SummaryWriter = None,
                    global_step: int = 0, prompt_counts=None, args=None, ):
    model.train()
    original_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    if data_loader.total_len is None:
        data_loader.total_len = len(data_loader)

    num_steps_per_epoch = data_loader.__batch_len__()

    for idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        current_global_step = global_step + idx

        with jt.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

        output = model(input, task_id=task_id, cls_features=cls_features, train=True)
        logits = output['logits']

        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = jittor.array(not_mask, dtype='int64')
            index_fill_inplace(logits, dim=1, index=not_mask, value=float('-1e36'))

        if 'selected_indices' in output:
            selected_indices = output['selected_indices']
            if selected_indices is not None:
                selected_indices_np = selected_indices.numpy().flatten().astype(np.int64)

                counts_in_batch = np.bincount(selected_indices_np, minlength=prompt_counts.shape[0])

                if counts_in_batch.shape[0] == prompt_counts.shape[0]:
                    prompt_counts += counts_in_batch

        loss = criterion(logits, target)

        if args.pull_constraint and 'pull_loss' in output:
            loss = loss - args.pull_constraint_coeff * output['pull_loss']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        print(f"current_step: {current_global_step}")
        print(f"Acc@1: {acc1.item()}")
        print(f"Acc@5: {acc5.item()}")

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.clip_grad_norm(max_norm)
        optimizer.step()

        jt.sync_all(True)

        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0].get('lr', optimizer.lr))
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

        if writer is not None:
            writer.add_scalar(f'Train/Task_{task_id + 1}/Loss', loss.item(), current_global_step)
            if args.pull_constraint and 'pull_loss' in output:
                writer.add_scalar(f'Train/Task_{task_id + 1}/Pull Loss', output['pull_loss'].item(),
                                  current_global_step)
            writer.add_scalar(f'Train/Task_{task_id + 1}/Acc@1', acc1.item(), current_global_step)
            writer.add_scalar(f'Train/Task_{task_id + 1}/Acc@5', acc5.item(), current_global_step)
            writer.add_scalar(f'Train/Task_{task_id + 1}/iter_time', metric_logger.last_iter_time, current_global_step)

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step + num_steps_per_epoch


def evaluate(model: jt.nn.Module, original_model: jt.nn.Module, data_loader,
             task_id=-1, class_mask=None, args=None, ):
    criterion = jt.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with jt.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):

            # compute output
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[task_id]
                mask = jt.array(mask, dtype='int64')
                logits_mask = jt.ones_like(logits) * float('-inf')

                # logits_mask = jt.index_fill_(logits_mask, 1, mask, 0.0)
                index_fill_inplace(logits_mask, 1, mask, 0.0)

                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                  losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_till_now(model: jt.nn.Module, original_model: jt.nn.Module, data_loader,
                      task_id=-1, class_mask=None, acc_matrix=None, writer: SummaryWriter = None, global_step: int = 0,
                      args=None, ):
    with jt.no_grad():
        stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

        for i in range(task_id + 1):
            test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                                  task_id=i, class_mask=class_mask, args=args)

            jt.sync_all(True)

            stat_matrix[0, i] = test_stats['Acc@1']
            stat_matrix[1, i] = test_stats['Acc@5']
            stat_matrix[2, i] = test_stats['Loss']

            acc_matrix[i, task_id] = test_stats['Acc@1']

            if writer is not None:
                writer.add_scalar(f'Eval/Task_{i + 1}_Acc@1', test_stats['Acc@1'], global_step)
                writer.add_scalar(f'Eval/Task_{i + 1}_Acc@5', test_stats['Acc@5'], global_step)
                writer.add_scalar(f'Eval/Task_{i + 1}_Loss', test_stats['Loss'], global_step)

        avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

        diagonal = np.diag(acc_matrix)

        result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id + 1,
                                                                                                         avg_stat[0],
                                                                                                         avg_stat[1],
                                                                                                         avg_stat[2])
        if writer is not None:
            writer.add_scalar('Eval/Avg_Acc@1', avg_stat[0], global_step)
            writer.add_scalar('Eval/Avg_Acc@5', avg_stat[1], global_step)
            writer.add_scalar('Eval/Avg_Loss', avg_stat[2], global_step)

        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) -
                                  acc_matrix[:, task_id])[:task_id])
            backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

            result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)

            if writer is not None:
                writer.add_scalar('Eval/Avg_Forgetting', forgetting, task_id + 1)
                writer.add_scalar('Eval/Avg_BackwardTransfer', backward, task_id + 1)

        print(result_str)

    return test_stats


def train_and_evaluate(model: jt.nn.Module, original_model: jt.nn.Module,
                       criterion, data_loader: Iterable, optimizer: jt.optim.Optimizer, lr_scheduler,
                       class_mask=None, args=None, ):
    # --- TensorBoard setup---
    writer = None
    if args.tensorboard_dir:
        tensorboard_dir = args.tensorboard_dir
        run_name = f"L2P_jittor_{args.dataset}_{args.model}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log_dir = os.path.join(tensorboard_dir, run_name)

        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging enabled. Logs will be saved to: {log_dir}")
        except Exception as e:
            print(f"Error creating TensorBoard log directory or SummaryWriter: {e}")
            print("TensorBoard logging disabled.")
            writer = None
    else:
        print("No output directory specified, TensorBoard logging disabled.")
    # --- End TensorBoard Setup ---

    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    global_step = 1
    global_epoch = 0

    cumulative_times_list = []
    per_epoch_times_list = []

    start_time = time.time()

    for task_id in range(args.num_tasks):
        prompt_counts = None
        if args.prompt_pool:
            pool_size = args.size
            if hasattr(model, 'prompt') and hasattr(model.prompt, 'pool_size'):
                pool_size = model.prompt.pool_size
            if pool_size:
                prompt_counts = np.zeros(pool_size, dtype=np.int64)
            else:
                print("Warning: Unable to determine pool_size, prompt frequency counting disabled.")

        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with jt.no_grad():
                        model.prompt.prompt_pool.grad.zero_()
                        model.prompt.prompt_pool[cur_idx] = model.prompt.prompt_pool[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()

        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with jt.no_grad():
                    model.prompt.prompt_key.grad.zero_()
                    model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = jt.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            epoch_start_time = time.time()

            train_stats, global_step = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                                       data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                                       epoch=epoch, max_norm=args.clip_grad,
                                                       task_id=task_id, class_mask=class_mask, writer=writer,
                                                       global_step=global_step, prompt_counts=prompt_counts,
                                                       args=args, )

            if lr_scheduler:
                lr_scheduler.step(epoch)

            jt.sync_all(True)

            global_epoch += 1

            current_time = time.time()
            cumulative_elapsed_time_sec = current_time - start_time
            epoch_elapsed_time_sec = current_time - epoch_start_time

            cumulative_times_list.append(cumulative_elapsed_time_sec)
            per_epoch_times_list.append(epoch_elapsed_time_sec)

            if writer is not None:
                writer.add_scalar(f'Performance/Cumulative_Time',
                                  cumulative_elapsed_time_sec, global_epoch)
                writer.add_scalar(f'Performance/Per_Epoch_Time',
                                  epoch_elapsed_time_sec, global_epoch)
                # if prompt_counts is not None:
                #     for i, freq in enumerate(prompt_counts):
                #         writer.add_scalar(f'Prompt_Frequencies/Task_{task_id + 1}', freq, i + 1)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                       task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, writer=writer,
                                       global_step=global_step, args=args)

        jt.sync_all(True)

        if args.output_dir:
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.jt'.format(task_id + 1))
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            print(f"  Checkpoint saved to {checkpoint_path}")
            jt.save(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, }

        if args.output_dir:
            current_time_str_log = datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M_%S')
            log_stats_file = os.path.join(args.output_dir, f'{current_time_str_log}_stats.txt')

            with open(log_stats_file, 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    if args.output_dir:
        current_time_str_log = datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M_%S')
        log_stats_file = os.path.join(args.output_dir, f'{current_time_str_log}_final_log.txt')

        if log_stats_file:
            num_tasks = args.num_tasks

            avg_epoch_time = np.mean(per_epoch_times_list)
            final_avg_forgetting = np.mean(
                (np.max(acc_matrix, axis=1) - acc_matrix[:, num_tasks - 1])[:num_tasks - 1])
            avg_final_acc = np.mean(acc_matrix[:, -1])
            avg_inc_acc = np.mean(np.diag(acc_matrix))

            summary_data = {
                'cumulative_time': str(datetime.timedelta(seconds=int(cumulative_times_list[-1]))),
                'avg_epoch_times_sec': f"{avg_epoch_time:.2f}s",
                'final_accuracy_matrix': acc_matrix.tolist(),
                'avg_incremental_accuracy': f"{avg_inc_acc * 100:.2f}%",
                'avg_final_accuracy': f"{avg_final_acc * 100:.2f}%",
                'final_avg_forgetting': f"{final_avg_forgetting * 100:.2f}%"
            }
            try:
                with open(log_stats_file, 'a') as f:
                    f.write("\n" + "=" * 20 + " TRAINING SUMMARY " + "=" * 20 + "\n")
                    f.write(json.dumps(summary_data, indent=2) + '\n')
                print(f"Final summary saved to {log_stats_file}")
            except Exception as e:
                print(f"Error writing final summary to {log_stats_file}: {e}")
