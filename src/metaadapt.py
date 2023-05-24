import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime

from roberta_utils import *
from dataloader import *
from dataset import get_dataset
from config import *
from utils import *
from tqdm import tqdm
import json
import os


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluation(args, model, eval_dataloader):
    eval_preds, eval_labels, eval_losses = [], [], []
    tqdm_dataloader = tqdm(eval_dataloader)
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, attention_mask, labels = batch     
            outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                    )

            loss = outputs[0]
            logits = outputs[1]
            eval_preds += torch.argmax(logits, dim=1).cpu().numpy().tolist()
            eval_labels += labels.cpu().numpy().tolist()
            eval_losses.append(loss.item())

            tqdm_dataloader.set_description('Eval bacc: {:.4f}, acc: {:.4f}, f1: {:.4f}, loss: {:.4f}'.format(
                balanced_accuracy_score(eval_labels, eval_preds),
                np.mean(np.array(eval_labels)==np.array(eval_preds)), 
                f1_score(eval_labels, eval_preds),
                np.mean(eval_losses)
            ))

    final_bacc = balanced_accuracy_score(eval_labels, eval_preds)
    final_acc = np.mean(np.array(eval_preds)==np.array(eval_labels))
    final_f1 = f1_score(eval_labels, eval_preds)
    final_precision = precision_score(eval_labels, eval_preds)
    final_recall = recall_score(eval_labels, eval_preds)
    
    return final_bacc, final_acc, final_f1, final_precision, final_recall


def adapt(args):
    fix_random_seed_as(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.output_dir:
        args.output_dir = datetime.now().strftime("%Y%m%d%H%M%S")
    export_root = os.path.join(EXPERIMENT_ROOT_FOLDER, args.output_dir)
    if not os.path.exists(export_root):
        os.makedirs(export_root)
    
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    # training data
    data, labels = get_dataset(args, 'train', args.source_data_type, args.source_data_path).load_dataset()
    inputs = preprocess(args, data)
    all_input_ids, all_attention_mask = tokenize(args, inputs, tokenizer)
    train_dataset = TensorDataset(
                all_input_ids,
                all_attention_mask,
                torch.tensor(labels)
                )
    train_dataloader = DataLoader(
                train_dataset,  
                sampler=InfiniteSampler(train_dataset),
                batch_size=args.train_batchsize
                )
    train_iter = iter(train_dataloader)
    # query data
    target_data, target_labels = get_dataset(args, 'val', args.target_data_type, args.target_data_path).load_dataset()
    pos_ids = (target_labels == 1).nonzero()[0]
    neg_ids = (target_labels == 0).nonzero()[0]
    # pos_ids = np.random.permutation(pos_ids)
    # neg_ids = np.random.permutation(neg_ids)
    
    query_data, query_labels, val_data, val_labels = [], [], [], []
    for i, j in zip(pos_ids[:args.num_shots], neg_ids[:args.num_shots]):
        query_data.append(target_data[i])
        query_labels.append(1)
        query_data.append(target_data[j])
        query_labels.append(0)
    
    inputs = preprocess(args, query_data)
    all_input_ids, all_attention_mask = tokenize(args, inputs, tokenizer)
    query_dataset = TensorDataset(
                all_input_ids,
                all_attention_mask,
                torch.tensor(query_labels)
                )
    query_loader = DataLoader(
                query_dataset,  
                sampler=RandomSampler(query_dataset),
                batch_size=args.train_batchsize
                )
    # val data
    for i in pos_ids[args.num_shots:]:
        val_data.append(target_data[i])
        val_labels.append(1)
    for j in neg_ids[args.num_shots:]:
        val_data.append(target_data[j])
        val_labels.append(0)
    inputs = preprocess(args, val_data)
    all_input_ids, all_attention_mask = tokenize(args, inputs, tokenizer)
    val_dataset = TensorDataset(
                all_input_ids,
                all_attention_mask,
                torch.tensor(val_labels)
                )
    val_dataloader = DataLoader(
                val_dataset,  
                sampler=RandomSampler(val_dataset),
                batch_size=args.eval_batchsize
                )

    # test data
    test_dataloader = get_target_loader(args, mode='target_test', tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(args.lm_model,
                                                               num_labels=2, 
                                                               output_attentions=False, 
                                                               output_hidden_states=False
                                                               )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate_meta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=args.num_iterations,
                                                           eta_min=args.learning_rate_meta*args.lr_decay_meta)
    inner_loop_optimizer = LSLRGradientDescentLearningRule(device=args.device,
                                                           total_num_inner_loop_steps=args.num_updates,
                                                           init_learning_rate=args.learning_rate_learner,
                                                           use_learnable_lr=True,
                                                           lr_of_lr=args.learning_rate_lr)
    inner_loop_optimizer.initialize(names_weights_dict=model.named_parameters())
    
    print('***** Running adaptation *****')
    
    losses = []
    scaler = torch.cuda.amp.GradScaler()
    trange = tqdm(range(args.num_iterations))
    best_bacc, best_acc, best_f1, _, _ = evaluation(args, model, val_dataloader)
    
    for i in trange:
        model.train()
        with torch.autocast(device_type=args.device, dtype=torch.float16):
            mean_outer_loss = 0.
            meta_gradients = []
            task_similarity = []
            for j in range(args.num_pi):
                support = tuple(t.to(args.device) for t in next(train_iter))
                fast_weights = OrderedDict(model.named_parameters())  # start from original params
                for k in range(args.num_updates):
                    input_ids, attention_mask, labels = support
                    outputs = functional_roberta_for_classification(fast_weights, model.config, 
                                                                    input_ids=input_ids, 
                                                                    attention_mask=attention_mask,
                                                                    labels=labels,
                                                                    is_train=True)
                    loss = outputs[0]
                    model.zero_grad()
                    scaled_grads = torch.autograd.grad(scaler.scale(loss), fast_weights.values(),
                                                       create_graph=True, retain_graph=True)
                    inv_scale = 1. / scaler.get_scale()
                    grads = [p * inv_scale for p in scaled_grads]  # manually unscale task gradients
                    if any([False in torch.isfinite(g) for g in grads]):
                        print('Invalid task gradients, adjust scale and zero out gradients')
                        if scaler.get_scale() * scaler.get_backoff_factor() >= 1.:
                            scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
                        for g in grads: g.zero_()
                    fast_weights = inner_loop_optimizer.update_params(fast_weights, grads, k)
                
                # use phi - theta as task gradients
                task_gradients = tuple()
                weight_before = OrderedDict(model.named_parameters())
                for _, (params_before, params_after) in enumerate(zip(weight_before, fast_weights)):
                    task_gradients += (fast_weights[params_before].detach() - weight_before[params_before].detach(),)

                query_loss = 0.
                for k, query in enumerate(query_loader):
                    query = tuple(t.to(args.device) for t in query)
                    input_ids, attention_mask, labels = query
                    outputs = functional_roberta_for_classification(fast_weights, model.config, 
                                                                    input_ids=input_ids, 
                                                                    attention_mask=attention_mask,
                                                                    labels=labels,
                                                                    is_train=True)
                    query_loss += outputs[0]
                
                query_loss /= len(query_loader)
                scaled_meta_grads = torch.autograd.grad(scaler.scale(query_loss),
                                                        model.parameters(), retain_graph=True)
                meta_gradients.append(scaled_meta_grads)  # use scaled gradients as meta gradients
                
                # normalize task and meta gradients before computing task similarity
                cur_similarity = [F.cosine_similarity(x.view(-1), y.view(-1), dim=-1) for x, y in zip(task_gradients, scaled_meta_grads)]
                cur_similarity = torch.mean(torch.tensor(cur_similarity))
                task_similarity.append(cur_similarity)
                
                # use meta loss to update learnable lr
                inner_loop_optimizer.update_lrs(query_loss, scaler)
                mean_outer_loss += query_loss
            
            task_similarity = F.softmax(torch.tensor(task_similarity) / args.softmax_temp, -1)
            for weights in zip(model.parameters(), *meta_gradients):
                for k in range(len(meta_gradients)):
                    if k == 0: weights[0].grad = task_similarity[k] * weights[k+1]
                    else: weights[0].grad += task_similarity[k] * weights[k+1]
            
            # notice the above meta gradients are scaled
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            scheduler.step()
            
            # sometimes scale becomes too small
            if scaler.get_scale() < 1.:
                scaler.update(1.)

            mean_outer_loss /= args.num_pi
            losses.append(mean_outer_loss.item())
            trange.set_description('Meta loss, lr and scale: {:.4f}, {:.8f}, {:.0f}'.format(np.array(losses).mean(), scheduler.get_last_lr()[-1], scaler.get_scale()))
        
        if (i+1) % args.eval_interval == 0:
            bacc, acc, f1, _, _ = evaluation(args, model, val_dataloader)
            if bacc + acc + f1 > best_bacc + best_acc + best_f1:
                best_bacc = bacc
                best_acc = acc
                best_f1 = f1
                print()
                print('***** Saving best model *****')
                model.save_pretrained(export_root)
                tokenizer.save_pretrained(export_root)
                output_args_file = os.path.join(export_root, 'training_args.bin')
                torch.save(args, output_args_file)
    
    print('***** Running evaluation *****')
    model = AutoModelForSequenceClassification.from_pretrained(
        export_root,
        num_labels=2, 
        output_attentions=False, 
        output_hidden_states=False
        ).to(args.device)
    test_bacc, test_acc, test_f1, test_precision, test_recall = evaluation(args, model, test_dataloader)
    result = {
            'bacc': test_bacc,
            'acc': test_acc,
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall
            }
    
    # remove model files to save space
    if args.del_model:
        for filename in os.listdir(export_root):
            os.remove(os.path.join(export_root, filename))
    
    # only save result file
    print('Result', result)
    with open(os.path.join(export_root, 'test_metrics.json'), 'w') as f:
        json.dump(result, f) 


if __name__ == '__main__':
    print(args)
    adapt(args)
