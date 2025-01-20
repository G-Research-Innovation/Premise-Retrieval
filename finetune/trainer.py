from sentence_transformers import SentenceTransformer, models
from transformers.trainer import *
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from typing import cast
import torch.distributed as dist
import re
import torch

def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normlized: bool = True,
                                        use_lora: bool = False, ori_dir: str = None):
    if use_lora:
        model = AutoModel.from_pretrained(ori_dir)
        tokenizer = AutoTokenizer.from_pretrained(ori_dir)
        lora_config = PeftConfig.from_pretrained(ckpt_dir)
        lora_config.init_lora_weights = True
        model = PeftModel.from_pretrained(model, ckpt_dir, config=lora_config)
        model = model.merge_and_unload()
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if normlized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normlize_layer], device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)





# 这个是先升再降的
class CustomLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, steps_per_phase: int, lr_schedule: list, last_epoch: int = -1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            steps_per_phase (int): Number of steps per phase (e.g., 2*3545).
            lr_schedule (list of tuples): List of (start_lr, end_lr) for each phase.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.steps_per_phase = steps_per_phase
        self.lr_schedule = lr_schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the new learning rates based on the current step within each phase."""
        current_phase = self.last_epoch // self.steps_per_phase
        step_within_phase = self.last_epoch % self.steps_per_phase

        if current_phase < len(self.lr_schedule):
            start_lr, end_lr = self.lr_schedule[current_phase]
            # Linear interpolation between start_lr and end_lr within the current phase
            progress = step_within_phase / self.steps_per_phase
            current_lr = start_lr + (end_lr - start_lr) * progress
            return [current_lr for _ in self.base_lrs]
        else:
            return [0.0 for _ in self.base_lrs]  # Default to 0 if beyond defined phases

    def step(self, epoch=None):
        """Update learning rates."""
        super().step(epoch)



class BiTrainer(Trainer):
    def __init__(self, *args1, orig_dir=None, use_lora=False, model_type='encoder_only', eval_dataset_train='', used_premise='',bili=0.5,device_num=8, **kwargs):
        super(BiTrainer, self).__init__(*args1, **kwargs)
        self.orig_dir = orig_dir
        self.use_lora = use_lora
        self.model_type = model_type
        self.eval_dataset_train = eval_dataset_train
        self.used_premise = used_premise
        self.bili= bili
        self.device_num= device_num


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        if self.is_world_process_zero():
            save_ckpt_for_sentence_transformers(output_dir,
                                                pooling_mode=self.args.sentence_pooling_method,
                                                normlized=self.args.normlized,
                                                use_lora=self.use_lora,
                                                ori_dir=self.orig_dir
                                                )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:

            lr_schedule = []
            initial_lr = self.args.learning_rate
            print(initial_lr)
            for i in range(int(self.args.num_train_epochs)):
                if i % 2 == 0:
                    lr_schedule.append((0, initial_lr))
                else:
                    lr_schedule.append((initial_lr, 0))
                    initial_lr /= 2
            steps_per_epoch = math.floor(
                len(self.train_dataset) / (self.args.per_device_train_batch_size * self.device_num)
            ) // 1
            print(steps_per_epoch)
            self.lr_scheduler = CustomLR(optimizer, steps_per_phase=steps_per_epoch * 2, lr_schedule=lr_schedule)

        return self.lr_scheduler


    def process_strings(self, string_list):
        processed_list = []
        for s in string_list:
            s = "<VAR>" + s
            s = re.sub(r'\n\s+', ' ', s)
            processed_list.append(s)
        result = ''.join(processed_list)
        return result

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):

        eval_data = datasets.load_dataset('json',
                                          data_files=self.eval_dataset_train,
                                          split='train')
        corpus = datasets.load_dataset('json',
                                       data_files=self.used_premise,
                                       split='train')

        if dist.get_rank() == 0:

            corpus_context = []
            corpus_goal = []
            for item in corpus['premise']:
                combine = self.process_strings(item['state']["context"])
                corpus_context.append(combine)
                goal = re.sub(r'\n\s+', ' ', item['state']["goal"])
                corpus_goal.append("<GOAL>" + goal)

            corpus_embeddings_context = self.index(
                corpus=corpus_context,
                batch_size=64,
                max_length=256
            )
            corpus_embeddings_goal = self.index(
                corpus=corpus_goal,
                batch_size=64,
                max_length=256
            )

            scores, indices = self.search(
                queries=eval_data,
                corpus_embeddings_context=corpus_embeddings_context,
                corpus_embeddings_goal=corpus_embeddings_goal,
                batch_size=64,
                max_length=512
            )
            retrieval_results = []
            for indice in indices:
                # filter invalid indices
                indice = indice[indice != -1].tolist()
                retrieval_results.append(corpus[indice]["id"])

            ground_truths = []
            for sample in eval_data:
                ground_truths.append(sample["premise"])

            metrics = self.evaluate_final(retrieval_results, scores, ground_truths)

            for key, value in metrics.items():
                self.log({f"{metric_key_prefix}_{key}": value})
            temp_dict = {}
            temp_dict['eval_Recall@10'] = metrics['Recall@10']
            return temp_dict

    def evaluate_final(self, preds,
                       preds_scores,
                       labels,
                       cutoffs=[1, 5, 10]):
        """
        Evaluate MRR and Recall at cutoffs.
        """
        metrics = {}

        # MRR
        mrrs = np.zeros(len(cutoffs))
        for pred, label in zip(preds, labels):
            jump = False
            for i, x in enumerate(pred, 1):
                if x in label:
                    for k, cutoff in enumerate(cutoffs):
                        if i <= cutoff:
                            mrrs[k] += 1 / i
                    jump = True
                if jump:
                    break
        mrrs /= len(preds)
        for i, cutoff in enumerate(cutoffs):
            mrr = mrrs[i]
            metrics[f"MRR@{cutoff}"] = mrr

        # Recall
        recalls = np.zeros(len(cutoffs))
        for pred, label in zip(preds, labels):
            for k, cutoff in enumerate(cutoffs):
                recall = np.intersect1d(label, pred[:cutoff])
                recalls[k] += len(recall) / max(min(cutoff, len(label)), 1)
            # break
        recalls /= len(preds)
        for i, cutoff in enumerate(cutoffs):
            recall = recalls[i]
            metrics[f"Recall@{cutoff}"] = recall

        return metrics

    def search(self, queries: datasets, corpus_embeddings_context, corpus_embeddings_goal, k: int = 10,
               batch_size: int = 256,
               max_length: int = 512):
        """
        1. Encode queries into dense embeddings;
        2. Search through faiss index
        """
        query = []
        for item in queries['state']:
            context = item['context']
            goal = item['goal']
            context = self.process_strings(context)
            goal = re.sub(r'\n\s+', ' ', goal)
            combine = context + '<GOAL>' + goal
            query.append(combine)

        query_embeddings_e = self.encode(query, batch_size=batch_size, max_length=max_length)
        query_size = len(query_embeddings_e)
        all_scores = []
        all_indices = []
        for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
            j = min(i + batch_size, query_size)
            query_embeddings = query_embeddings_e[i: j]
            similarities_context = np.dot(query_embeddings, corpus_embeddings_context.T)
            similarities_goal = np.dot(query_embeddings, corpus_embeddings_goal.T)
            similarities = similarities_context *self.bili + similarities_goal*(1-self.bili)
            indices = np.argsort(similarities, axis=1)[:, ::-1][:, :k]
            scores = np.take_along_axis(similarities, indices, axis=1)
            all_scores.append(scores)
            all_indices.append(indices)

        all_scores1 = np.concatenate(all_scores, axis=0)
        all_indices1 = np.concatenate(all_indices, axis=0)

        return all_scores1, all_indices1

    def index(self, corpus, batch_size: int = 256, max_length: int = 512,
              save_path: str = None, save_embedding: bool = False,
              load_embedding: bool = False):
        """
        1. Encode the entire corpus into dense embeddings;
        2. Create faiss index;
        3. Optionally save embeddings.
        """

        corpus_embeddings = self.encode(corpus, batch_size=batch_size, max_length=max_length)
        corpus_embeddings = corpus_embeddings.astype(np.float32)
        return corpus_embeddings

    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 256,
               max_length: int = 512,
               convert_to_numpy: bool = True) -> np.ndarray:

        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
        if self.model_type != 'encoder_only':
            sentences = [item + ' <|endoftext|>' for item in sentences]
        all_embeddings = []
        print(len(sentences))
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to('cuda')
            if self.model_type == 'encoder_only':
                last_hidden_state = self.model(input=inputs)
                embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
                if self.args.normlized:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            else:
                hidden_states = self.model(input=inputs)
                lens = inputs['attention_mask'].sum(dim=1)
                p_reps = (hidden_states * inputs['attention_mask'].unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
                embeddings = torch.nn.functional.normalize(p_reps, dim=1)

            embeddings = cast(torch.Tensor, embeddings)
            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.stack(all_embeddings)

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor = None):
        if self.args.sentence_pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.args.sentence_pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
