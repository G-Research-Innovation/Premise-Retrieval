from typing import cast, List, Union, Tuple
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, is_torch_npu_available
from transformers import AutoModelForTextEncoding
import transformers
from typing import Dict

DEFAULT_PAD_TOKEN = "[PAD]"


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


class FlagModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'mean',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            use_fp16: bool = True,
            model_type: str = 'encoder_only',
    ) -> None:
        if model_type == 'encoder_only' or model_type == 'decoder_only':
            self.model = AutoModel.from_pretrained(model_name_or_path)
        elif model_type == 'reprover':
            self.model = AutoModelForTextEncoding.from_pretrained(model_name_or_path)
        else:
            self.model = T5EncoderModel.from_pretrained(model_name_or_path)

        if model_type == 'encoder_only' or model_type=='no_pretrain':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=model_name_or_path,
                padding_side="right",
                use_fast=True,
                # use_fast=False,
            )

            if self.tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                    tokenizer=self.tokenizer,
                    model=self.model,
                )
        self.model_type = model_type
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        # encoder
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 512,
                       convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.query_instruction_for_retrieval + queries
            else:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = 512,
                      convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        return self.encode(corpus, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    @torch.no_grad()
    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 256,
               max_length: int = 512,
               convert_to_numpy: bool = True) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
        if self.model_type != 'encoder_only':
            sentences = [item + ' <|endoftext|>' for item in sentences]
        all_embeddings = []
        # print(len(sentences))
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            try:
                inputs = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=max_length,
                ).to(self.device)
            except:
                print(sentences_batch)
                raise ValueError('error')
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

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
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
