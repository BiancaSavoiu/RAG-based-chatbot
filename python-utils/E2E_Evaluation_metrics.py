#E2E_Evaluation_metrics.py
import evaluate
import torch
from rouge_score import rouge_scorer
from nltk.util import ngrams
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class RAGEvaluator:
    def __init__(self):
        self.gpt2_model, self.gpt2_tokenizer = self.load_gpt2_model()

    def load_gpt2_model(self):
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return model, tokenizer
    
    def evaluate_bert_score(self, candidates, references):
        bertscore = evaluate.load("bertscore")
        results = bertscore.compute(predictions=candidates, references=references, lang="it")
        P = float(results['precision'][0])
        R = float(results['recall'][0])
        F1 = float(results['f1'][0])
        return P, R, F1

    def evaluate_perplexity(self, text):
        encodings = self.gpt2_tokenizer(text, return_tensors='pt')
        max_length = self.gpt2_model.config.n_positions
        stride = 512
        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len
            lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl.item()
        
    def evaluate_bleu(self, candidates, references):
        bleu = evaluate.load("bleu")

        results = bleu.compute(predictions=candidates, references=references)
        return results['bleu']
    
    def evaluate_rouge(self, candidates, references):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
        rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        return rouge1, rouge2, rougeL

    def evaluate_diversity(self, texts):
        all_tokens = [tok for text in texts for tok in text.split()]
        unique_bigrams = set(ngrams(all_tokens, 2))
        diversity_score = len(unique_bigrams) / len(all_tokens) if all_tokens else 0
        return diversity_score
        
    def evaluate_all(self, response, reference):
        candidates = [response]
        references = [reference]
        bleu = self.evaluate_bleu(candidates, references)
        rouge1, rouge2, rougeL = self.evaluate_rouge(candidates, references)
        bert_p, bert_r, bert_f1 = self.evaluate_bert_score(candidates, references)
        perplexity = self.evaluate_perplexity(response)
        diversity = self.evaluate_diversity(candidates)
        return {
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "ROUGE-2": rouge2, 
            "ROUGE-L": rougeL,
            "BERT P": bert_p,
            "BERT R": bert_r,
            "BERT F1": bert_f1,
            "Perplexity": perplexity,
            "Diversity": diversity
        }

class SemScore:
    def __init__(self):
        pass

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute_sem_score(self, eval_df, model_name, response_keys):
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel
        import pandas as pd
        from tqdm import tqdm
        """
        Computes cosine similarity between the reference answer and different model responses in eval_df.
        
        Parameters:
        - eval_df: DataFrame containing "answer" and model response columns
        - model_name: name of the model to load from HuggingFace
        - tokenizer_name: name of the tokenizer to load from HuggingFace
        - response_keys: list of column names in eval_df for different model responses

        Returns:
        - cosine_similarities_df: DataFrame of individual cosine similarities for each example and each model response
        - average_cosine_similarities_df: DataFrame of average cosine similarities for each model response
        """
        
        # Load specified model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Initialize a list to store cosine similarity results for each example
        cosine_similarities_list = []

        # Loop over all examples in eval_df
        for idx in tqdm(range(len(eval_df)), desc="Processing examples"):
            sentences = [eval_df["answer"][idx]] + [eval_df[key][idx] for key in response_keys]
            
            # Tokenize sentences
            encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            # Perform mean pooling
            sentence_embeddings = SemScore.mean_pooling(self, model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            # Calculate cosine similarity for this example and store in a dictionary
            cosine_scores = {"example_index": idx}
            for i, key in enumerate(response_keys, start=1):
                cosine_similarity = (sentence_embeddings[0] @ sentence_embeddings[i]).item()
                cosine_scores[key] = cosine_similarity
            
            # Append the scores dictionary to the list
            cosine_similarities_list.append(cosine_scores)

        # Convert the list of cosine similarities into a DataFrame
        cosine_similarities_df = pd.DataFrame(cosine_similarities_list)
        cosine_similarities_df = cosine_similarities_df.drop(columns= "example_index")

        # Calculate average similarity scores for each model response and store in a new DataFrame
        average_cosine_similarities_df = pd.DataFrame(cosine_similarities_df[response_keys].mean()).T

        # Return both the individual and average DataFrames
        return cosine_similarities_df, average_cosine_similarities_df
    
class SemScoreQueryRewriting:
    def __init__(self):
        pass

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute_sem_score(self, eval_df, model_name, reference, response):
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel
        import pandas as pd
        from tqdm import tqdm
        
        """
        Computes cosine similarity between the Original_Question and Rewritten_Question in eval_df.
        
        Parameters:
        - eval_df: DataFrame containing "Original_Question" and "Rewritten_Question" columns
        - model_name: name of the model to load from HuggingFace

        Returns:
        - cosine_similarities_df: DataFrame of individual cosine similarities for each example
        - average_cosine_similarity: Float representing the average cosine similarity across all examples
        """
        
        # Load specified model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Initialize a list to store cosine similarity results for each example
        cosine_similarities = []

        # Loop over all examples in eval_df
        for idx in tqdm(range(len(eval_df)), desc="Processing examples"):
            sentences = [
                eval_df[reference][idx], 
                eval_df[response][idx]
            ]
            
            # Tokenize sentences
            encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            # Perform mean pooling
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            # Calculate cosine similarity for this example
            cosine_similarity = (sentence_embeddings[0] @ sentence_embeddings[1]).item()
            
            # Append the similarity score to the list
            cosine_similarities.append(cosine_similarity)

        # Convert the list of cosine similarities into a DataFrame
        cosine_similarities_df = pd.DataFrame({"Cosine_Similarity": cosine_similarities})
        
        # Calculate the average similarity score
        average_cosine_similarity = cosine_similarities_df["Cosine_Similarity"].mean()

        # Return both the individual DataFrame and the average similarity score
        return cosine_similarities_df, average_cosine_similarity