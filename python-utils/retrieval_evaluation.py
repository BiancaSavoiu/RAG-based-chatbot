import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util
import tqdm
import csv

class RetrievalEvaluator:
    def __init__(self, retriever, similarity_model=None, similarity_threshold=0.7):
        self.retriever = retriever
        self.similarity_model = similarity_model if similarity_model else SentenceTransformer('BAAI/bge-m3')
        self.similarity_threshold = similarity_threshold
        self.run_results = {}  # Store results as {query_id: {retrieved_context_id: score}}

    def evaluate(self, eval_df):
        """
        Evaluate the retrieval technique on the provided dataframe.
        """
        for idx, row in tqdm.tqdm(eval_df.iterrows(), total=len(eval_df), desc="Processing queries"):
            query_id = f"q{idx}"  # Query IDs: q0, q1, etc.
            question = row['question']
            eval_context = row['context']  # The ground truth context from the evaluation dataframe
            print(question)

            # Retrieve relevant documents using the retriever
            retrieved_docs = self.retriever.get_relevant_documents(question)
            print(retrieved_docs)

            # Prepare results: Compute semantic similarity and determine matches
            retrieved_chunks_scores = {}
            for doc in retrieved_docs:
                retrieved_context = doc.page_content  # Extract document text
                score = util.pytorch_cos_sim(
                    self.similarity_model.encode(eval_context, convert_to_tensor=True),
                    self.similarity_model.encode(retrieved_context, convert_to_tensor=True)
                ).item()  # Compute semantic similarity score

                if score >= self.similarity_threshold:
                    retrieved_chunks_scores[retrieved_context] = 1  # Match
                else:
                    retrieved_chunks_scores[retrieved_context] = 0  # No match
                
                print(score)

            self.run_results[query_id] = retrieved_chunks_scores

    def save_to_csv(self, file_path, k = 4):
        """
        Save the results to a CSV file.
        """
        header_contexts = [f'context{i+1}' for i in range(k)]  # Define header for up to 4 contexts

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Question'] + header_contexts
            writer.writerow(header)
            for question, contexts in self.run_results.items():
                row = [question]
                for _, score in contexts.items():
                    row.append(score)

                writer.writerow(row)

        print(f"Data saved to {file_path}")

    def calculate_metrics(self, file_path, model_name):
        """
        Calculate retrieval metrics including HR, Precision, Recall, F1, MAP, MRR, and NDCG.
        """
        df = pd.read_csv(file_path)
        df.fillna(0, inplace=True)
        df[['context1', 'context2', 'context3', 'context4']] = df[['context1', 'context2', 'context3', 'context4']].astype(int)

        binary_relevance = df.iloc[:, 1:].values  # Extract only 0/1 relevance values

        # Compute Hit Rate
        hit_rate = np.mean(np.any(binary_relevance, axis=1))

        # Compute Precision, Recall, F1 per query and averages
        precision_per_q = [precision_score(row, np.ones_like(row), zero_division=0) for row in binary_relevance]
        recall_per_q = [recall_score(row, np.ones_like(row), zero_division=0) for row in binary_relevance]
        f1_per_q = [f1_score(row, np.ones_like(row), zero_division=0) for row in binary_relevance]

        precision_avg = np.mean(precision_per_q)
        recall_avg = np.mean(recall_per_q)
        f1_avg = np.mean(f1_per_q)

        # Mean Average Precision (MAP)
        def average_precision(row):
            relevant = np.sum(row)
            if relevant == 0:
                return 0
            precisions = [np.sum(row[:i+1]) / (i+1) for i in range(len(row)) if row[i] == 1]
            return np.mean(precisions) if precisions else 0

        map_score = np.mean([average_precision(row) for row in binary_relevance])

        # Mean Reciprocal Rank (MRR)
        def reciprocal_rank(row):
            for i, val in enumerate(row):
                if val == 1:
                    return 1 / (i + 1)
            return 0

        mrr_score = np.mean([reciprocal_rank(row) for row in binary_relevance])

        # Normalized Discounted Cumulative Gain (NDCG)
        def dcg(row):
            return np.sum(row / np.log2(np.arange(2, len(row) + 2)))  # DCG formula

        def ndcg(row):
            ideal = np.sort(row)[::-1]  # Ideal ranking
            ideal_dcg = dcg(ideal) if np.sum(ideal) > 0 else 1  # Avoid division by zero
            return dcg(row) / ideal_dcg if ideal_dcg > 0 else 0

        ndcg_score = np.mean([ndcg(row) for row in binary_relevance])

        # Create results dictionary
        results = {
            "Model": model_name,
            "Hit Rate": hit_rate,
            "Precision": precision_avg,
            "Recall": recall_avg,
            "F1-score": f1_avg,
            "MAP": map_score,
            "MRR": mrr_score,
            "NDCG": ndcg_score
        }

        return results