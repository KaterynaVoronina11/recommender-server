import pandas as pd
import numpy as np
import joblib
import os
import logging
from collections import defaultdict
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrictRecommenderTester:
    def __init__(self, recommender, model_dir="models"):
        self.recommender = recommender
        self.model_dir = model_dir
        
        # Завантаження моделей та даних
        # self.svd_model = joblib.load(os.path.join(model_dir, "svd_model.pkl"))
        # self.item_factors = joblib.load(os.path.join(model_dir, "item_factors.pkl"))
        # self.item_similarities = joblib.load(os.path.join(model_dir, "item_similarities.pkl"))
        self.isbn_to_idx = joblib.load(os.path.join(model_dir, "isbn_to_idx.pkl"))
        self.idx_to_isbn = joblib.load(os.path.join(model_dir, "idx_to_isbn.pkl"))
        self.books_df = pd.read_pickle(os.path.join(model_dir, "books_df_slim.pkl"))
        self.ratings_df = pd.read_pickle(os.path.join(model_dir, "ratings_slim.pkl"))
        
    def prepare_test_data(self, min_ratings_per_user=10, good_rating_threshold=7, random_state=42):
        """Готує дані з поділом 70/30, враховуючи тільки користувачів з ≥10 оцінками."""
        logger.info("Підготовка даних з поділом 70/30")
        
        # Відбираємо користувачів з достатньою кількістю оцінок
        user_counts = self.ratings_df['User-ID'].value_counts()
        eligible_users = user_counts[user_counts >= min_ratings_per_user].index
        eligible_data = self.ratings_df[self.ratings_df['User-ID'].isin(eligible_users)].copy()
        
        logger.info(f"Відібрано {len(eligible_users)} користувачів з ≥{min_ratings_per_user} оцінками")
   
        train_data = []
        test_data = []
        user_test_items = {}  # {user_id: {'isbns': [...], 'ratings': [...]}}
        
        for user_id in eligible_users:
            user_ratings = eligible_data[eligible_data['User-ID'] == user_id]
            
            # Поділ 70/30 з фіксованим random_state для відтворюваності
            user_train, user_test = train_test_split(
                user_ratings, 
                test_size=0.3, 
                random_state=random_state
            )
            
            train_data.append(user_train)
            test_data.append(user_test)
            
            # Зберігаємо тестові ISBN та їх рейтинги
            user_test_items[user_id] = {
                'isbns': user_test['ISBN'].tolist(),
                'ratings': user_test['Book-Rating'].tolist()
            }
        
        train_data = pd.concat(train_data)
        test_data = pd.concat(test_data)
        
        logger.info(f"Розмір тренувального набору: {len(train_data)}")
        logger.info(f"Розмір тестового набору: {len(test_data)}")
        
        return train_data, test_data, user_test_items
    
    def test_all_methods(self, user_test_items, methods=['svd', 'item_cf', 'hybrid'], k=10):
        """Тестує всі методи на заданих користувачах."""
        results = {method: defaultdict(list) for method in methods}
        
        for user_id, test_data in list(user_test_items.items())[:1000]:
            test_isbns = test_data['isbns']
            test_ratings = test_data['ratings']
            
            # Тренувальні рейтинги користувача (всі, крім тестових)
            user_ratings = self.ratings_df[
                (self.ratings_df['User-ID'] == user_id) & 
                (~self.ratings_df['ISBN'].isin(test_isbns))
            ]
            user_ratings_dict = dict(zip(user_ratings['ISBN'], user_ratings['Book-Rating']))
            
            if len(user_ratings_dict) < 7:
                continue  # Пропускаємо, якщо замало тренувальних даних
                
            # Тестуємо кожен метод
            for method in methods:
                try:
                    if method == 'svd':
                        recs = self.recommender._get_svd_recommendations(
                            user_ratings_dict, 
                            valid_isbns=list(user_ratings_dict.keys())
                        )
                    elif method == 'item_cf':
                        recs = self.recommender._get_item_based_recommendations(
                            user_ratings_dict,
                            valid_isbns=list(user_ratings_dict.keys())
                        )
                    elif method == 'hybrid':
                        recs = self.recommender.get_hybrid_recommendations(
                            user_ratings_dict,
                            n_recommendations=k,
                        )
                    
                    # Перетворюємо рекомендації у список кортежів (ISBN, score) та сортуємо
                    if isinstance(recs, dict):
                        recs_sorted = sorted(recs.items(), key=lambda x: x[1], reverse=True)[:k]
                    else:
                        recs_sorted = recs[:k]  # Якщо вже список
                    
                    metrics = self.evaluate_recommendations(recs_sorted, test_isbns, test_ratings, k)
                    
                    for metric, value in metrics.items():
                        results[method][metric].append(value)
                except Exception as e:
                    logger.warning(f"Помилка для користувача {user_id}, метод {method}: {str(e)}")
                    continue
        
        # Середні значення метрик
        avg_results = {}
        for method in methods:
            avg_results[method] = {
                metric: np.mean(values) if values else 0
                for metric, values in results[method].items()
            }
        
        return avg_results

    def evaluate_recommendations(self, recommendations, test_isbns, test_ratings, k=10):
        """Обчислює метрики, враховуючи реальні рейтинги тестових книг."""
        if not recommendations:
            return {
                'precision@k': 0,
                'recall@k': 0,
                'f1@k': 0,
                'ndcg@k': 0,
                'good_hits@k': 0
            }
        
        # Визначаємо "хороші" книги (рейтинг ≥ порогу)
        good_test_items = [isbn for isbn, rating in zip(test_isbns, test_ratings) if rating >= 7]
        
        # Топ-k рекомендацій (рекомендації вже мають бути відсортовані)
        rec_isbns = [isbn for isbn, _ in recommendations[:k]]
        
        # Точність, повнота, F1
        precision = len(set(rec_isbns) & set(good_test_items)) / len(rec_isbns) if rec_isbns else 0
        recall = len(set(rec_isbns) & set(good_test_items)) / len(good_test_items) if good_test_items else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # NDCG (враховує порядок рекомендацій)
        relevance_true = np.zeros(len(rec_isbns))
        for i, isbn in enumerate(rec_isbns):
            if isbn in test_isbns:
                idx = test_isbns.index(isbn)
                relevance_true[i] = test_ratings[idx] / 10.0  # Нормалізуємо до [0, 1]
        
        relevance_pred = np.arange(len(rec_isbns), 0, -1)  # Ідеальний порядок
        ndcg = ndcg_score([relevance_true], [relevance_pred]) if len(relevance_true) > 0 else 0
        
        # Кількість "влучень" у хороші книги
        good_hits = len(set(rec_isbns) & set(good_test_items))
        
        return {
            'precision@k': precision,
            'recall@k': recall,
            'f1@k': f1,
            'ndcg@k': ndcg,
            'good_hits@k': good_hits
        }
    
    def visualize_comparison(self, results, k=10):
        """Візуалізує порівняння методів."""
        methods = list(results.keys())
        metrics = ['precision@k', 'recall@k', 'f1@k', 'ndcg@k', 'good_hits@k']
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [results[method][metric] for method in methods]
            
            ax.bar(methods, values, color=['blue', 'green', 'red'])
            ax.set_title(f"{metric} (k={k})")
            ax.set_ylabel(metric)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('recommendation_comparison.png')
        plt.show()

# Приклад використання:
if __name__ == "__main__":
    from book_recommender import BookRecommender
    
    recommender = BookRecommender()
    recommender.svd_model = joblib.load(os.path.join('models', "svd_model_compressed.pkl"))
    recommender.item_factors = joblib.load(os.path.join('models', "item_factors_compressed.pkl"))
    recommender.item_similarities = joblib.load(os.path.join('models', "item_similarities_compressed.pkl"))
    recommender.isbn_to_idx = joblib.load(os.path.join('models', "isbn_to_idx.pkl"))
    recommender.idx_to_isbn = joblib.load(os.path.join('models', "idx_to_isbn.pkl"))
    tester = StrictRecommenderTester(recommender)
    
    # Підготовка даних (70/30 split, мінімум 10 оцінок)
    train_data, test_data, user_test_items = tester.prepare_test_data(min_ratings_per_user=10)
    # Тестування методів
    results = tester.test_all_methods(user_test_items, k=10)
    
    # Вивід результатів
    print("\nРезультати тестування (k=10):")
    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Візуалізація
    tester.visualize_comparison(results)
