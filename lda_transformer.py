"""
lda_transformer.py — sklearn-совместимый обёрточный класс для gensim LDA.

Предотвращает утечку данных (feature leakage):
  - Dictionary и LDA обучаются ТОЛЬКО на train fold
  - transform() применяется к любому набору документов (train или test)
  - Может использоваться внутри sklearn Pipeline / ColumnTransformer / cross_val_predict

Использование:
    from lda_transformer import LDATransformer
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("lda", LDATransformer(num_topics=5)),
        ("clf", RandomForestClassifier()),
    ])
    cross_val_predict(pipe, texts_series, y, cv=5, method="predict_proba")
"""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Русские стоп-слова (те же, что в extract_features.py / shap_tfidf.py)
_RU_STOP = {
    "и", "в", "на", "не", "что", "с", "по", "а", "как", "это", "у", "за",
    "но", "из", "от", "к", "о", "же", "то", "для", "бы", "так", "вот",
    "был", "быть", "есть", "или", "еще", "уже", "мы", "вы", "он", "она",
    "они", "я", "ты", "наш", "ваш", "свой", "этот", "тот", "весь", "все",
}


class LDATransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-совместимый трансформер LDA (gensim).

    Parameters
    ----------
    num_topics : int
        Количество тем.
    passes : int
        Количество проходов при обучении LDA.
    random_state : int
        Seed для воспроизводимости.
    no_below : int
        filter_extremes: минимальное число документов для слова.
    no_above : float
        filter_extremes: максимальная доля документов для слова.
    min_docs_for_filter : int
        При меньшем N filter_extremes не применяется (малые корпуса).
    """

    def __init__(
        self,
        num_topics: int = 5,
        passes: int = 10,
        random_state: int = 42,
        no_below: int = 2,
        no_above: float = 0.9,
        min_docs_for_filter: int = 30,
    ):
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.no_below = no_below
        self.no_above = no_above
        self.min_docs_for_filter = min_docs_for_filter

        # состояние (заполняется в fit)
        self._dictionary = None
        self._lda = None

    # ── вспомогательные ──────────────────────────────────────────────────────

    def _texts_to_docs(self, X) -> list[list[str]]:
        """
        Принимает Series/list строк.
        Возвращает list of list of lemmas (очищенных и лемматизированных).
        """
        try:
            from text_features import tokenize, lemmatize_tokens, clean_text_for_lda
        except ImportError:
            # fallback: простая токенизация без лемматизации
            def clean_text_for_lda(t):
                return re.sub(r"[^а-яёА-ЯЁa-zA-Z\s]", " ", t or "")

            def tokenize(t):
                return t.split()

            def lemmatize_tokens(tokens):
                return tokens

        docs = []
        for text in X:
            if not isinstance(text, str) or not text.strip():
                docs.append([])
                continue
            cleaned = clean_text_for_lda(text)
            tokens = tokenize(cleaned)
            try:
                lemmas = lemmatize_tokens([w.lower() for w in tokens])
            except Exception:
                lemmas = [w.lower() for w in tokens]
            lemmas = [w for w in lemmas if len(w) > 2 and w not in _RU_STOP]
            docs.append(lemmas)
        return docs

    # ── sklearn API ───────────────────────────────────────────────────────────

    def fit(self, X, y=None):
        """
        Обучает Dictionary + LDA только на переданных документах (train fold).
        X — pandas Series или list строк.
        """
        try:
            from gensim.corpora import Dictionary
            from gensim.models import LdaModel
        except ImportError as exc:
            raise ImportError("gensim не установлен: pip install gensim") from exc

        docs = self._texts_to_docs(X)
        self._dictionary = Dictionary(docs)

        if len(docs) >= self.min_docs_for_filter:
            self._dictionary.filter_extremes(
                no_below=self.no_below, no_above=self.no_above
            )

        corpus = [self._dictionary.doc2bow(d) for d in docs]

        self._lda = LdaModel(
            corpus=corpus,
            id2word=self._dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            random_state=self.random_state,
        )
        return self

    def transform(self, X, y=None) -> np.ndarray:
        """
        Преобразует документы в матрицу (N, num_topics) — вероятности тем.
        Использует Dictionary и LDA, обученные только на train fold.
        """
        if self._lda is None:
            raise RuntimeError("LDATransformer не обучен: вызови fit() сначала")

        docs = self._texts_to_docs(X)
        result = np.zeros((len(docs), self.num_topics), dtype=float)
        for i, doc in enumerate(docs):
            bow = self._dictionary.doc2bow(doc)
            for tid, prob in self._lda.get_document_topics(
                bow, minimum_probability=0.0
            ):
                result[i, tid] = prob
        return result

    def get_feature_names_out(self):
        return np.array([f"lda_topic_{i}" for i in range(self.num_topics)])
