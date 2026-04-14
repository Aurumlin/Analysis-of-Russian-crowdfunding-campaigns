"""
normalization_scaler.py

Управление параметрами нормализации readability признаков.

Проблема: если нормализовать readability_avg в extract_features.py,
то при добавлении новых данных:
  - mean/std изменяются
  - старые z-score становятся несопоставимы
  - теряется воспроизводимость

Решение: сохранять параметры (mean, std) и применять их консистентно.

Использование:
  # При обучении моделей:
  scaler = ReadabilityScaler(fit_on_train=True)
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  scaler.save("readability_params.json")

  # При применении к новым данным:
  scaler = ReadabilityScaler.load("readability_params.json")
  X_new = scaler.transform(X_new)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


class ReadabilityScaler:
    """
    Нормализует readability признаки (flesch, fog, lix, readability_avg)
    через z-score с сохранением параметров.
    """

    def __init__(self, fit_on_train=True):
        """
        fit_on_train=True:  fit только на train (для ML) → избегаем утечки
        fit_on_train=False: fit на всех данных (для эконометрики) → интерпретируемость
        """
        self.fit_on_train = fit_on_train
        self.params = {}  # {"feature": {"mean": ..., "std": ...}, ...}
        self.fitted = False

    def fit(self, X_train, columns=None):
        """
        Вычислить mean/std на train.

        Args:
            X_train: DataFrame или dict
            columns: список колонок для нормализации. Если None, берём readability_*
        """
        if columns is None:
            if isinstance(X_train, pd.DataFrame):
                columns = [c for c in X_train.columns if "readability" in c]
            else:
                raise ValueError("columns не указаны и X_train не DataFrame")

        if isinstance(X_train, dict):
            X_train = pd.DataFrame(X_train)

        self.params = {}
        for col in columns:
            if col not in X_train.columns:
                continue
            data = X_train[col].dropna().values
            self.params[col] = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "count": int(len(data)),
            }

        self.fitted = True
        return self

    def transform(self, X):
        """
        Применить нормализацию.

        Args:
            X: DataFrame или dict

        Returns:
            Нормализованный DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler не подогнан. Вызови fit() сначала.")

        if isinstance(X, dict):
            X = pd.DataFrame(X)

        X = X.copy()
        for col, stats in self.params.items():
            if col not in X.columns:
                continue
            mean = stats["mean"]
            std = stats["std"]
            if std < 1e-10:
                X[col] = 0.0
            else:
                X[col] = (X[col] - mean) / std

        return X

    def fit_transform(self, X):
        """Fit и transform в один шаг."""
        return self.fit(X).transform(X)

    def save(self, filepath):
        """Сохранить параметры в JSON."""
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(
                {
                    "fit_on_train": self.fit_on_train,
                    "fitted": self.fitted,
                    "params": self.params,
                },
                f,
                indent=2,
            )
        print(f"✓ параметры нормализации сохранены: {filepath}")

    @classmethod
    def load(cls, filepath):
        """Загрузить параметры из JSON."""
        filepath = Path(filepath)
        with open(filepath) as f:
            data = json.load(f)

        scaler = cls(fit_on_train=data.get("fit_on_train", True))
        scaler.params = data.get("params", {})
        scaler.fitted = data.get("fitted", False)
        print(f"✓ параметры нормализации загружены: {filepath}")
        return scaler

    def summary(self):
        """Вывести summary параметров."""
        lines = ["=== Readability Normalization Summary ==="]
        lines.append(f"Fitted: {self.fitted}, fit_on_train: {self.fit_on_train}")
        lines.append("")
        for col, stats in self.params.items():
            lines.append(
                f"{col:25s} mean={stats['mean']:7.3f}, std={stats['std']:6.3f}, n={stats['count']}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Примеры использования
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Демо: создание и сохранение параметров
    import pandas as pd

    # Подделаем train/test данные
    X_train = pd.DataFrame({
        "readability_flesch": np.random.normal(50, 10, 80),
        "readability_fog": np.random.normal(10, 3, 80),
        "readability_lix": np.random.normal(50, 10, 80),
        "readability_avg": np.random.normal(5, 2, 80),
    })

    X_test = pd.DataFrame({
        "readability_flesch": np.random.normal(50, 10, 20),
        "readability_fog": np.random.normal(10, 3, 20),
        "readability_lix": np.random.normal(50, 10, 20),
        "readability_avg": np.random.normal(5, 2, 20),
    })

    # Fit на train
    scaler = ReadabilityScaler(fit_on_train=True)
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    print(scaler.summary())
    print("\nTrain (нормализованный):")
    print(X_train_norm.describe())

    # Сохранить параметры
    scaler.save("readability_params.json")

    # Загрузить и применить к новым данным
    X_new = pd.DataFrame({
        "readability_flesch": [45, 55, 60],
        "readability_fog": [9, 11, 12],
        "readability_lix": [48, 52, 55],
        "readability_avg": [4.5, 5.5, 6.0],
    })

    scaler_loaded = ReadabilityScaler.load("readability_params.json")
    X_new_norm = scaler_loaded.transform(X_new)
    print("\nNew data (применены загруженные параметры):")
    print(X_new_norm)
