"""
run_all.py — запуск полного пайплайна анализа.

Порядок:
  1. extract_features.py    — извлечение текстовых признаков
  2. eda_text_features.py   — EDA + статтесты + violin-plots
  3. econometrics.py        — Logit / OLS / Tobit / LASSO
  4. ml_models.py           — ML-модели + SHAP
  5. interpretability.py    — PDP / ICE / Permutation / LIME / GAM
  6. shap_tfidf.py          — SHAP на уровне слов

Запуск:
  python3 run_all.py                        # всё подряд
  python3 run_all.py --skip extract         # пропустить шаг(и)
  python3 run_all.py --only ml,econ         # только указанные шаги
  python3 run_all.py --use-tobit            # передать флаг в econometrics
  python3 run_all.py --log-funding-pct      # передать флаг в econometrics
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("extract",  "extract_features.py",   "Извлечение текстовых признаков (1.1–1.7)"),
    ("detailed_eda", "detailed_eda.py",   "Детальный EDA в папке eda_detailed/"),
    ("eda",      "eda_text_features.py",  "EDA + статтесты + violin-plots"),
    ("econ",     "econometrics.py",       "Logit / OLS / Tobit / LASSO"),
    ("ml",       "ml_models.py",          "ML-модели + SHAP"),
    ("interp",   "interpretability.py",   "PDP / ICE / Permutation / LIME / GAM"),
    ("tfidf",    "shap_tfidf.py",         "SHAP на уровне слов через TF-IDF"),
]


def run_step(key: str, script: str, desc: str, extra_args: list[str]) -> tuple[bool, float]:
    """Запускает один скрипт. Возвращает (success, elapsed_seconds)."""
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        print(f"  ⚠️  {script} не найден — пропуск")
        return False, 0.0

    # Только econometrics.py принимает наши флаги
    args = [sys.executable, path]
    if key == "econ":
        args += extra_args

    print(f"\n{'═' * 70}")
    print(f"▶  [{key:7s}]  {script}")
    print(f"   {desc}")
    if key == "econ" and extra_args:
        print(f"   args: {' '.join(extra_args)}")
    print("═" * 70)

    t0 = time.time()
    try:
        result = subprocess.run(args, cwd=HERE, check=False)
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"\n✓ [{key}] завершено за {elapsed:.1f}s")
            return True, elapsed
        else:
            print(f"\n✗ [{key}] упал с кодом {result.returncode} за {elapsed:.1f}s")
            return False, elapsed
    except KeyboardInterrupt:
        print(f"\n⛔ [{key}] прерван пользователем")
        raise
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n✗ [{key}] ОШИБКА: {e}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Запуск полного пайплайна анализа",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Шаги: " + ", ".join(s[0] for s in STEPS),
    )
    parser.add_argument("--skip", default="",
                        help="Пропустить шаги (через запятую), напр. --skip tfidf,interp")
    parser.add_argument("--only", default="",
                        help="Запустить только эти шаги, напр. --only extract,ml")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Остановиться при первой ошибке (по умолчанию продолжает)")
    # Флаги для econometrics.py
    parser.add_argument("--use-tobit", action="store_true",
                        help="Передать --use-tobit в econometrics.py")
    parser.add_argument("--log-funding-pct", action="store_true",
                        help="Передать --log-funding-pct в econometrics.py")
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    econ_args = []
    if args.use_tobit:
        econ_args.append("--use-tobit")
    if args.log_funding_pct:
        econ_args.append("--log-funding-pct")

    # Фильтруем шаги
    steps_to_run = []
    for key, script, desc in STEPS:
        if only and key not in only:
            continue
        if key in skip:
            continue
        steps_to_run.append((key, script, desc))

    print(f"\n{'█' * 70}")
    print(f"  ПАЙПЛАЙН: {len(steps_to_run)} шагов")
    for key, script, desc in steps_to_run:
        print(f"    • {key:7s}  {script:30s} — {desc}")
    print("█" * 70)

    results = []
    t_total = time.time()
    for key, script, desc in steps_to_run:
        success, elapsed = run_step(key, script, desc, econ_args)
        results.append((key, script, success, elapsed))
        if not success and args.stop_on_error:
            print(f"\n⛔ Остановка из-за --stop-on-error")
            break

    total = time.time() - t_total

    # Сводка
    print(f"\n\n{'█' * 70}")
    print("  СВОДКА")
    print("█" * 70)
    for key, script, success, elapsed in results:
        mark = "✓" if success else "✗"
        print(f"  {mark}  [{key:7s}]  {script:30s}  {elapsed:6.1f}s")
    ok_count = sum(1 for _, _, s, _ in results if s)
    print(f"\n  Успешно: {ok_count} / {len(results)}   Всего: {total:.1f}s")
    print("█" * 70)

    # Код выхода: 0 если все OK, иначе 1
    sys.exit(0 if ok_count == len(results) else 1)


if __name__ == "__main__":
    main()
