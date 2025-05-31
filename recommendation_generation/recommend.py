import time

import lenskit.crossfold as xf
import numpy as np
import pandas as pd
from ConfigSpace import *
from lenskit import topn
from lenskit.batch import recommend
from lkauto.lkauto import get_best_recommender_model
from lkauto.algorithms.item_knn import ItemItem
from lkauto.algorithms.user_knn import UserUser
from lkauto.algorithms.funksvd import FunkSVD
from lkauto.utils.filer import Filer


# Funktion, die für ein gegebenes Szenario und einen Algorithmus die Empfehlungen berechnet und evaluiert
def compute_recommendations(scenario, algorithm, id):
    ndcgs, precisions, recalls, train_times, test_times = [], [], [], [], []
    filer = Filer(f'output/run_{id}_{algorithm}')

    for i, tp in enumerate(xf.partition_users(scenario, 5, xf.SampleFrac(0.2), rng_spec=42)):
        print('Fold: ' + str(i + 1))
        train_split = tp.train.copy()
        test_split = tp.test.copy()

        print(f'Test size: {test_split.shape}')
        print(f'Train size: {train_split.shape}')

        truth = test_split[test_split['rating'] >= 4]

        cs = ConfigurationSpace()
        cs.seed(42)

        if algorithm == 'ItemItem':
            cs.add_hyperparameters([
                Constant("algo", "ItemItem"),
                Integer('nnbrs', bounds=(1, 10000), default=1000, log=True),
                Integer('min_nbrs', bounds=(1, 1000), default=1, log=True),
                Float('min_sim', bounds=(1.0e-10, 1.0e-2), default=1.0e-6),
                Categorical("center", [True, False]),
                Constant("use_ratings", 'True')
            ])
        elif algorithm == 'UserUser':
            cs.add_hyperparameters([
                Constant("algo", "UserUser"),
                Integer('nnbrs', bounds=(1, 10000), default=1000, log=True),
                Integer('min_nbrs', bounds=(1, 1000), default=1, log=True),
                Float('min_sim', bounds=(1.0e-10, 1.0e-2), default=1.0e-6),
                Categorical("center", [True, False]),
                Constant("use_ratings", 'True')
            ])
        elif algorithm == 'FunkSVD':
            cs.add_hyperparameters([
                Constant("algo", "FunkSVD"),
                Integer("features", bounds=(2, 10000), default=1000, log=True),
                Float('lrate', bounds=(0.0001, 0.01), default=0.001),
                Float('reg', bounds=(0.001, 0.1), default=0.015),
                Float('damping', bounds=(0.01, 1000), default=5, log=True),
                Categorical("bias", [True, False])
            ])

        # Optimiert den Algorithmus mittels Bayesianischer Optimierung und findet die beste Konfiguration
        model, config = get_best_recommender_model(
            train=train_split,
            optimization_strategie='bayesian',
            optimization_metric=topn.ndcg,
            random_state=42,
            filer=filer,
            cs=cs,
            include_timestamp=False,
            num_evaluations=50,
            num_recommendations=5,
            log_level='FATAL'
        )

        # Trainiert das Modell mit der gefundenen besten Konfiguration
        start_train = time.time()
        model.fit(train_split)
        train_time = (time.time() - start_train)
        train_times.append(train_time)

        num_users = test_split['user'].unique()

        # Generiert Empfehlungen für die Test-User
        start_test = time.time()
        recs = recommend(algo=model, users=num_users, n=5)
        test_time = (time.time() - start_test)
        test_times.append(test_time)

        # Berechnet Bewertungsmetriken für die Empfehlungen
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.ndcg, name='ndcg')
        rla.add_metric(topn.precision, name='precision')
        rla.add_metric(topn.recall, name='recall')

        scores = rla.compute(recs, truth, include_missing=True)

        ndcgs.append(scores['ndcg'].mean())
        precisions.append(scores['precision'].mean())
        recalls.append(scores['recall'].mean())
        print('ndcgs: ', ndcgs)
        print('precisions: ', precisions)
        print('recalls: ', recalls)
        print('Train-Times: ', train_times)
        print('Test-Times: ', test_times)

    # Aggregiert Ergebnisse aller Folds zu einem abschließenden Summary
    summary = {
        'scenario': f"scenario{id + 1}",
        'algorithm': algorithm,
        'ndcg@5': np.mean(ndcgs),
        'precision@5': np.mean(precisions),
        'recall@5': np.mean(recalls),
        'train_time': np.mean(train_times),
        'test_time': np.mean(test_times),
        'params': config,
        'fold_results': {
            f"fold_{i + 1}": {
                'ndcg': ndcgs[i],
                'precision': precisions[i],
                'recall': recalls[i],
                'train_time': train_times[i],
                'test_time': test_times[i]
            }
            for i in range(len(ndcgs))
        }
    }

    print('Summary: ', summary)

    return summary


# Hauptfunktion zum iterativen Durchlaufen aller Szenarien und Algorithmen
def main():
    total_start = time.time()
    summary_rows = []
    scenarios = [
        pd.read_csv('../data_generation/scenarios/scenario01.csv'),
        pd.read_csv('../data_generation/scenarios/scenario02.csv'),
        pd.read_csv('../data_generation/scenarios/scenario03.csv'),
        pd.read_csv('../data_generation/scenarios/scenario04.csv'),
        pd.read_csv('../data_generation/scenarios/scenario05.csv'),
        pd.read_csv('../data_generation/scenarios/scenario06.csv'),
        pd.read_csv('../data_generation/scenarios/scenario07.csv'),
        pd.read_csv('../data_generation/scenarios/scenario08.csv'),
        pd.read_csv('../data_generation/scenarios/scenario09.csv'),
        pd.read_csv('../data_generation/scenarios/scenario10.csv'),
        pd.read_csv('../data_generation/scenarios/scenario11.csv'),
        pd.read_csv('../data_generation/scenarios/scenario12.csv')
    ]

    algorithms = [
        'ItemItem',
        'UserUser',
        'FunkSVD'
    ]

    for id, scenario in enumerate(scenarios):
        for algorithm in algorithms:
            try:
                print(f"\nRunning {algorithm} on scenario{id + 1}")
                summary = compute_recommendations(scenario, algorithm, id)
                summary_rows.append(summary)
            except Exception as e:
                print(f"Fehler bei scenario{id + 1}, {algorithm}: {e}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.set_index('scenario', inplace=True)
    df_summary.to_csv('output/summary.csv', index=True)
    total_end = time.time()
    print(f"Gesamtlaufzeit: {total_end - total_start:.2f} Sekunden")


if __name__ == "__main__":
    main()
