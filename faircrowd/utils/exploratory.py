import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Sequence
from ..core import FairCrowdDataset, FairnessMetric, AccuracyMetric, TDAlgorithm
from .string_utils import title_case, name
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator


def convert_data_to_glad(answers):

    #answers=df.answers.values
    L_task=[]
    L_worker=[]
    L_label=[]

    for i in range(len(answers)):
        for j in range(len(answers[0])):
            if not(np.isnan(answers[i,j])):
                L_task.append(i)
                L_worker.append(j)
                L_label.append(int(answers[i,j]))

    Dic={"task":L_task,"worker":L_worker,"label":L_label}
    dg=pd.DataFrame(Dic)
    return dg


def maj_gold_prior_extension(set_experts,nb_answers,nb_experts):
    
    L_task=[]
    L_answers=np.zeros((3,nb_experts))
    L_label=[]
    Ground_truth=[]
    for j in range(3):
        L_task.append(nb_answers+j)
        if j<2:
            Ground_truth.append(1)
        else:
            Ground_truth.append(1)

        for el in range(nb_experts):
            if j<2:
                if el in set_experts:
                    L_answers[j,el]=1
                else:
                    L_answers[j,el]=np.nan

            else:
                if el in set_experts:
                    L_answers[j,el]=0
                else:
                    L_answers[j,el]=np.nan
    
    
    return L_answers,np.array(Ground_truth)

def maj_gold_prior_extension_old(set_experts,nb_answers):
    
    L_task=[]
    L_worker=[]
    L_label=[]
    Ground_truth=[]
    for l,el in enumerate(set_experts):
        L_task.append(nb_answers+l*3)
        L_worker.append(el)
        L_label.append(1)
        Ground_truth.append(1)
        L_task.append(nb_answers+l*3+1)
        L_worker.append(el)
        L_label.append(1)
        Ground_truth.append(1)
        L_task.append(nb_answers+l*3+2)
        L_worker.append(el)
        L_label.append(0)
        Ground_truth.append(0)

    Dic={"task":L_task,"worker":L_worker,"label":L_label}
    dg=pd.DataFrame(Dic)
    d_truth=pd.DataFrame(Ground_truth,index=L_task)
    return dg,d_truth








def get_accuracy_metrics(
    df: FairCrowdDataset,
    metrics: Sequence[AccuracyMetric],
) -> pd.DataFrame:
    """
    Get the accuracy metrics for each worker in the dataset and return them in a DataFrame
    """
    result = pd.DataFrame(index=map(name, metrics), columns=df.answers.columns)
    for worker in df.answers.columns:
        subset = df.loc[:, worker].dropna().index
        predictions = df.loc[subset, worker]
        ground_truth = df["y"].loc[subset]
        result[worker] = [metric(ground_truth, predictions) for metric in metrics]

    result = result.transpose()
    return result


def get_fairness_metrics(
    df: FairCrowdDataset, metrics: Sequence[FairnessMetric]
) -> pd.DataFrame:
    """
    Get the fairness metrics for each worker in the dataset and return them in a DataFrame
    """
    result = pd.DataFrame(index=map(name, metrics), columns=df.answers.columns)
    for worker in df.answers.columns:
        subset = df.loc[:, worker].dropna().index
        predictions = df.loc[subset, worker]
        sensitive = df["s"].loc[subset]
        ground_truth = df["y"].loc[subset]
        result[worker] = [
            metric.compute(predictions, sensitive, ground_truth) for metric in metrics
        ]

    result = result.transpose()
    return result


def plot_histograms(metrics: pd.DataFrame) -> None:
    """
    Plot histograms of accuracy or fairness metrics
    """
    for metric in metrics.columns:
        sns.histplot(metrics[metric], binrange=(0, 1), bins=20)
        plt.ylabel("Count")
        plt.xlabel(title_case(metric))
        plt.show()


def print_table(accuracy_metric: pd.Series, fairness_metrics: pd.DataFrame) -> None:
    """
    Print a table of an accuracy metric vs fairness metrics
    """
    accuracy_ranges = np.arange(0, 1, 0.1)
    result = pd.DataFrame(
        index=accuracy_ranges,
        columns=fairness_metrics.columns,
    )
    for start, end in zip(accuracy_ranges, accuracy_ranges + 0.1):
        for metric in fairness_metrics.columns:
            result.loc[start, metric] = np.nanmean(
                fairness_metrics.loc[
                    (accuracy_metric > start) & (accuracy_metric <= end), metric
                ]
            )
    result.index = map(
        lambda rng: f"({rng[0]:.1f}, {rng[1]:.1f}]:",
        zip(accuracy_ranges, accuracy_ranges + 0.1),
    )
    result.index.name = title_case(accuracy_metric.name)
    result.columns = map(title_case, result.columns)
    print(result.to_string())


def plot_scatter(x_metric: pd.Series, y_metrics: pd.DataFrame) -> None:
    """
    Create scatterplots of one metric vs all the others

    Example: accuracy vs fairness metrics, demographic parity vs performance metrics
    """
    for metric in y_metrics.columns:
        sns.scatterplot(x=x_metric, y=y_metrics[metric])
        plt.xlabel(title_case(x_metric.name))
        plt.ylabel(title_case(metric))
        plt.show()


def plot_unfair_proportion(
    df: FairCrowdDataset, fairness_metrics: pd.DataFrame
) -> None:
    """
    Plot the proportion of tasks dominated by unfair workers depending on the threshold
    """
    xs = np.linspace(0, 0.5, 11)
    ticks = np.concatenate([np.arange(0, len(xs) + 1, 2), [-1]])
    thresholds = np.concatenate([xs, [1]])
    unfairs = pd.DataFrame(columns=fairness_metrics.columns, index=thresholds)
    label_counts = np.isfinite(df.answers).sum(axis=1)
    for threshold in thresholds:
        unf = []
        for metric in fairness_metrics.columns:
            fair_workers = fairness_metrics[fairness_metrics[metric] <= threshold].index
            unf.append(
                np.where(df[fair_workers].count(axis=1) < label_counts / 2, 1, 0).mean()
            )
        unfairs.loc[threshold] = unf

    for col in unfairs.columns:
        plt.plot(
            np.concatenate([xs, [xs[-1] + 2 * xs[1]]]),
            unfairs[col],
            label=title_case(col),
        )
        plt.xticks(
            np.concatenate([xs, [xs[-1] + 2 * xs[1]]])[ticks],
            map(lambda x: f"{x:.2f}", thresholds[ticks]),
        )
    plt.xlabel("Threshold")
    plt.ylabel("Unfair Proportion")
    # plt.legend()
    plt.show()


def plot_questions_and_accuracy(
    df: FairCrowdDataset, fairness_metrics: pd.DataFrame
) -> None:
    """
    Plot the number of abandoned tasks and the overall accuracy removing unfair workers
    """
    xs = np.linspace(0, 0.5, 11)
    ticks = np.concatenate([np.arange(0, len(xs) + 1, 2), [-1]])
    thresholds = np.concatenate([xs, [1]])
    n_questions = pd.DataFrame(columns=fairness_metrics.columns, index=thresholds)
    accuracies = pd.DataFrame(columns=fairness_metrics.columns, index=thresholds)
    for threshold in thresholds:
        n_question = []
        accuracy = []
        for metric in fairness_metrics.columns:
            fair_workers = fairness_metrics[fairness_metrics[metric] <= threshold].index
            retained_answers = df.loc[:, fair_workers].dropna(how="all").index
            n_question.append(len(retained_answers))
            accuracy.append(
                (
                    df["y"].loc[retained_answers]
                    == np.where(
                        np.nanmean(df.loc[retained_answers], axis=1) > 0.5, 1, 0
                    )
                ).sum()
                / len(df)
            )
        n_questions.loc[threshold] = n_question
        accuracies.loc[threshold] = accuracy

    for col in n_questions.columns:
        plt.plot(
            np.concatenate([xs, [xs[-1] + 2 * xs[1]]]),
            n_questions[col],
            label=title_case(col),
        )
        plt.xticks(
            np.concatenate([xs, [xs[-1] + 2 * xs[1]]])[ticks],
            map(lambda x: f"{x:.2f}", thresholds[ticks]),
        )
    plt.xlabel("Threshold")
    plt.ylabel("# Questions")
    # plt.legend()
    plt.show()

    for col in accuracies.columns:
        plt.plot(
            np.concatenate([xs, [xs[-1] + 2 * xs[1]]]),
            accuracies[col],
            label=title_case(col),
        )
        plt.xticks(
            np.concatenate([xs, [xs[-1] + 2 * xs[1]]])[ticks],
            map(lambda x: f"{x:.2f}", thresholds[ticks]),
        )
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    # plt.legend()
    plt.show()


def print_td_table(
    df: FairCrowdDataset,
    accuracy_metrics: Sequence[AccuracyMetric],
    fairness_metrics: Sequence[FairnessMetric],
    td_algorithms: Sequence[TDAlgorithm],
) -> None:
    """
    Print a summary table with accuracy and fairness of td algorithms
    """
    accuracy_result = pd.DataFrame(
        index=map(name, accuracy_metrics),
        columns=list(map(name, td_algorithms)),
    )
    fairness_result = pd.DataFrame(
        index=map(name, fairness_metrics),
        columns=list(map(name, td_algorithms)),
    )
    for algo in td_algorithms:
        predictions = algo.run(df).labels
        accuracy_result[name(algo)] = [
            metric(df["y"], predictions) for metric in accuracy_metrics
        ]
        fairness_result[name(algo)] = [
            metric.compute(predictions, df["s"], df["y"]) for metric in fairness_metrics
        ]

    result = pd.concat([accuracy_result, fairness_result]).transpose()
    result.columns = map(title_case, result.columns)
    result.index = map(title_case, result.index)
    result.index.name = "TD Algorithm"
    print(result.to_string())


def print_td_table_eps(
    df: FairCrowdDataset,
    accuracy_metrics: Sequence[AccuracyMetric],
    fairness_metrics: Sequence[FairnessMetric],
    td_algorithms: Sequence[TDAlgorithm],
    eps_list: Sequence[float]
) -> None:
    """
    Print a summary table with accuracy and fairness of td algorithms
    """
    accuracy_result = pd.DataFrame(
        index=map(name, accuracy_metrics),
        columns=list(map(name, td_algorithms)),
    )
    fairness_result = pd.DataFrame(
        index=map(name, fairness_metrics),
        columns=list(map(name, td_algorithms)),
    )
    for eps in eps_list:
        for algo in td_algorithms:
            predictions = algo.run(df,eps=eps).labels
            accuracy_result[name(algo)+"_"+"{:.2f}".format(eps)] = [
            metric(df["y"], predictions) for metric in accuracy_metrics
        ]
            fairness_result[name(algo)+"_"+"{:.2f}".format(eps)] = [
            metric.compute(predictions, df["s"], df["y"]) for metric in fairness_metrics
        ]

    result = pd.concat([accuracy_result, fairness_result]).transpose()
    result.columns = map(title_case, result.columns)
    result.index = map(title_case, result.index)
    result.index.name = "TD Algorithm"
    print(result.to_string())


def print_downstream_tasks_table(
    df: FairCrowdDataset,
    accuracy_metrics: Sequence[AccuracyMetric],
    fairness_metrics: Sequence[FairnessMetric],
    td_algorithms: Sequence[TDAlgorithm],
    ml_models: Sequence[BaseEstimator],
):
    """
    Print a summary table with accuracy and fairness of TD algorithms & ML models
    """
    results = pd.DataFrame(
        0,
        index=pd.MultiIndex.from_product(
            [map(name, ml_models), map(name, td_algorithms)]
        ),
        columns=list(map(name, accuracy_metrics + fairness_metrics)),
    )
    n_folds = 10
    test_size = 0.5
    for algo in td_algorithms:
        predictions = pd.Series(algo.run(df).labels, index=df["y"].index)
        for model in ml_models:
            index = (name(model), name(algo))
            model.fit(df.x, df["y"])
            base_predictions = model.predict(df.x)
            base_accuracies = {
                name(metric): metric(df["y"], base_predictions)
                for metric in accuracy_metrics
            }
            base_fairnesses = {
                name(metric): metric.compute(base_predictions, df["s"], df["y"])
                for metric in fairness_metrics
            }
            for seed in range(n_folds):
                train_i, test_i = train_test_split(
                    df.x.index, test_size=test_size, shuffle=True, random_state=seed
                )
                model.fit(df.x.loc[train_i], predictions.loc[train_i])
                down_predictions = model.predict(df.x.loc[test_i])
                for metric in accuracy_metrics:
                    results.loc[index, name(metric)] += (
                        metric(df["y"].loc[test_i], down_predictions)
                        - base_accuracies[name(metric)]
                    )
                for metric in fairness_metrics:
                    results.loc[index, name(metric)] += (
                        metric.compute(
                            down_predictions, df["s"].loc[test_i], df["y"].loc[test_i]
                        )
                        - base_fairnesses[name(metric)]
                    )

    results *= 100 / n_folds
    results.columns = map(title_case, results.columns)
    results.index = results.index.set_levels(
        map(title_case, results.index.levels[0]), level=0
    )
    results.index = results.index.set_levels(
        map(title_case, results.index.levels[1]), level=1
    )
    results.index.names = ["ML Model", "TD Algorithm"]
    print(results.to_string())
