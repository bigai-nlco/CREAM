import os
import json
import argparse

from metrics import (
    qa_f1_score,
    rouge_score,
    classification_score,
    retrieval_score,
    count_score,
    code_sim_score,
)


dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--scaling_type', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    return parser.parse_args(args)

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    path = args.input_path

    dataset_name = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", \
                    "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    for dataset in dataset_name:
        filepath = os.path.join(path, f'{dataset}-{args.scaling_type}.jsonl')
        print("Evaluating on:", filepath)

        try:
            with open(f"{filepath}", "r", encoding="utf-8") as f:
                json_data = f.read()
                all_data = json.loads(json_data)
        except Exception as e:
            print(e)
            continue
        
        predictions, answers, lengths = [], [], []
        for data in all_data:
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data["all_classes"]
            if "length" in data:
                lengths.append(data["length"])

        scores[dataset] = scorer(dataset, predictions, answers, all_classes)

    out_path = f"{args.output_dir}/{args.model_name}_result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
