import sys
import json
import torch
from random import shuffle, seed
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from saf_datasets import EntailmentBankDataSet
from langvae import LangVAE
from langvae.data_conversion.tokenization import TokenizedDataSet, TokenizedAnnotatedDataSet
from langspace.metrics.disentanglement import DisentanglementMetric as Metric
from langspace.probe import DisentanglementProbe

GEN_FACTORS = {
    "direction": ["ARGM-DIR"],
    "because": ["ARGM-CAU"],
    "purpose": ["ARGM-PRP", "ARGM-PNC", "ARGM-GOL"],
    "more": ["ARGM-EXT"],
    "location": ["ARGM-LOC"],
    "argument": ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4"],
    "manner": ["ARGM-MNR"],
    "can": ["ARGM-MOD"],
    "argm-prd": ["ARGM-PRD"],
    "empty": ["O"],
    "negation": ["ARGM-NEG"],
    "verb": ["V"],
    "if-then": ["ARGM-ADV", "ARGM-DIS"],
    "time": ["ARGM-TMP"],
    "C-ARG": ["C-ARG1", "C-ARG0", "C-AGR2"]
}

for factor in GEN_FACTORS:
    GEN_FACTORS[factor] = ["I-" + lbl if (lbl != "O") else lbl for lbl in GEN_FACTORS[factor]]


def load_model(model_name: str) -> LangVAE:
    if (model_name.startswith("neuro-symbolic-ai")):
        model = LangVAE.load_from_hf_hub(model_name)
    else:
        model = LangVAE.load_from_folder(model_name)

    model.eval()

    if (torch.cuda.is_available()):
        model.encoder.to("cuda")
        model.decoder.to("cuda")
        # model.encoder.init_pretrained_model()
        # model.decoder.init_pretrained_model()

    return model


def load_eb_data() -> tuple[EntailmentBankDataSet, dict[str, list[str]]]:
    eb_dataset = EntailmentBankDataSet.from_resource("pos+lemma+ctag+dep+srl#expl_only-noreps")
    annotations = {"srl_f": eb_dataset.annotations["srl"]}
    seed(0)
    shuffle(eb_dataset.data)

    for sent in eb_dataset:
        for token in sent.tokens:
            srl = token.annotations["srl"]
            token_annot = [lbl for lbl in srl if (lbl != "O")][0] if (len(set(srl)) > 1) else srl[0]
            token.annotations["srl_f"] = token_annot

    return eb_dataset, annotations


def calc_bleu(decoded: dict) -> float:
    bleu_scores = list()

    for sent in decoded:
        bleu_scores.append(sentence_bleu([ref.split() for ref in decoded[sent]], sent.split()))

    bleu_scores = [score for score in bleu_scores if (score > 0)]

    return sum(bleu_scores) / len(bleu_scores)


def calc_disentanglement(model: LangVAE, dataset: EntailmentBankDataSet, annotations: dict[str, list[str]]):

    metrics = [Metric.Z_DIFF, Metric.Z_MIN_VAR, Metric.MIG, Metric.INFORMATIVENESS, Metric.COMPLETENESS]
    disentang_report = DisentanglementProbe(model, dataset, sample_size=200, metrics=metrics, gen_factors=GEN_FACTORS,
                                            annotations=annotations).report()

    return disentang_report


def eval_model(model: LangVAE) -> dict:
    bsize = 10
    sample_size = 5
    results = dict()
    eb_dataset, annotations = load_eb_data()
    eb_dataset = eb_dataset[-200:]

    tok_ds = TokenizedAnnotatedDataSet(eb_dataset, model.decoder.tokenizer, model.decoder.max_len, annotations)

    decoded = dict()
    for k in range(sample_size):
        for i in tqdm(range(0, 200, bsize)):
            latent, cvars_emb = model.encode_z(tok_ds[i:i + bsize].data, {"srl_f": tok_ds[i:i + bsize].srl_f})
            dec = [sent for sent in model.decode_sentences(latent, cvars_emb)]
            orig = [sent.surface for sent in eb_dataset[i:i + bsize]]
            for j in range(len(orig)):
                if (orig[j] not in decoded):
                    decoded[orig[j]] = list()
                decoded[orig[j]].append(dec[j])

    # print(decoded)
    avg_bleu_score = calc_bleu(decoded)
    results["bleu"] = avg_bleu_score

    results["disentanglement"] = calc_disentanglement(model, eb_dataset, annotations).to_dict("records")

    return results


def main(argv):
    results = dict()

    if (len(argv) > 1):
        with open(argv[1]) as models_list_file:
            model_names = [name for name in json.load(models_list_file)]
    else:
        model_names = ["neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128"]

    for model_name in model_names:
        print(f"Testing [{model_name}] ...", file=sys.stderr)
        model = load_model(model_name)
        results[model_name] = eval_model(model)

        json.dump(results, open("eval_results.json", "w"), indent=2)



if __name__ == "__main__":
    main(sys.argv)