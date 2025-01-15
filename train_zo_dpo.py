import argparse

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from ZOPrO.ZO.custom_datasets import PrefDataset,ClassificationDataset
from mezo_dataset import GLUEDataModule
from trainers import TrainerZODPOPrime, TrainerZOPrime
from modeling_roberta import RobertaConfig
from models import RobertaModelForPromptFinetuning
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    OPTForSequenceClassification
)

# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data.csv")
    parser.add_argument("--datasets", type=str, default="sst2")
    parser.add_argument("--model-base", type=str, default="roberta-large")
    parser.add_argument("--model-ref", type=str, default="roberta-large", help="Reference model")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--task", type=str, default='classification', help="classification or DPO")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--step-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--zo-eps", type=float, default=1e-3)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--optimizer", type=str, default='SGD')
    args = parser.parse_args()

    # model_base = AutoModelForCausalLM.from_pretrained(args.model_base,
    #                                                   torch_dtype=torch.bfloat16,
    #                                                   device_map="auto")
    # model_ref = AutoModelForCausalLM.from_pretrained(args.model_ref,
    #                                                  torch_dtype=torch.bfloat16,
    #                                                  device_map="auto")

    # tokenizer = AutoTokenizer.from_pretrained(args.model_ref,use_auth_token=True)
    # tokenizer.pad_token_id = 0

    # train_data = PrefDataset("hh", "train", tokenizer, max_length=None)
    # eval_data = PrefDataset("hh", "test", tokenizer, max_length=None)

    # model, model_base, model_ft, train_data, eval_data, step_accum, lr, weight_decay, epochs, zo_eps
    if args.task == 'DPO':
        # auto should be enough to enable distributed inference, at least for zo training
        model_base = AutoModelForCausalLM.from_pretrained(args.model_base,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map="auto")
        model_ref = AutoModelForCausalLM.from_pretrained(args.model_ref,
                                                         torch_dtype=torch.bfloat16,
                                                         device_map="auto")

        tokenizer = AutoTokenizer.from_pretrained(args.model_ref, use_auth_token=True)
        tokenizer.pad_token_id = 0

        train_data = PrefDataset("hh", "train", tokenizer, max_length=None)
        eval_data = PrefDataset("hh", "test", tokenizer, max_length=None)
        trainer = TrainerZODPOPrime(
            model2train=model_base,  # model to train
            model_ref=model_ref,  # model used as ref
            output_dir=args.output,
            tokenizer=tokenizer,
            train_data=train_data,
            eval_data=eval_data,
            batch_size=args.batch_size,
            step_accum=args.step_accum,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            zo_eps=args.zo_eps,
            use_wandb=args.use_wandb,
            dpo_beta=args.dpo_beta,
            use_cpu=args.debug,
            optimizer=args.optimizer,
        )

    elif args.task == 'classification':
        #just test the template, so I use model_ref rather than model_base. Remember fix later on

        model_base = None
        train_data = None
        test_data = None
        num_labels = 2
        if 'Llama' in args.model_base:
            model_base = AutoModelForCausalLM.from_pretrained(args.model_base,
                                                              torch_dtype=torch.bfloat16,
                                                              device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(args.model_ref, use_auth_token=True)

        elif 'roberta' in args.model_base:
            config = RobertaConfig.from_pretrained(
                'roberta-large',
                num_labels=num_labels,
                finetuning_task=args.datasets)

            model_base = RobertaModelForPromptFinetuning.from_pretrained(
                "roberta-large",
                config=config,
                device_map=0
            )

            config = AutoConfig.from_pretrained(args.model_base, num_labels=num_labels)
            model_base=AutoModelForSequenceClassification.from_pretrained(args.model_base, config=config)


        elif 'opt' in args.model_base:
            model_base = OPTForSequenceClassification.from_pretrained(
                args.model_base,
                num_labels=num_labels,
                torch_dtype=torch.float16,
                device_map=0
            )



        if args.datasets == 'hh':
            train_data = ClassificationDataset("hh", "train", tokenizer, max_length=None)
            eval_data = ClassificationDataset("hh", "test", tokenizer, max_length=None)

        else:
            dm = GLUEDataModule(
                model_name_or_path=args.model_base,
                task_name=args.datasets,
                train_batch_size=args.batch_size,
                validation_sample_size=args.batch_size,
                eval_batch_size=1,
                soft_prompt=True
            )
            dm.setup("fit")

            trainer = TrainerZOPrime(
                model2train=model_base,  # model to train
                #model_ref=model_ref,  # model used as ref
                output_dir=args.output,
                tokenizer=None,
                task_name=args.datasets,
                train_data=dm.train_dataloader(),
                eval_data=dm.val_dataloader(),
                num_labels = dm.num_labels,
                batch_size=args.batch_size,
                step_accum=args.step_accum,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                zo_eps=args.zo_eps,
                use_wandb=args.use_wandb,
                use_cpu=args.debug,
                optimizer=args.optimizer,
            )
            trainer.train()
    else:
        raise ValueError(f"Unsupported task name: {args.task}")
    #trainer.train()


if __name__ == "__main__":
    main()
