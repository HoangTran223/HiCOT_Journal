from utils import config, log, miscellaneous, seed
import os
import numpy as np
import basic_trainer
from HiCOT.HiCOT import HiCOT
from HiCOT.HiCOT_C import HiCOT_C
from HiCOT.HiCOT_Enhanced import HiCOT_Enhanced
import evaluations
import datasethandler
import scipy
import torch
from tqdm import tqdm

RESULT_DIR = 'results'
DATA_DIR = 'datasets'

def get_model_params_vector(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)

def set_model_params_vector(model, vector):
    pointer = 0
    for param in model.parameters():
        num_param = param.numel() 
        param.data.copy_(vector[pointer:pointer + num_param].view(param.size()))
        pointer += num_param

def random_directions(param_vector):
    direction1 = torch.randn_like(param_vector)
    direction2 = torch.randn_like(param_vector)
    return direction1, direction2


if __name__ == "__main__":
    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_model_argument(parser)
    config.add_training_argument(parser)
    config.add_eval_argument(parser)
    # Enhanced coherence parameters - CHỈ GIỮ LẠI MULTI-SCALE COHERENCE
    parser.add_argument('--weight_loss_coherence', type=float, default=1.0)
    parser.add_argument('--coherence_window_sizes', nargs='+', type=int, default=[2, 5, 10])
    parser.add_argument('--coherence_norm', type=str, default='pmi')
    parser.add_argument('--coherence_top_k', type=int, default=15)
    args = parser.parse_args()

    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR, current_time)
    miscellaneous.create_folder_if_not_exist(current_run_dir)

    config.save_config(args, os.path.join(current_run_dir, 'config.txt'))
    seed.seedEverything(args.seed)
    print(args)

    if args.dataset in ['YahooAnswers', '20NG', 'AGNews', 'IMDB', 'SearchSnippets', 'GoogleNews']:
        read_labels = True
    else:
        read_labels = False
    print(f"read labels = {read_labels}")

    ## Set contextual_embed = False
    dataset = datasethandler.BasicDatasetHandler(
        os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
        as_tensor=True, contextual_embed=False, args=args)

    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, args.dataset, "word_embeddings.npz")).toarray()
    
    if args.model == "HiCOT":
        model = HiCOT(vocab_size=dataset.vocab_size,
                        data_name=args.dataset,
                        num_topics=args.num_topics,
                        dropout=args.dropout,
                        pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                        weight_loss_ECR=args.weight_ECR,
                        alpha_ECR=args.alpha_ECR,
                        weight_loss_TP=args.weight_loss_TP,
                        weight_loss_DT= args.weight_loss_DT,
                        alpha_TP=args.alpha_TP,
                        alpha_DT=args.alpha_DT,
                        beta_temp=args.beta_temp,
                        vocab=dataset.vocab,
                        weight_loss_CLC=args.weight_loss_CLT,
                        max_clusters=args.max_clusters,
                        weight_loss_CLT = args.weight_loss_CLT,
                        threshold_epoch=args.threshold_epoch,
                        threshold_cluster=args.threshold_cluster,
                        doc_embeddings=torch.tensor(dataset.train_doc_embeddings).float().to(args.device),
                        method_CL=args.method_CL,
                        metric_CL=args.metric_CL
                        ) 
        model = model.to(args.device)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")

    elif args.model == "HiCOT_C":
        model = HiCOT_C(
            vocab_size=dataset.vocab_size,
            data_name=args.dataset,
            num_topics=args.num_topics,
            dropout=args.dropout,
            pretrained_WE=pretrainWE if args.use_pretrainWE else None,
            weight_loss_ECR=args.weight_ECR,
            alpha_ECR=args.alpha_ECR,
            weight_loss_TP=args.weight_loss_TP,
            weight_loss_DT=args.weight_loss_DT,
            alpha_TP=args.alpha_TP,
            alpha_DT=args.alpha_DT,
            beta_temp=args.beta_temp,
            vocab=dataset.vocab,
            weight_loss_CLC=args.weight_loss_CLC,
            max_clusters=args.max_clusters,
            weight_loss_CLT=args.weight_loss_CLT,
            threshold_epoch=args.threshold_epoch,
            threshold_cluster=args.threshold_cluster,
            doc_embeddings=torch.tensor(dataset.train_doc_embeddings).float().to(args.device),
            method_CL=args.method_CL,
            metric_CL=args.metric_CL,
            # Thêm các tham số cho loss_coherence
            weight_loss_coherence=getattr(args, 'weight_loss_coherence', 1.0),
            cooc_norm=getattr(args, 'cooc_norm', 'log'),
            train_data=dataset.train_bow,  # Đảm bảo truyền đúng train_bow
        )
        model = model.to(args.device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")

    elif args.model == "HiCOT_Enhanced":
        # Validate coherence parameters
        print(f"Coherence parameters:")
        print(f"  weight_loss_coherence: {getattr(args, 'weight_loss_coherence', 1.0)}")
        print(f"  coherence_window_sizes: {getattr(args, 'coherence_window_sizes', [2, 5, 10])}")
        print(f"  coherence_norm: {getattr(args, 'coherence_norm', 'pmi')}")
        print(f"  coherence_top_k: {getattr(args, 'coherence_top_k', 15)}")
        
        model = HiCOT_Enhanced(
            vocab_size=dataset.vocab_size,
            data_name=args.dataset,
            num_topics=args.num_topics,
            dropout=args.dropout,
            pretrained_WE=pretrainWE if args.use_pretrainWE else None,
            weight_loss_ECR=args.weight_ECR,
            alpha_ECR=args.alpha_ECR,
            weight_loss_TP=args.weight_loss_TP,
            weight_loss_DT=args.weight_loss_DT,
            alpha_TP=args.alpha_TP,
            alpha_DT=args.alpha_DT,
            beta_temp=args.beta_temp,
            vocab=dataset.vocab,
            weight_loss_CLC=args.weight_loss_CLC,
            max_clusters=args.max_clusters,
            weight_loss_CLT=args.weight_loss_CLT,
            threshold_epoch=args.threshold_epoch,
            threshold_cluster=args.threshold_cluster,
            doc_embeddings=torch.tensor(dataset.train_doc_embeddings).float().to(args.device),
            method_CL=args.method_CL,
            metric_CL=args.metric_CL,
            # CHỈ GIỮ LẠI Multi-Scale Coherence parameters
            weight_loss_coherence=getattr(args, 'weight_loss_coherence', 1.0),
            coherence_window_sizes=getattr(args, 'coherence_window_sizes', [2, 5, 10]),
            coherence_norm=getattr(args, 'coherence_norm', 'pmi'),
            coherence_top_k=getattr(args, 'coherence_top_k', 15),
            train_data=dataset.train_bow,
        )
        model = model.to(args.device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")
        
        # Test coherence loss immediately after model creation
        print("Testing coherence loss computation...")
        with torch.no_grad():
            test_loss = model.get_loss_multi_scale_coherence()
            print(f"Initial coherence loss: {test_loss}")

    else:
        print(f"Wrong model")


    # Create a trainer
    trainer = basic_trainer.BasicTrainer(model, model_name=args.model,
                                            epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size,
                                            device=args.device
                                            )


    # Train the model
    trainer.train(dataset)

    # Save embeddings
    trainer.save_embeddings(current_run_dir)

    # Save beta, theta and top words
    beta = trainer.save_beta(current_run_dir)
    train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)
    top_words_10 = trainer.save_top_words(
        dataset.vocab, 10, current_run_dir)
    top_words_15 = trainer.save_top_words(
        dataset.vocab, 15, current_run_dir)
    top_words_20 = trainer.save_top_words(
        dataset.vocab, 20, current_run_dir)
    top_words_25 = trainer.save_top_words(
        dataset.vocab, 25, current_run_dir)

    # Argmax of train and test theta
    train_theta_argmax = train_theta.argmax(axis=1)
    test_theta_argmax = test_theta.argmax(axis=1)        

    TD_15 = evaluations.compute_topic_diversity(
        top_words_15, _type="TD")
    print(f"TD_15: {TD_15:.5f}")


    # Evaluating clustering
    if read_labels:
        clustering_results = evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])
    
    
    TC_15_list, TC_15 = evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_15.txt'))
    print(f"TC_15: {TC_15:.5f}")

    filename = f"results_{args.dataset}_topics{args.num_topics}_epochs{args.epochs}_model{args.model}.txt"
    filename = filename.replace(' ', '_')
    filepath = os.path.join(current_run_dir, filename)
    with open(filepath, 'w') as f:
        if read_labels:
            f.write(f"NMI: {clustering_results['NMI']}\n")
            f.write(f"Purity: {clustering_results['Purity']}\n")
        else:
            f.write("NMI: N/A\n")
            f.write("Purity: N/A\n")
        f.write(f"TD_15: {TD_15:.5f}\n")
        f.write(f"TC_15: {TC_15:.5f}\n")

    print(f"Done in {filepath}")



