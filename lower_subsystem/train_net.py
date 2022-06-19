import logging
import os
from collections import OrderedDict
import wandb
import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import launch, default_argument_parser, default_setup, default_writers, PeriodicCheckpointer
from detectron2.evaluation import inference_on_dataset, print_csv_format, COCOEvaluator
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer, build_lr_scheduler
from detectron2.utils import comm
from detectron2.utils.events import EventStorage

from utils.early_stopping import EarlyStopping
from utils.setup_source_dataset import setup_source_dataset
from utils.setup_target_val_test_dataset import setup_target_val_test_dataset
from utils.setup_target_dataset import setup_target_dataset
from utils.albumentations_mapper import AlbumentationsMapper
from utils.loss import MeanTrainLoss

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return COCOEvaluator(dataset_name, output_dir=output_folder)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        mapper = AlbumentationsMapper(cfg, is_train=False)
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        evaluator = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    return results


def do_train(cfg, model, patience, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter, max_to_keep=1)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    mapper = AlbumentationsMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    mean_train_loss = MeanTrainLoss()
    early_stopping = EarlyStopping(patience=patience)
    iters_per_epoch = cfg.SOLVER.ITERS_PER_EPOCH
    epoch = 0

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, start_iter + max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            mean_train_loss.update(loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            periodic_checkpointer.step(iteration)

            # At the end of each epoch
            if (iteration - start_iter + 1) % iters_per_epoch == 0:
                # Train loss averaged over the epoch
                with storage.name_scope("Train losses"):
                    storage.put_scalars(total_loss=mean_train_loss.get_total_loss(),
                                        **mean_train_loss.get_losses(),
                                        smoothing_hint=False)
                mean_train_loss.reset()

                # COCO Evaluation
                test_results = do_test(cfg, model)
                for dataset_name, dataset_results in test_results.items():
                    for name, results in dataset_results.items():
                        with storage.name_scope(f"{dataset_name}_{name}"):
                            storage.put_scalars(**results, smoothing_hint=False)

                # Early stopping and best model checkpointing
                print("#######################################")
                metric = test_results["new_dataset_validation"]["segm"]["AP"]
                early_stopping.on_epoch_end(metric, epoch)
                storage.put_scalar(name='best_segm_AP', value=early_stopping.best_value, smoothing_hint=False)
                if early_stopping.has_improved:
                    print(f"New best model -> epoch: {epoch} -> segm AP: {metric}")
                    checkpointer.save("best_model")
                print("#######################################")

                comm.synchronize()

                # Write events to EventStorage
                for writer in writers:
                    writer.write()

                if early_stopping.should_stop():
                    print(f"Early stopping at epoch {epoch}")
                    print(f"Best model was at epoch {early_stopping.best_epoch} "
                          f"with {early_stopping.best_value} segm AP")
                    break

                epoch += 1


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)  # to allow merging new keys
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # Init Weight & Biases and sync with Tensorboard
    wandb.init(project="GrapeDnT", sync_tensorboard=True)
    # Save config.yaml on wandb
    wandb.save(os.path.join(cfg.OUTPUT_DIR, "config.yaml"))

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if args.eval_only:
        setup_target_val_test_dataset()
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # Evaluate
        return do_test(cfg, model)

    # Register datasets
    setup_source_dataset()
    setup_target_val_test_dataset()

    if cfg.PSEUDOMASKS.ENABLED:
        for i in range(len(cfg.PSEUDOMASKS.DATASET_NAME)):
            setup_target_dataset(
                cfg.PSEUDOMASKS.DATASET_NAME[i], 
                cfg.PSEUDOMASKS.DATA_FOLDER[i], 
                cfg.PSEUDOMASKS.EXTENSION[i], 
                cfg.PSEUDOMASKS.PSEUDOMASKS_FOLDER[i])

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )    

    # Train
    do_train(cfg, model, patience=args.patience, resume=args.resume)
    return


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--patience", default=-1, help="Patience value for early stopping (default -1 -> No early stopping")
    args = parser.parse_args()
    if args.wandb:
        os.environ['WANDB_MODE'] = 'enabled'
    else:
        os.environ['WANDB_MODE'] = 'disabled'

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
