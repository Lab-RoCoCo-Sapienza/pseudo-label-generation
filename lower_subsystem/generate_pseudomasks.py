import logging
import os

from detectron2.config import get_cfg
from detectron2.engine import launch, default_argument_parser, default_setup
from detectron2.modeling import build_model

from pseudo_labeling.mask_processing import process_pseudomasks
from pseudo_labeling.masks_from_bboxes import generate_masks_from_bboxes

logger = logging.getLogger("detectron2")


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

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    model_weights = cfg.PSEUDOMASKS.MODEL_WEIGHTS
    process_method = cfg.PSEUDOMASKS.PROCESS_METHOD
    assert process_method in ['none', 'dilation', 'slic', 'grabcut']
    num_datasets = len(cfg.PSEUDOMASKS.DATA_FOLDER)

    # Generate pseudo-masks
    for i in range(num_datasets):
        data_folder = cfg.PSEUDOMASKS.DATA_FOLDER[i]
        dest_folder = cfg.PSEUDOMASKS.PSEUDOMASKS_FOLDER[i]
        img_ext = cfg.PSEUDOMASKS.EXTENSION[i]
        
        print(f"Generating pseudo-masks for dataset {i + 1} out of {num_datasets}...")
        generate_masks_from_bboxes(cfg,
                                   data_folder=data_folder,
                                   dest_folder=dest_folder,
                                   model_weights=model_weights,
                                   img_ext=img_ext)

        # Post-process pseudo-masks
        if process_method in ['dilation', 'slic', 'grabcut']:
            print(f"Applying post-processing with '{process_method}' method to the pseudo-masks "
                    f"of dataset {i + 1} out of {num_datasets}...")
            process_pseudomasks(cfg,
                                method=process_method,
                                masks_folder=dest_folder,
                                data_path=data_folder,
                                output_path=dest_folder,
                                img_ext=img_ext)
    return


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
