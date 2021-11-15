import torch
from callbacks import WandBCallback, ModelCheckpointer

model = {
    "Model params": [
        {
            "name": "n_d",
            "type": int,
            "default": 8,
            "help": "Width of the decision prediction layer. Bigger values gives more capacity to the model with "
            "the risk of overfitting. Values typically range from 8 to 64.",
        },
        {
            "name": "n_a",
            "type": int,
            "default": 8,
            "help": "Width of the attention embedding for each mask. According to the paper n_d=n_a is usually "
            "a good choice. (default=8)",
        },
        {
            "name": "n_steps",
            "type": int,
            "default": 3,
            "help": "Number of steps in the architecture (usually between 3 and 10)",
        },
        {
            "name": "gamma",
            "type": float,
            "default": 1.3,
            "help": "This is the coefficient for feature reusage in the masks. A value close to 1 will make mask "
            "selection least correlated between layers. Values range from 1.0 to 2.0.",
        },
        {
            "name": "cat_idxs",
            "type": list,
            "default": [],
            "help": "Mandatory for embeddings. List of categorical feature indices",
        },
        {
            "name": "cat_dims",
            "type": list,
            "default": [],
            "help": "List of categorical features number of modalities "
            "(number of unique values for a categorical feature) /!\\ no new modalities can be predicted",
        },
        {
            "name": "cat_emb_dim",
            "type": list,
            "default": 1,
            "help": "List of embeddings size for each categorical features. (Default=2)",
        },
        {
            "name": "n_independent",
            "type": int,
            "default": 2,
            "help": "Number of independent Gated Linear Units layers at each step. Usual values range from 1 to 5.",
        },
        {
            "name": "seed",
            "type": int,
            "default": 42,
            "help": "Random seed for reproducibility",
        },
        {
            "name": "momentum",
            "type": float,
            "default": 0.02,
            "help": "Momentum for batch normalization, typically ranges from 0.01 to 0.4 (default=0.02)",
        },
        {
            "name": "clip_value",
            "type": float,
            "default": None,
            "help": "If a float is given this will clip the gradient at clip_value",
        },
        {
            "name": "lambda_sparse",
            "type": float,
            "default": 1e-3,
            "help": "This is the extra sparsity loss coefficient as proposed in the original paper. "
            "The bigger this coefficient is, the sparser your model will be in terms of feature selection. Depending on the difficulty of your problem, reducing this value could help.",
        },
        {
            "name": "optimizer_fn",
            "type": torch.optim.Optimizer,
            "default": torch.optim.Adam,
            "help": "Pytorch optimizer function",
        },
        {
            "name": "optimizer_params",
            "type": dict,
            "default": dict(lr=2e-2),
            "help": "Parameters compatible with optimizer_fn used initialize the optimizer. Since we have Adam as our default optimizer, we use this to define the initial learning rate used for training. As mentionned in the original paper, a large initial learning rate of 0.02 with decay is a good option.",
        },
        {
            "name": "scheduler_fn",
            "default": None,
            "help": "Pytorch Scheduler to change learning rates during training.",
        },
        {
            "name": "scheduler_params",
            "type": dict,
            "default": dict(),
            "help": 'Dictionnary of parameters to apply to the scheduler_fn. Ex : {"gamma": 0.95, "step_size": 10}',
        },
        {
            "name": "model_name",
            "type": str,
            "default": "DreamQuarkTabNet",
            "help": "Name of the model used for saving in disk, you can customize this to easily retrieve "
            "and reuse your trained models.",
        },
        {
            "name": "verbose",
            "type": int,
            "default": 1,
            "help": "Verbosity for notebooks plots, set to 1 to see every epoch, 0 to get None.",
        },
        {
            "name": "device_name",
            "type": str,
            "default": "auto",
            "help": "'cpu' for cpu training, 'gpu' for gpu training, 'auto' to automatically detect gpu.",
        },
        {
            "name": "mask_type",
            "type": str,
            "default": "sparsemax",
            "help": 'Either "sparsemax" or "entmax" : this is the masking function to use for selecting features.',
        },
        {
            "name": "n_shared_decoder",
            "type": int,
            "default": 1,
            "help": "Number of shared GLU block in decoder, this is only useful for TabNetPretrainer.",
        },
        {
            "name": "n_indep_decoder",
            "type": int,
            "default": 1,
            "help": "Number of independent GLU block in decoder, this is only useful for TabNetPretrainer.",
        },
    ],
    "Fit params": [
        {
            "name": "max_epochs",
            "type": int,
            "default": 1000,
            "help": "Maximum number of epochs for trainng.",
        },
        {
            "name": "patience",
            "type": int,
            "default": 15,
            "help": """
                    Number of consecutive epochs without improvement before performing early stopping.
                    If patience is set to 0, then no early stopping will be performed.
                    Note that if patience is enabled, then best weights from best epoch will automatically be loaded at the end of fit.
                    """,
        },
        {
            "name": "batch_size",
            "type": int,
            "default": 512,
            "help": "Number of examples per batch. Large batch sizes are recommended.",
        },
        {
            "name": "virtual_batch_size",
            "type": int,
            "default": 128,
            "help": 'Size of the mini batches used for "Ghost Batch Normalization". /!\ virtual_batch_size should '
            "divide batch_size",
        },
        {
            "name": "num_workers",
            "type": int,
            "default": 0,
            "help": "Number or workers used in torch.utils.data.Dataloader",
        },
        {
            "name": "drop_last",
            "type": bool,
            "default": False,
            "help": "Whether to drop last batch if not complete during training",
        },
        {
            "name": "callbacks",
            "type": list,
            "default": [],
            "help": "List of custom callbacks. Early stopping is built-in",
        },
        {
            "name": "pretraining_ratio",
            "type": float,
            "help": """
                      /!\ TabNetPretrainer Only : Percentage of input features to mask during pretraining.
                      Should be between 0 and 1. The bigger the harder the reconstruction task is.
                    """,
        },
    ],
}
