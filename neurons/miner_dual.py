import asyncio
import math
import os
import random
import typing
import wandb
import torch
import argparse
import constants
from taoverse.metagraph import utils as metagraph_utils
from taoverse.model.storage.chain.chain_model_metadata_store import (
    ChainModelMetadataStore,
)
from taoverse.model.storage.hugging_face.hugging_face_model_store import (
    HuggingFaceModelStore,
)
from taoverse.model.storage.model_metadata_store import ModelMetadataStore
from taoverse.utilities.enum_action import IntEnumAction
from competitions.data import CompetitionId
import pretrain as pt
import bittensor as bt
from transformers import PreTrainedModel
import datetime as dt
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--hf_repo_id", type=str)
    parser.add_argument("--avg_loss_upload_threshold", type=float, default=2)
    parser.add_argument("--model_dir", default=os.path.join(constants.ROOT_DIR, "local-models/"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load_best", action="store_true")
    parser.add_argument("--load_uid", type=int, default=None)
    parser.add_argument("--load_model_dir", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--bs", type=int, default=constants.batch_size)
    parser.add_argument("--sl", type=int, default=constants.SEQUENCE_LENGTH_2)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--pages_per_epoch", type=int, default=10)
    parser.add_argument("--netuid", type=str, default=constants.SUBNET_UID)
    parser.add_argument("--use_hotkey_in_hash", action="store_true")
    parser.add_argument("--list_competitions", action="store_true")
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    config = bt.config(parser)
    return config

async def load_starting_model(config: bt.config, metagraph: bt.metagraph, metadata_store: ModelMetadataStore, kwargs: typing.Dict[str, typing.Any]) -> PreTrainedModel:
    if config.load_best:
        model = await pt.mining.load_best_model(config.model_dir, config.competition_id, metagraph=metagraph, metadata_store=metadata_store)
        bt.logging.success(f"Training with best model from competition: {config.competition_id}. Model={str(model)}")
        return model
    if config.load_uid is not None:
        model = await pt.mining.load_remote_model(config.load_uid, config.model_dir, metagraph=metagraph, metadata_store=metadata_store)
        bt.logging.success(f"Training with model from uid: {config.load_uid}. Model={str(model)}")
        return model
    if config.load_model_dir:
        model = pt.mining.load_local_model(config.load_model_dir, kwargs)
        bt.logging.success(f"Training with model from disk. Model={str(model)}")
        return model
    if config.load_model:
        model = pt.mining.load_gpt2_model(config.load_model)
        bt.logging.success(f"Training with model from disk. Model={str(model)}")
        return model
    model = pt.model.get_model()
    bt.logging.success(f"Training from scratch. Model={str(model)}")
    return model

async def main(config: bt.config):
    bt.logging(config=config)
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    chain_metadata_store = ChainModelMetadataStore(subtensor=subtensor, subnet_uid=config.netuid, wallet=wallet)
    my_uid = None
    if not config.offline:
        my_uid = 126#metagraph_utils.assert_registered(wallet, metagraph)
        HuggingFaceModelStore.assert_access_token_exists()
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = pt.mining.model_path(config.model_dir, run_id)
    os.makedirs(model_dir, exist_ok=True)
    use_wandb = False
    if not config.offline:
        if config.wandb_project is None or config.wandb_entity is None:
            bt.logging.warning("Wandb project or entity not specified. This run will not be logged to wandb")
        else:
            use_wandb = True
    config.competition_id = CompetitionId.B3_MODEL
    model_constraints = constants.MODEL_CONSTRAINTS_BY_COMPETITION_ID.get(CompetitionId.B3_MODEL, None)
    if not model_constraints:
        raise RuntimeError(f"No competition found for {config.competition_id}")
    kwargs = model_constraints.kwargs.copy()
    tokenizer = pt.model.load_tokenizer(model_constraints, cache_dir=config.model_dir)
    model = await load_starting_model(config, metagraph, chain_metadata_store, kwargs)
    model = model.train()
    
    # Move the model to the appropriate device(s) - use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        bt.logging.success(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    model = model.to(config.device) # Move the model after DataParallel
    
    print("models init ok~")
    bt.logging.success(f"Saving model to path: {model_dir}.")
    pt.mining.save(model, model_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    wandb_run = None
    epoch_step = 0
    global_step = 0
    n_acc_steps = 0
    best_avg_loss = math.inf
    accumulation_steps = config.accumulation_steps
    try:
        while epoch_step < config.num_epochs or config.num_epochs == -1:
            epoch_loss = 0.0
            bt.logging.success(f"Loading {config.pages_per_epoch} pages for training this epoch")
            random_pages = [random.randint(1, pt.dataset.SubsetFalconLoader.max_pages) for _ in range(config.pages_per_epoch)]
            bt.logging.success(f"Load pages done")
            loader = pt.dataset.SubsetFineWebEdu2Loader(
                batch_size=config.bs,
                sequence_length=config.sl,
                num_pages=config.pages_per_epoch,
                tokenizer=tokenizer,
            )
            bt.logging.success(f" loader done")
            n_batches = 0
            optimizer.zero_grad()
            for i, batch in enumerate(loader):
                inputs = torch.from_numpy(batch).to(config.device)
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss / accumulation_steps
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    n_acc_steps += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    bt.logging.success(f"Step: {n_acc_steps} loss: {outputs.loss.detach().item()}")
                    if use_wandb:
                        wandb_run.log({"loss": outputs.loss.detach(), "n_batches": n_batches}, step=n_acc_steps)
                torch.cuda.empty_cache()
                n_batches += 1
                global_step += 1
                epoch_loss += outputs.loss.detach().item()
            avg_loss = epoch_loss / n_batches
            bt.logging.success(f"Epoch: {epoch_step} average loss: {avg_loss}")
            epoch_step += 1
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                bt.logging.success(f"New best average loss: {best_avg_loss}.")
                bt.logging.success(f"Saving model to path: {model_dir}.")
                if isinstance(model, torch.nn.DataParallel):  # Check if it's wrapped in DataParallel
                  pt.mining.save(model.module, model_dir) # Save model.module
                else:
                  pt.mining.save(model, model_dir)

        bt.logging.success("Finished training")
        if not config.offline:
            if best_avg_loss < config.avg_loss_upload_threshold:
                bt.logging.success(f"Trained model had a best_avg_loss of {best_avg_loss} which is below the threshold of {config.avg_loss_upload_threshold}. Uploading to hugging face. ")
                model_to_upload = pt.mining.load_local_model(model_dir, model_constraints.kwargs)
                await pt.mining.push(
                    model_to_upload,
                    config.hf_repo_id,
                    wallet,                    
                    config.competition_id,
                    60,
                    metadata_store=chain_metadata_store,
                    #use_hotkey_in_hash=False,
                )
            else:
                bt.logging.success(f"This training run achieved a best_avg_loss={best_avg_loss}, which did not meet the upload threshold. Not uploading to hugging face.")
        else:
            bt.logging.success("Not uploading to hugging face because --offline was specified.")
    finally:
        if wandb_run:
            wandb_run.finish()

if __name__ == "__main__":
    config = get_config()
    if config.list_competitions:
        print(constants.COMPETITION_SCHEDULE_BY_BLOCK)
    else:
        print(config)
        asyncio.run(main(config))
