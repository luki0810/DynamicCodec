# GANTrainer.py
import warnings
import torch
import argbind
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from audiotools import ml
from audiotools import AudioSignal
from audiotools.core import util
from audiotools.ml.decorators import Tracker, when, timer
from torch.utils.tensorboard import SummaryWriter
from audiotools.data.datasets import AudioDataset, AudioLoader, ConcatDataset
from audiotools.data import transforms  # audio data transform to tensor

import functools

import model
from model.build import DynamicTask, DynamicCodec
from model.utils.dynamic_argbind_loader import load_config_for_argbind


warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))

def _dump_args(args, save_path):
    if save_path.exists():
        try:
            os.remove(save_path)
            print(f"[INFO] Removed existing file: {save_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {save_path}: {e}")
    argbind.dump_args(args, save_path)


# in CLI, with no use of prefix, example: --precision=bf16, --use_ddp=true, --cpu=false
# amp: false
Accelerator = argbind.bind(ml.Accelerator)

# Optimizers for generator and discriminator
# generator/Adam.xx: xx
AdamW = argbind.bind(torch.optim.AdamW, group=["generator", "discriminator"])

# argbind scheduler
# generator/ExponentialLR.xx: xx
@argbind.bind("generator", "discriminator")  # 装饰器修饰函数
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Discriminator.xx: xx
Discriminator = argbind.bind(model.discriminator.Discriminator)

# Dataloader
# train/AudioDataset.xx: xx
DyAudioDataset = argbind.bind(AudioDataset, group=["train", "val"])
DyAudioLoader = argbind.bind(AudioLoader, group=["train", "val"])

filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(model.nn.loss, filter_fn=filter_fn)


# infinite loader for infinite steps
# 不关心epoch，只考虑steps
def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


# train/build_transform.xx: xx
@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    # 这里的to_cfm是为了：把字符串类名（如 "VolumeNorm"）变成已带参数的实例（依据 train/val 命名空间）
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]

    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


# train/build_dataset.xx: xx
@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    # Give one loader per key/value of dictionary, where
    # value is a list of folders. Create a dataset for each one.
    # Concatenate the datasets with ConcatDataset, which
    # cycles through them.
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: DynamicCodec
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss

    train_data: DyAudioDataset
    val_data: DyAudioDataset

    tracker: Tracker


@argbind.bind()
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    only_load_weights: bool = True,
):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not only_load_weights,
            # package === load full training state
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        # 分开两个文件夹，对应的需要看看save的逻辑
        if (Path(kwargs["folder"]) / "dynamiccodec").exists():
            generator, g_extra = DynamicCodec.load_from_folder(**kwargs)
        if (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)
    else:
        tracker.print(
            "No parameters specified for loading, randomly initialized generator and discriminator"
        )
        with argbind.scope(args):
            generator = DynamicTask.build_model()  # from args
        discriminator = Discriminator() if discriminator is None else discriminator

    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


@timer()
def train_loop(state, batch, accel, lambdas):
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )

    with accel.autocast():
        out = state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)
        # ===============================================
        model_loss_dict = out.get("loss", {}) or {}
        # ===============================================

    with accel.autocast():
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, signal)

    state.optimizer_d.zero_grad()
    accel.backward(output["adv/disc_loss"])
    accel.scaler.unscale_(state.optimizer_d)
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0
    )
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    with accel.autocast():
        output["stft/loss"] = state.stft_loss(recons, signal)
        output["mel/loss"] = state.mel_loss(recons, signal)
        output["waveform/loss"] = state.waveform_loss(recons, signal)
        (
            output["adv/gen_loss"],
            output["adv/feat_loss"],
        ) = state.gan_loss.generator_loss(recons, signal)

        # ===============================================
        for k, v in model_loss_dict.items():
            if not torch.is_tensor(v):
                v = torch.as_tensor(v, device=accel.device, dtype=torch.float32)
            else:
                v = v.to(accel.device)
            output[k] = v

        # weighted sum loss by lambdas
        weighted_terms = [w * output[k] for k, w in lambdas.items() if k in output]
        if len(weighted_terms) == 0:
            raise RuntimeError(
                f"No matching loss keys between lambdas and output. "
                f"lambdas keys={list(lambdas.keys())}, output keys={list(output.keys())}"
            )
        output["loss"] = sum(weighted_terms)
        output["loss"] = output["loss"].mean()
        # ===============================================

    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    signal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    return {
        "loss": state.mel_loss(recons, signal),
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
    }


# 这里的保存方式与load方法相对应
def checkpoint(state, save_iters, save_path, accel):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.generator).metadata = metadata

        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}",
            generator_extra,
            package=False,
        )
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(
            f"{save_path}/{tag}",
            discriminator_extra,
            package=False,
        )


# validation_step
def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


# save wav to tensorboard
@torch.no_grad()
def save_samples(state, val_idx, writer, accel):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    signal = state.train_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    audio_dict = {"recons": recons}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

    for key, batch_signal in audio_dict.items():
        for cur_idx in range(batch_signal.batch_size):
            batch_signal[cur_idx].cpu().write_audio_to_tb(
                f"{key}/sample_{cur_idx}.wav", writer, state.tracker.step
            )


class GANTrainer:
    """
    把原来的 train() 流程封装成一个类，便于在其它脚本中复用：
        from GANTrainer import GANTrainer
        ...
        trainer = GANTrainer(args, accel, log_path="ckpt", ...)
        trainer.run()
    """

    def __init__(
        self,
        args,
        accel: ml.Accelerator,
        seed: int = 0,
        log_path: str = "ckpt",
        num_iters: int = 250000,
        save_iters: list = [10000, 50000, 100000, 200000],
        sample_freq: int = 10000,
        valid_freq: int = 1000,
        batch_size: int = 12,
        val_batch_size: int = 10,
        num_workers: int = 8,
        val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
        lambdas: dict = None,
    ):
        self.args = args
        self.accel = accel
        self.seed = seed
        self.log_path = log_path
        self.num_iters = num_iters
        self.save_iters = save_iters
        self.sample_freq = sample_freq
        self.valid_freq = valid_freq
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.val_idx = val_idx
        self.lambdas = lambdas or {
            "mel/loss": 100.0,
            "adv/feat_loss": 2.0,
            "adv/gen_loss": 1.0,
            "vq/commitment_loss": 0.25,
            "vq/codebook_loss": 1.0,
        }

        self.writer = None
        self.tracker = None
        self.state: State = None
        self.train_dataloader = None
        self.val_dataloader = None

        # 完成初始化构建（相当于原来的 train 函数前半部分）
        self._setup()

    def _setup(self):
        # seed
        util.seed(self.seed)

        Path(self.log_path).mkdir(exist_ok=True, parents=True)
        self.writer = (
            SummaryWriter(log_dir=f"{self.log_path}/logs")
            if self.accel.local_rank == 0
            else None
        )
        self.tracker = Tracker(
            writer=self.writer,
            log_file=f"{self.log_path}/log.txt",
            rank=self.accel.local_rank,
        )

        if int(os.getenv("LOCAL_RANK", 0)) == 0:
            _dump_args(self.args, Path(self.log_path) / "args.yaml")

        # load model, optimizer, scheduler, data
        self.state = load(self.args, self.accel, self.tracker, self.log_path)

        # 再次 dump args 和日志提示
        if int(os.getenv("LOCAL_RANK", 0)) == 0:
            argbind.dump_args(self.args, Path(self.log_path) / "args.yaml")
            self.tracker.print("----------  writer and tracker created  ----------")
            self.tracker.print("----------  State module loaded  ----------")
            self.tracker.print("----------  Arguments saved to args.yaml  ----------")
            self.tracker.print(
                f"----------  Training for {self.num_iters} iterations  ----------"
            )

        self.train_dataloader = self.accel.prepare_dataloader(
            self.state.train_data,
            start_idx=self.state.tracker.step * self.batch_size,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=self.state.train_data.collate,
        )
        self.train_dataloader = get_infinite_loader(self.train_dataloader)
        self.val_dataloader = self.accel.prepare_dataloader(
            self.state.val_data,
            start_idx=0,
            num_workers=self.num_workers,
            batch_size=self.val_batch_size,
            collate_fn=self.state.val_data.collate,
            persistent_workers=True if self.num_workers > 0 else False,
        )
        self.tracker.print(
            "----------  Dataloader prepared (train & val)  ----------"
        )

        # 包装日志 / 进度条等（对应原来 train() 里的 wrap 部分）
        self._wrap_functions()

    def _wrap_functions(self):
        global train_loop, val_loop, validate, save_samples, checkpoint

        train_loop = self.tracker.log("train", "value", history=False)(
            self.tracker.track("train", self.num_iters, completed=self.state.tracker.step)(
                train_loop
            )
        )
        val_loop = self.tracker.track("val", len(self.val_dataloader))(val_loop)
        validate = self.tracker.log("val", "mean")(validate)

        # bind accel 到 save_samples 和 checkpoint
        save_samples = functools.partial(save_samples, accel=self.accel)
        checkpoint = functools.partial(checkpoint, accel=self.accel)

        # 只在 rank 0 上运行
        save_samples = when(lambda: self.accel.local_rank == 0)(save_samples)
        checkpoint = when(lambda: self.accel.local_rank == 0)(checkpoint)

    def run(self):
        # 对应原来的 train() 主循环部分
        with self.tracker.live:
            for self.tracker.step, batch in enumerate(
                self.train_dataloader, start=self.state.tracker.step
            ):
                train_loop(self.state, batch, self.accel, self.lambdas)

                last_iter = (
                    self.tracker.step == self.num_iters - 1
                    if self.num_iters is not None
                    else False
                )

                if self.tracker.step % self.sample_freq == 0 or last_iter:
                    save_samples(self.state, self.val_idx, self.writer)

                if self.tracker.step % self.valid_freq == 0 or last_iter:
                    validate(self.state, self.val_dataloader, self.accel)
                    checkpoint(self.state, self.save_iters, self.log_path)
                    # Reset validation progress bar, print summary since last validation.
                    self.tracker.done("val", f"Iteration {self.tracker.step}")

                if last_iter:
                    break


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    log_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    lambdas: dict = None,
):
    """
    CLI 入口保持不变，只是内部改成创建 GANTrainer 再调用 run。
    """
    trainer = GANTrainer(
        args=args,
        accel=accel,
        seed=seed,
        log_path=log_path,
        num_iters=num_iters,
        save_iters=save_iters,
        sample_freq=sample_freq,
        valid_freq=valid_freq,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        val_idx=val_idx,
        lambdas=lambdas,
    )
    trainer.run()


@argbind.bind(without_prefix=True)
def main(load_path: str = None, save_path: str = None):
    cli = argbind.parse_args(argv=sys.argv)
    load_path = cli.get("load_path", load_path)
    save_path = cli.get("save_path", save_path)
    cfg = load_config_for_argbind(main_yaml=load_path)
    args = argbind.parse_args(argv=sys.argv)
    args.update(cfg)

    # Set debug mode if LOCAL_RANK is 0
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args=args, accel=accel, log_path=save_path)


if __name__ == "__main__":
    main()
