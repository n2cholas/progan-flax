from functools import partial

import flax
import hydra
import jax
import jax.numpy as jnp
import jax_utils as ju
import optax
import tensorflow as tf
from flax.metrics import tensorboard

from model import PGGANDiscriminator, PGGANGenerator
from training import TrainState, get_train_step


def get_dataset(*, batch_size, image_size, dtype, data_dir, distributed,
                dummy_data=False, **_):
    def decode_fn(s):
        img = tf.io.decode_jpeg(tf.io.read_file(s))
        img.set_shape([218, 178, 3])
        img = tf.cast(img[20:-20], tf.float32) / 127.5 - 1.0
        img = tf.image.resize(img, (image_size, image_size), antialias=True)
        # img = tf.clip_by_value(img, -1.0, 1.0)
        return tf.cast(img, dtype)

    if not dummy_data:
        ds = (tf.data.Dataset.list_files(data_dir+'*.jpg', shuffle=False)
              .map(decode_fn).cache()
              .map(tf.image.random_flip_left_right)
              .shuffle(batch_size*16)
              .repeat())
    else:
        dummy = tf.random.normal((image_size, image_size, 3), dtype=dtype)
        ds = tf.data.Dataset.from_tensors(dummy).repeat()

    if distributed:
        per_core_bs, remainder = divmod(batch_size, len(jax.devices()))
        assert remainder == 0
        ds = (ds
              .batch(per_core_bs, drop_remainder=True)
              .batch(len(jax.devices()), drop_remainder=True))
    else:
        ds = ds.batch(batch_size, drop_remainder=True)

    ds = map(lambda x: x._numpy(), ds.prefetch(tf.data.AUTOTUNE))
    return flax.jax_utils.prefetch_to_device(ds, 3) if distributed else ds


@partial(jax.jit, static_argnums=3)
def prep_images_for_tb(gen_imgs, gen_ema_imgs, data_imgs, max_outputs):
    # Scales images from [-1, 1] to [0, 1]
    print('Compiling prep_images_for_tb...')
    return (((gen_imgs + 1) / 2.0)[:max_outputs],
            ((gen_ema_imgs + 1) / 2.0)[:max_outputs],
            ((data_imgs + 1) / 2.0)[:max_outputs])


@hydra.main(config_name='config', config_path='conf')
def main(cfg):
    per_stage_items = (
        cfg.batch_sizes, cfg.n_steps, cfg.feat_sizes, cfg.transition_pcts)
    assert len(set(map(len, per_stage_items))) == 1
    assert 2**(1+len(cfg.feat_sizes)) == cfg.final_image_size
    # convert dtype_str explicitly due to https://github.com/google/jax/issues/6813
    dtype = jnp.dtype(cfg.dtype_str)

    # Initialize TrainState
    train_objs = {
        'generator': PGGANGenerator(cfg.feat_sizes, dtype=dtype),
        'discriminator': PGGANDiscriminator(cfg.feat_sizes[::-1], dtype=dtype),
        'g_optim': optax.adam(cfg.g_lr, b1=0.0, b2=0.99),
        'd_optim': optax.adam(cfg.g_lr, b1=0.0, b2=0.99)
    }
    # Metrics will keep a running average of the listed scalar quantities
    metrics = ju.Metrics.from_names(
        'g_loss', 'd_loss', 'w_dist', 'gp', 'drift', 'alpha', 'stage')
    state = TrainState.create(**train_objs, metrics=metrics, dtype=dtype, **cfg)

    tb = tensorboard.SummaryWriter(log_dir=f'./{cfg.name}_tb')
    # Reporter will log metrics and other values to TensorBoard and CSVs
    reporter = ju.Reporter(
        train_names=list(state.metrics.names()) + ['time/step'],
        val_names=[],
        summary_writer=tb,
        write_csv=True)

    def report_fn(state, _, step):
        gen_imgs, gen_ema_imgs, data_imgs = prep_images_for_tb(
            state.gen_imgs, state.gen_ema_imgs, state.data_imgs, cfg.max_tb_images)
        tb.image('gen_imgs', gen_imgs, step, max_outputs=cfg.max_tb_images)
        tb.image('gen_ema_imgs', gen_ema_imgs, step, max_outputs=cfg.max_tb_images)
        tb.image('data_imgs', data_imgs, step, max_outputs=cfg.max_tb_images)

    # Outer training loop cycles through each stage
    start_step, n_steps = 0, 0
    for stage, (t_pct, steps, bs) in enumerate(zip(
        cfg.transition_pcts, cfg.n_steps, cfg.batch_sizes), start=1):

        start_step = n_steps
        n_steps += steps
        img_sz = 2**(stage + 1)

        # Prepare dataset, train step, and state for current stage.
        ds = get_dataset(batch_size=bs, image_size=img_sz, dtype=dtype, **cfg)
        alpha_sched = optax.linear_schedule(
            0.0, 1.0, int(steps*t_pct), cfg.transition_delay)
        train_step = get_train_step(
            stage=stage, alpha_sched=alpha_sched, dtype=dtype, **train_objs, **cfg)

        dummy_imgs = jnp.zeros((bs, img_sz, img_sz, 3), dtype=dtype)
        if cfg.distributed:
            dummy_imgs = dummy_imgs[:(bs//len(jax.devices()))]
        state = state.replace(gen_imgs=dummy_imgs,
                              gen_ema_imgs=dummy_imgs,
                              data_imgs=dummy_imgs,
                              g_opt_state=train_objs['g_optim'].init(state.g_params),
                              d_opt_state=train_objs['d_optim'].init(state.d_params),
                              step=0)

        # Train for n_steps iterations
        state = ju.train(
            state=state,
            train_iter=ds,
            train_step=train_step,
            n_steps=n_steps+1,
            start_step=start_step+1,
            report_freq=cfg.report_freq,
            reporter=reporter,
            save_ckpts=False,
            distributed=cfg.distributed,
            extra_report_fn=report_fn,
        )

        cpu_state = jax.device_get(state)
        # Save state and generator checkpoints seperately for easy use later.
        flax.training.checkpoints.save_checkpoint(
            f'./{cfg.name}', cpu_state, step=n_steps, keep=20)
        flax.training.checkpoints.save_checkpoint(
            f'./{cfg.name}-gen', cpu_state.g_params, step=n_steps, keep=20)

    tb.close()


if __name__ == '__main__':
    main()
