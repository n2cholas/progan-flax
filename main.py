import typing as T
from functools import partial

import chex
import flax
import jax
import jax.numpy as jnp
import jax_utils as ju
import optax
import tensorflow as tf
from flax.metrics import tensorboard

from utils import PGGANDiscriminator, PGGANGenerator, downsample, lerp, upsample


class cfg:
    name = 'celeb_v16'
    distributed = True
    report_freq = 1000
    val_freq = None
    dtype = jnp.bfloat16

    batch_sizes = [512, 256, 128, 64, 32, 32]
    n_steps = [20_000, 40_000, 80_000, 160_000, 320_000, 640_000]
    feat_sizes = (512, 512, 256, 128, 64, 32)  # (128, 128) image
    transition_steps: T.Sequence[int] = []  # will be populated

    lr = 0.0015
    lamb = 10
    eps_drift = 0.001
    noise_size = 512
    ma_beta = 0.9  # original paper used 0.999

    data_dir = '/home/nvadivelu/img_align_celeba/img_align_celeba/'


cfg.transition_steps = [int(n * 0.8) for n in cfg.n_steps]
assert (len(cfg.batch_sizes) == len(cfg.n_steps) == len(cfg.feat_sizes)
        == len(cfg.transition_steps))
generator = PGGANGenerator(cfg.feat_sizes, dtype=cfg.dtype)
discriminator = PGGANDiscriminator(cfg.feat_sizes[::-1], dtype=cfg.dtype)
optim = optax.adam(cfg.lr, b1=0.0, b2=0.99)

def get_dataset(image_size, batch_size):
    def decode_fn(s):
        img = tf.io.decode_jpeg(tf.io.read_file(s))
        img.set_shape([218, 178, 3])
        img = tf.cast(img[20:-20], tf.float32) / 127.5 - 1.0
        img = tf.image.resize(img, (image_size, image_size), antialias=True)
        # img = tf.clip_by_value(img, -1.0, 1.0)
        return tf.cast(img, cfg.dtype)

    ds = (tf.data.Dataset.list_files(cfg.data_dir+'.jpg', shuffle=False)
          .map(decode_fn).cache()
          .map(tf.image.random_flip_left_right)
          .shuffle(batch_size*16)
          .repeat())

    if cfg.distributed:
        per_core_bs, remainder = divmod(batch_size, len(jax.devices()))
        assert remainder == 0
        ds = (ds
              .batch(per_core_bs, drop_remainder=True)
              .batch(len(jax.devices()), drop_remainder=True))
    else:
        ds = ds.batch(batch_size, drop_remainder=True)

    ds = map(lambda x: x._numpy(), ds.prefetch(tf.data.AUTOTUNE))
    return flax.jax_utils.prefetch_to_device(ds, 3) if cfg.distributed else ds

class TrainState(flax.struct.PyTreeNode):
    g_params: ju.Variables
    d_params: ju.Variables
    g_opt_state: optax.OptState
    d_opt_state: optax.OptState
    metrics: ju.Metrics
    rngs: T.Optional[ju.PRNGDict]
    gen_imgs: jnp.ndarray
    data_imgs: jnp.ndarray
    step: chex.Scalar

def get_train_step(stage, transition_steps, transition_begin=cfg.report_freq):
    transition_sched = optax.linear_schedule(
        0.0, 1.0, transition_steps, transition_begin=transition_begin)

    def train_step(state, batch):
        print(f'Compiling batch shape: {batch.shape}')
        chex.assert_equal(batch.dtype, cfg.dtype)
        noise = jax.random.normal(state.rngs['generator'],
                                  shape=(len(batch), 1, 1, cfg.noise_size),
                                  dtype=cfg.dtype)
        epsilon = jax.random.uniform(state.rngs['epsilon'],
                                     (len(batch), 1, 1, 1),
                                     dtype=cfg.dtype)
        alpha = transition_sched(state.step).astype(cfg.dtype)
        kws = {'alpha': alpha, 'stage': stage}

        # Cross fade batch
        if stage > 1:
            batch = lerp(upsample(downsample(batch)), batch, alpha)

        def g_loss_fn(g_params):
            gen_imgs = generator.apply({'params': g_params}, noise, **kws)
            loss = -discriminator.apply({'params': state.d_params}, gen_imgs, **kws)
            return loss.mean(), gen_imgs

        (g_loss, gen_imgs), g_grads = jax.value_and_grad(
            g_loss_fn, has_aux=True)(state.g_params)
        chex.assert_equal(g_loss.dtype, cfg.dtype)
        chex.assert_equal(gen_imgs.dtype, cfg.dtype)

        def d_loss_fn(d_params):
            fake_preds = discriminator.apply(
                {'params': d_params}, gen_imgs, **kws).squeeze(-1)
            real_preds = discriminator.apply(
                {'params': d_params}, batch, **kws).squeeze(-1)

            @jax.grad
            def input_grad_fn(x):
                return discriminator.apply({'params': d_params}, x, **kws).sum()

            x_hat = lerp(gen_imgs, batch, epsilon)
            slopes = jnp.sqrt((input_grad_fn(x_hat)**2).sum(axis=(1,2,3)))

            chex.assert_shape([slopes, fake_preds, real_preds], (len(batch),))
            return jnp.mean(fake_preds - real_preds  # wasserstein loss
                            + cfg.lamb * (slopes - 1)**2  # gradient penalty
                            + cfg.eps_drift * (real_preds ** 2))  # drift penalty

        d_loss, d_grads = jax.value_and_grad(d_loss_fn)(state.d_params)
        chex.assert_equal(d_loss.dtype, cfg.dtype)

        if cfg.distributed:
            (d_grads, g_grads) = jax.lax.pmean((d_grads, g_grads), axis_name='batch')

        g_updates, g_opt_state = optim.update(g_grads, state.g_opt_state, state.g_params)
        d_updates, d_opt_state = optim.update(d_grads, state.d_opt_state, state.d_params)

        ma_g_params = jax.tree_multimap(partial(lerp, pct=cfg.ma_beta),
                                        optax.apply_updates(state.g_params, g_updates),
                                        state.g_params)

        return state.replace(
            g_params=ma_g_params,
            d_params=optax.apply_updates(state.d_params, d_updates),
            g_opt_state=g_opt_state,
            d_opt_state=d_opt_state,
            metrics=state.metrics.update(g_loss=g_loss, d_loss=d_loss, **kws),
            rngs=jax.tree_map(partial(jax.random.fold_in, data=0), state.rngs),
            gen_imgs=gen_imgs,
            data_imgs=batch,
            step=state.step+1
        )
    return train_step


def main():
    print(jax.devices())
    jit_fn = partial(jax.pmap, axis_name='batch') if cfg.distributed else jax.jit

    rngs = ju.PRNGSeq(0)
    image_size = 2**(1+len(cfg.feat_sizes))
    assert image_size == 128

    fake_noise = jax.random.normal(next(rngs), (4, 1, 1, cfg.noise_size), dtype=cfg.dtype)
    fake_image = jax.random.normal(next(rngs), (4, image_size, image_size, 3), dtype=cfg.dtype)

    g_params = generator.init(next(rngs), fake_noise)['params'].unfreeze()
    d_params = discriminator.init(next(rngs), fake_image)['params'].unfreeze()

    state = TrainState(
        g_params=g_params,
        d_params=d_params,
        g_opt_state=optim.init(g_params),
        d_opt_state=optim.init(d_params),
        metrics=ju.Metrics.from_names('g_loss', 'd_loss', 'alpha', 'stage'),
        rngs={'generator': next(rngs), 'epsilon': next(rngs)},
        gen_imgs=jnp.ones((cfg.batch_sizes[0], image_size, image_size, 3), dtype=cfg.dtype),
        data_imgs=jnp.ones((cfg.batch_sizes[0], image_size, image_size, 3), dtype=cfg.dtype),
        step=jnp.zeros([], dtype=jnp.int32)
    )

    # Test
    ds = get_dataset(image_size=16, batch_size=512)
    train_step = jit_fn(get_train_step(3, 4))

    if cfg.distributed:
        rep_state = flax.jax_utils.replicate(state)
        _ = train_step(rep_state, next(ds))
        del rep_state
    else:
        _ = train_step(state, next(ds))

    # Train
    tb = tensorboard.SummaryWriter(log_dir=f'./pggan-celeb/{cfg.name}_tb')

    reporter = ju.Reporter(
        train_names=list(state.metrics.names()) + ['time/step'],
        val_names=[],
        summary_writer=tb,
        write_csv=True,
    )

    def report_fn(state, step):
        tb.image('gen_imgs', (state.gen_imgs + 1) / 2.0, step, max_outputs=12)
        tb.image('data_imgs', (state.data_imgs + 1) / 2.0, step, max_outputs=12)

        start_step, n_steps = 0, 0

        for stage, (t_steps, steps, bs) in enumerate(zip(
            cfg.transition_steps, cfg.n_steps, cfg.batch_sizes), start=1):

            start_step = n_steps
            n_steps += steps
            img_sz = 2**(stage + 1)

            ds = get_dataset(image_size=img_sz, batch_size=bs)
            train_step = jit_fn(get_train_step(stage, t_steps))
            state = state.replace(gen_imgs=jnp.ones((bs, img_sz, img_sz, 3), dtype=cfg.dtype),
                                  data_imgs=jnp.ones((bs, img_sz, img_sz, 3), dtype=cfg.dtype),
                                  step=0)

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
            flax.training.checkpoints.save_checkpoint(f'./pggan-celeb/{cfg.name}',
                                                      jax.device_get(state),
                                                      step=n_steps,
                                                      keep=20)

        tb.close()
        # new_state = flax.training.checkpoints.restore_checkpoint(
        #       f'./pggan-celeb/{cfg.name}', target=state)


if __name__ == '__main__':
    main()
