import typing as T
from functools import partial

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import jax_utils as ju
import optax
import tensorflow as tf
from flax.metrics import tensorboard

from model import PGGANDiscriminator, PGGANGenerator, downsample, lerp, upsample


def get_dataset(*, batch_size, image_size, dtype, data_dir, distributed, **_):
    def decode_fn(s):
        img = tf.io.decode_jpeg(tf.io.read_file(s))
        img.set_shape([218, 178, 3])
        img = tf.cast(img[20:-20], tf.float32) / 127.5 - 1.0
        img = tf.image.resize(img, (image_size, image_size), antialias=True)
        # img = tf.clip_by_value(img, -1.0, 1.0)
        return tf.cast(img, dtype)

    ds = (tf.data.Dataset.list_files(data_dir+'*.jpg', shuffle=False)
          .map(decode_fn).cache()
          .map(tf.image.random_flip_left_right)
          .shuffle(batch_size*16)
          .repeat())

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

class TrainState(flax.struct.PyTreeNode):
    g_params: ju.Variables
    g_ema_params: ju.Variables
    d_params: ju.Variables
    g_opt_state: optax.OptState
    d_opt_state: optax.OptState
    metrics: ju.Metrics
    rngs: T.Optional[ju.PRNGDict]
    data_imgs: jnp.ndarray
    gen_ema_imgs: jnp.ndarray
    gen_imgs: jnp.ndarray
    step: chex.Scalar

    @classmethod
    def create(cls, *, generator, discriminator, optim, metrics, noise_size, seed, dtype, **_):
        rngs = ju.PRNGSeq(seed)
        assert generator.feature_sizes == discriminator.feature_sizes[::-1]
        img_sz = 2**(1+len(generator.feature_sizes))

        noise = jax.random.normal(next(rngs), (4, 1, 1, noise_size), dtype=dtype)
        fake_images = jax.random.normal(next(rngs), (4, img_sz, img_sz, 3), dtype=dtype)
        g_params = jax.jit(generator.init)(next(rngs), noise)['params'].unfreeze()
        d_params = jax.jit(discriminator.init)(next(rngs), fake_images)['params'].unfreeze()

        return cls(
            g_params=g_params,
            g_ema_params=g_params,
            d_params=d_params,
            g_opt_state=optim.init(g_params),
            d_opt_state=optim.init(d_params),
            metrics=metrics,
            rngs={'generator': next(rngs), 'epsilon': next(rngs)},
            gen_imgs=fake_images,
            gen_ema_imgs=fake_images,
            data_imgs=fake_images,
            step=jnp.zeros([], dtype=jnp.int32),
        )


def get_train_step(*, generator, discriminator, optim, stage, alpha_sched,
                   noise_size, eps_drift, lamb, ma_beta, dtype, report_freq, distributed, **_):
    def train_step(state, batch):
        print(f'Compiling batch shape: {batch.shape}')
        chex.assert_equal(batch.dtype, dtype)
        noise = jax.random.normal(state.rngs['generator'],
                                  shape=(len(batch), 1, 1, noise_size),
                                  dtype=dtype)
        alpha = alpha_sched(state.step).astype(dtype)
        kws = {'alpha': alpha, 'stage': stage}

        if stage > 1:  # Cross fade batch
            batch = lerp(upsample(downsample(batch)), batch, alpha)

        def g_loss_fn(g_params):
            gen_imgs = generator.apply({'params': g_params}, noise, **kws)
            loss = -discriminator.apply({'params': state.d_params}, gen_imgs, **kws)
            return loss.mean(), gen_imgs

        (g_loss, gen_imgs), g_grads = jax.value_and_grad(
            g_loss_fn, has_aux=True)(state.g_params)
        chex.assert_equal(g_loss.dtype, dtype)
        chex.assert_equal(gen_imgs.dtype, dtype)

        def d_loss_fn(d_params):
            fake_preds = discriminator.apply(
                {'params': d_params}, gen_imgs, **kws).squeeze(-1)
            real_preds = discriminator.apply(
                {'params': d_params}, batch, **kws).squeeze(-1)

            @jax.grad
            def input_grad_fn(x):
                return discriminator.apply({'params': d_params}, x, **kws).sum()

            epsilon = jax.random.uniform(state.rngs['epsilon'],
                                         (len(batch), 1, 1, 1),
                                         dtype=dtype)
            x_hat = lerp(gen_imgs, batch, epsilon)
            slopes = jnp.sqrt((input_grad_fn(x_hat)**2).sum(axis=(1,2,3)))

            chex.assert_shape([slopes, fake_preds, real_preds], (len(batch),))
            w_dist = jnp.mean(fake_preds - real_preds)
            gp = jnp.mean((slopes - 1)**2)  # gradient penalty
            drift = jnp.mean(real_preds ** 2)  # drift penalty
            d_metrics = {'w_dist': w_dist, 'gp': gp, 'drift': drift}
            return w_dist + lamb * gp + eps_drift * drift, d_metrics

        (d_loss, d_metrics), d_grads = jax.value_and_grad(
            d_loss_fn, has_aux=True)(state.d_params)
        chex.assert_equal(d_loss.dtype, dtype)

        if distributed:
            (d_grads, g_grads) = jax.lax.pmean((d_grads, g_grads), axis_name='batch')

        g_updates, g_opt_state = optim.update(g_grads, state.g_opt_state, state.g_params)
        d_updates, d_opt_state = optim.update(d_grads, state.d_opt_state, state.d_params)

        g_params = optax.apply_updates(state.g_params, g_updates)
        g_ema_params = jax.tree_multimap(partial(lerp, pct=ma_beta),
                                         g_params, state.g_ema_params)

        gen_ema_imgs = jax.lax.cond(
            (state.step + 1) % report_freq == 0,
            lambda _: generator.apply({'params': g_ema_params}, noise, **kws),
            lambda _: batch,  # dummy value
            operand=None)

        return state.replace(
            g_params=g_params,
            g_ema_params=g_ema_params,
            d_params=optax.apply_updates(state.d_params, d_updates),
            g_opt_state=g_opt_state,
            d_opt_state=d_opt_state,
            metrics=state.metrics.update(g_loss=g_loss, d_loss=d_loss, **d_metrics, **kws),
            rngs=jax.tree_map(partial(jax.random.fold_in, data=0), state.rngs),
            data_imgs=batch,
            gen_imgs=gen_imgs,
            gen_ema_imgs=gen_ema_imgs,
            step=state.step+1,
        )

    if distributed:
        return jax.pmap(train_step, axis_name='batch')
    else:
        return jax.jit(train_step)


@partial(jax.jit, static_argnums=3)
def prep_images_for_tb(gen_imgs, gen_ema_imgs, data_imgs, max_outputs):
    print('Compiling plotting...')
    return (((gen_imgs + 1) / 2.0)[:max_outputs],
            ((gen_ema_imgs + 1) / 2.0)[:max_outputs],
            ((data_imgs + 1) / 2.0)[:max_outputs])


@hydra.main(config_name='config', config_path='conf')
def main(cfg):
    per_stage_items = (
        cfg.batch_sizes, cfg.n_steps, cfg.feat_sizes, cfg.transition_pcts)
    assert len(set(map(len, per_stage_items))) == 1
    assert 2**(1+len(cfg.feat_sizes)) == cfg.final_image_size
    dtype = jnp.dtype(cfg.dtype_str)

    # Initialize TrainState
    train_objs = dict(
        generator = PGGANGenerator(cfg.feat_sizes, dtype=dtype),
        discriminator = PGGANDiscriminator(cfg.feat_sizes[::-1], dtype=dtype),
        optim = optax.adam(cfg.g_lr, b1=0.0, b2=0.99),
    )
    metrics = ju.Metrics.from_names(
        'g_loss', 'd_loss', 'w_dist', 'gp', 'drift', 'alpha', 'stage')
    state = TrainState.create(**train_objs, metrics=metrics, dtype=dtype, **cfg)

    # Train
    tb = tensorboard.SummaryWriter(log_dir=f'./pggan-celeb/{cfg.name}_tb')
    reporter = ju.Reporter(
        train_names=list(state.metrics.names()) + ['time/step'],
        val_names=[],
        summary_writer=tb,
        write_csv=True,
    )

    def report_fn(state, batch, step):
        del batch
        gen_imgs, gen_ema_imgs, data_imgs = prep_images_for_tb(
            state.gen_imgs, state.gen_ema_imgs, state.data_imgs, cfg.max_outputs)
        tb.image('gen_imgs', gen_imgs, step, max_outputs=cfg.max_outputs)
        tb.image('gen_ema_imgs', gen_ema_imgs, step, max_outputs=cfg.max_outputs)
        tb.image('data_imgs', data_imgs, step, max_outputs=cfg.max_outputs)
        tb.scalar('state.step', state.step, step)

    start_step, n_steps = 0, 0
    for stage, (t_pct, steps, bs) in enumerate(zip(
        cfg.transition_pcts, cfg.n_steps, cfg.batch_sizes), start=1):

        start_step = n_steps
        n_steps += steps
        img_sz = 2**(stage + 1)

        ds = get_dataset(batch_size=bs, image_size=img_sz, dtype=dtype, **cfg)
        alpha_sched = optax.linear_schedule(
            0.0, 1.0, steps*t_pct, cfg.transition_delay)
        train_step = get_train_step(
            stage=stage, alpha_sched=alpha_sched, dtype=dtype, **train_objs, **cfg)

        dummy_imgs = jnp.zeros((bs, img_sz, img_sz, 3), dtype=dtype)
        if cfg.distributed:
            dummy_imgs = dummy_imgs[:(bs//len(jax.devices()))]
        state = state.replace(gen_imgs=dummy_imgs,
                              gen_ema_imgs=dummy_imgs,
                              data_imgs=dummy_imgs,
                              g_opt_state=train_objs['optim'].init(state.g_params),
                              d_opt_state=train_objs['optim'].init(state.d_params),
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


if __name__ == '__main__':
    main()
