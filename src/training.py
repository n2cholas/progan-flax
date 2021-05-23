import typing as T
from functools import partial

import chex
import flax
import jax
import jax.numpy as jnp
import jax_utils as ju
import optax

from model import downsample, upsample


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
    def create(cls, *, generator, discriminator, g_optim, d_optim,
               metrics, noise_size, seed, dtype, **_):
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
            g_opt_state=g_optim.init(g_params),
            d_opt_state=d_optim.init(d_params),
            metrics=metrics,
            rngs={'generator': next(rngs), 'epsilon': next(rngs)},
            gen_imgs=fake_images,
            gen_ema_imgs=fake_images,
            data_imgs=fake_images,
            step=jnp.zeros([], dtype=jnp.int32),
        )


def get_train_step(*, generator, discriminator, g_optim, d_optim, stage, alpha_sched,
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
            batch = ju.lerp(upsample(downsample(batch)), batch, alpha)

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
            x_hat = ju.lerp(gen_imgs, batch, epsilon)
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

        g_updates, g_opt_state = g_optim.update(g_grads, state.g_opt_state, state.g_params)
        d_updates, d_opt_state = d_optim.update(d_grads, state.d_opt_state, state.d_params)

        g_params = optax.apply_updates(state.g_params, g_updates)
        g_ema_params = jax.tree_multimap(partial(ju.lerp, pct=ma_beta),
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
