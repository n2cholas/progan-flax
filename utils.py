import typing as T
from functools import partial, wraps

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, Scalar
from shapecheck import check_shapes

DType = T.Any

# https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain
GAIN: T.Dict[str, T.Callable[..., float]] = {
    'linear': lambda _=None: 1.,
    'conv': lambda _=None: 1.,
    'sigmoid': lambda _=None: 1.,
    'tanh': lambda _=None: 5. / 3.,
    'relu': lambda _=None: np.sqrt(2.),
    'leaky_relu': lambda ns: np.sqrt(2. / (1. + ns**2)),
    'selu': lambda _=None: 3.0 / 3.0,
}


def assert_dtype(f):
    @wraps(f)
    def inner(x, *args, **kwargs):
        out = f(x, *args, **kwargs)
        chex.assert_equal(x.dtype, out.dtype)
        return out
    return inner


@assert_dtype
def lerp(a, b, pct):
    chex.assert_equal_shape([a, b])  # avoid unwanted broadcasting
    chex.assert_rank(pct, {0, a.ndim})  # don't want implicit 1-dimensions to be added.
    return a + (b - a) * pct


def truncated_normal_init(lower, upper):
    def init_fn(key, shape, dtype=jnp.float32):
        return jax.random.truncated_normal(key, lower, upper, shape, dtype)
    return init_fn


class EqualizeLR(nn.Module):
    layer: nn.Module
    gain: float = GAIN['conv']()

    @check_shapes(x='N,H,W,-1')
    @nn.compact
    def __call__(self, x: Array) -> Array:
        assert isinstance(self.layer, (nn.Conv, nn.ConvTranspose))

        kernel = self.param(
            'kernel',
            nn.initializers.normal(stddev=1.0),
            (*self.layer.kernel_size, x.shape[-1], self.layer.features)
        )
        c = self.gain / np.sqrt(np.prod(kernel.shape[:-1]))

        variables = {'params': {'kernel': kernel * c}}
        if self.layer.use_bias:
            variables['params']['bias'] = self.param(
                'bias', nn.initializers.zeros, (self.layer.features,)
            )

        return self.layer.apply(variables, x)


@assert_dtype
@check_shapes(x='N,H,W,C', out_='N,H,W,C')
def pixel_norm(x: Array, eps: float = 1e-8) -> Array:
    y = x.astype(jnp.float32)
    inv_std = jax.lax.rsqrt(y.var(-1, keepdims=True) + eps)
    return (y * inv_std).astype(x.dtype)


@assert_dtype
@check_shapes(x='N,-1,-1,C', out_='N,-1,-1,C')
def upsample(x: Array, factor: int = 2):
    n, w, h, c = x.shape
    return jax.image.resize(x, (n, w * factor, h * factor, c), method='nearest')


@assert_dtype
@check_shapes(x='N,-1,-1,C', out_='N,-1,-1,C')
def downsample(x: Array, factor: int = 2):
    n, w, h, c = x.shape
    assert w % factor == 0 and h % factor == 0, (w, h, factor)
    return nn.avg_pool(x, (factor, factor), strides=(factor, factor))


@assert_dtype
@check_shapes(x='N,-1,-1,-1', out_='N,-1,-1,-1')
def append_minibatch_std(x: Array, group_size: int = 4, eps: float = 1e-8):
    n, h, w, c = x.shape
    gs = min(group_size, n)

    grouped_x = x.reshape((gs, -1, h, w, c)).astype(jnp.float32)
    per_group_std = jnp.sqrt(grouped_x.var(axis=0) + eps)  # (n/gs, w, h, c)
    mean_std = per_group_std.mean((1, 2, 3), keepdims=True)  # (n/gs, 1, 1, 1)
    tiled = jnp.tile(mean_std.astype(x.dtype), (gs, h, w, 1))  # (n, h, w, 1)
    return jnp.concatenate([x, tiled], axis=-1)  # (n, h, w, c+1)


class PGGANBlock(nn.Module):
    features: int
    kernel_size: T.Iterable[int] = (3, 3)
    conv_cls: T.Callable[..., nn.Conv] = nn.Conv
    norm_fn: T.Optional[T.Callable[[Array], Array]] = pixel_norm
    dtype: DType = jnp.float32

    @check_shapes(x='N,H,W,-1')
    @nn.compact
    def __call__(self, x: Array) -> Array:
        chex.assert_equal(x.dtype, self.dtype)
        x = EqualizeLR(self.conv_cls(self.features, self.kernel_size, dtype=self.dtype),
                       gain=GAIN['leaky_relu'](0.2))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        if self.norm_fn:
            x = self.norm_fn(x)
        chex.assert_equal(x.dtype, self.dtype)
        return x


class PGGANGenerator(nn.Module):
    feature_sizes: T.Sequence[int]
    stage: T.Optional[int] = None
    dtype: DType = jnp.float32

    @check_shapes(x='N,1,1,-1', out_='N,W,W,3')
    @nn.compact
    def __call__(
        self,
        x: Array,
        *,
        stage: T.Optional[int] = None,  # compile time value
        alpha: T.Optional[Scalar] = None
    ) -> Array:
        if stage is None and self.stage is None:
            stage = len(self.feature_sizes)
        else:
            stage = nn.module.merge_param('stage', self.stage, stage)
        stage = T.cast(int, stage)

        chex.assert_equal(x.dtype, self.dtype)
        Block = partial(PGGANBlock, dtype=self.dtype)
        Conv = partial(nn.Conv, dtype=self.dtype)
        n, w, sz = x.shape[0], 4, self.feature_sizes[0]

        x = Block(sz, (w, w), name=f'{w}x{w}_block_0',
                  conv_cls=partial(nn.ConvTranspose, padding='VALID'))(x)
        x = Block(sz, name=f'{w}x{w}_block_1')(x)
        chex.assert_shape(x, (n, w, w, sz))

        # Rely on JIT to prune unneeded rgb_out computations during inference.
        # Need all of them so variables are created during initialization
        rgb_out = EqualizeLR(Conv(3, (1, 1)), name=f'{w}x{w}_to_rgb',
                             gain=GAIN['tanh']())(x)

        for sz in self.feature_sizes[1:stage]:
            x = upsample(x, factor=2)
            w *= 2
            x = Block(sz, name=f'{w}x{w}_block_0')(x)
            x = Block(sz, name=f'{w}x{w}_block_1')(x)
            chex.assert_shape(x, (n, w, w, sz))

            skip_rgb_out = rgb_out
            rgb_out = EqualizeLR(Conv(3, (1, 1), dtype=self.dtype),
                                 name=f'{w}x{w}_to_rgb',
                                 gain=GAIN['tanh']())(x)

        if stage > 1 and alpha is not None:
            # Order of upsample and to_rgb is swapped from original paper
            rgb_out = lerp(upsample(skip_rgb_out), rgb_out, alpha)

        rgb_out = jnp.tanh(rgb_out)
        chex.assert_shape(rgb_out, (n, w, w, 3))
        chex.assert_equal(rgb_out.dtype, self.dtype)
        return rgb_out


class PGGANDiscriminator(nn.Module):
    feature_sizes: T.Sequence[int]  # reverse of Generator feature_sizes
    stage: T.Optional[int] = None
    dtype: DType = jnp.float32

    @check_shapes(x='N,W,W,3', out_='N,1')
    @nn.compact
    def __call__(
        self,
        x: Array,
        *,
        stage: T.Optional[int] = None,  # compile time value
        alpha: T.Optional[Scalar] = None
    ) -> Array:
        if stage is None and self.stage is None:
            stage = len(self.feature_sizes)
        else:
            stage = nn.module.merge_param('stage', self.stage, stage)
        stage = T.cast(int, stage)

        chex.assert_equal(x.dtype, self.dtype)
        n, w, *_ = x.shape
        chex.assert_equal(w, 2**(stage + 1))
        Block = partial(PGGANBlock, dtype=self.dtype, norm_fn=None)
        Conv = partial(nn.Conv, dtype=self.dtype)

        # Rely on JIT to prune unneeded rgb_out computations during inference.
        # Need all of them so needed variables are created during initialization
        from_rgbs = [
            EqualizeLR(Conv(sz, (1, 1)), name=f'{2**i}x{2**i}_from_rgb')(x)
            for i, sz in enumerate(self.feature_sizes[::-1], start=2)
        ]
        x = from_rgbs[stage - 1]  # since stage starts at 1

        for i in range(stage - 1, 0, -1):
            x = Block(self.feature_sizes[-i - 1], name=f'{w}x{w}_block_0')(x)
            x = Block(self.feature_sizes[-i], name=f'{w}x{w}_block_1')(x)
            x = downsample(x, factor=2)
            w //= 2

            if i == 0 and stage > 1 and alpha is not None:
                x = lerp(downsample(from_rgbs[stage - 2]), x, alpha)

        sz = x.shape[-1]
        x = append_minibatch_std(x)
        chex.assert_shape(x, (n, w, w, sz + 1))

        sz = self.feature_sizes[-1]
        x = Block(sz, name=f'{w}x{w}_block_0')(x)
        x = Block(sz, (4, 4), name=f'{w}x{w}_block_1',
                  conv_cls=partial(Conv, padding=((0, 0), (0, 0))))(x)
        chex.assert_shape(x, (n, 1, 1, sz))

        x = EqualizeLR(Conv(1, (1, 1)))(x).squeeze((1, 2))  # equivalent to Dense
        chex.assert_equal(x.dtype, self.dtype)
        return x
