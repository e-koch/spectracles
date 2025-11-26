"""line_dev.py - Developmental spectrospatial model of a single emission line for LVM."""

import jax
import jax.numpy as jnp
from jaxtyping import Array

from spectracles import (
    FourierGP,
    Kernel,
    Parameter,
    PerSpaxel,
    SpatialDataLVM,
    SpatialModel,
    SpectralSpatialModel,
    l_bounded,
)
from spectracles.lvm_models.constants import C_KMS
from spectracles.lvm_models.likelihood import ln_likelihood
from spectracles.model.data import SpatialData
from spectracles.model.spatial import PerIFUAndTile

A_LOWER = 0.0


class WaveCalVelocity(SpatialModel):
    # Model parameters
    C_v_cal: Parameter  # 2 value model parameter
    # Constants
    μ: Parameter  # line centre in Angstroms

    def __call__(self, s: SpatialData) -> Array:
        v0 = 0.0  # effectively pinned to v_syst
        v1 = v0 - C_KMS * (self.C_v_cal.val[0] / self.μ.val[0])
        v2 = v1 - C_KMS * (self.C_v_cal.val[1] / self.μ.val[0])
        v_ifu_vals = jnp.array([v0, v1, v2])
        return v_ifu_vals[s.ifu_idx]


class FluxCalFactor(SpatialModel):
    f_cal_raw: PerIFUAndTile  # Unconstrained flux calibration factor per tile and ifu

    def __init__(self, n_tiles, ifu_values):
        self.f_cal_raw = PerIFUAndTile(n_tiles=n_tiles, n_ifus=3, ifu_values=ifu_values)

    def __call__(self, s: SpatialData) -> Array:
        return l_bounded(self.f_cal_raw(s), lower=0.0) / l_bounded(0, lower=0.0)


class StellarContinuum(SpectralSpatialModel):
    offs: PerSpaxel

    def __init__(self, n_spaxels: int, offs: Parameter):
        self.offs = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=offs)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.offs(s)


class SmoothContinuum(SpectralSpatialModel):
    # Model components
    A_raw: SpatialModel  # Unconstrained line flux, shared with the emission line
    eps: Parameter  # Continuum level

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.eps.val * self.A(s)

    def A(self, s) -> Array:
        return l_bounded(self.A_raw(s), lower=A_LOWER)


class SkyBackground(SpectralSpatialModel):
    level: PerIFUAndTile  # Sky background level per tile and IFU

    def __init__(self, n_tiles, ifu_values):
        self.level = PerIFUAndTile(n_tiles=n_tiles, n_ifus=3, ifu_values=ifu_values)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.level(s)


class EmissionLine(SpectralSpatialModel):
    # Line centre in Angstroms
    μ: Parameter
    # Model components / line quantities
    A_raw: SpatialModel  # Unconstrained line flux
    v: SpatialModel  # Radial velocity in rest frame in km/s
    vσ_raw: SpatialModel  # Broadening velocity in km/s before constraint
    # Measured line quantities
    σ_lsf: SpatialModel  # LSF width (std dev) in Angstroms
    v_bary: SpatialModel  # Barycentric velocity CORRECTION in km/s
    # Systematics
    v_syst: Parameter  # Systematic velocity offset in km/s
    v_cal: WaveCalVelocity  # Per-IFU Velocity calibration offset in km/s

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> Array:
        μ_obs = self.μ_obs(spatial_data)
        σ2_obs = self.σ2_obs(spatial_data, μ_obs)
        peak = self.A(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs)
        return peak * jnp.exp(-0.5 * (λ - μ_obs) ** 2 / σ2_obs)

    def A(self, s) -> Array:
        return l_bounded(self.A_raw(s), lower=A_LOWER)

    def vσ(self, s) -> Array:
        return l_bounded(self.vσ_raw(s), lower=0.0)

    def v_obs(self, s) -> Array:
        return self.v(s) + self.v_syst.val + self.v_cal(s) - self.v_bary(s)

    def μ_obs(self, s) -> Array:
        return self.μ.val * (1 + self.v_obs(s) / C_KMS)

    def σ2_obs(self, s, μ_obs) -> Array:
        return (self.vσ(s) * μ_obs / C_KMS) ** 2 + self.σ_lsf(s) ** 2

    def f_cal(self, s) -> Array:
        return l_bounded(self.f_cal_raw(s), lower=0.0) / l_bounded(0, lower=0.0)


class LVMModelSingle(SpectralSpatialModel):
    # Model components
    line: EmissionLine  # Emission line model
    stars: StellarContinuum  # Nuisance offsets per spaxel
    cont: SmoothContinuum  # Smooth continuum model

    # Calibration/Nuisances
    flux_cal: FluxCalFactor  # Flux calibration factor per tile
    sky: SkyBackground  # Sky background per tile and IFU

    def __init__(
        self,
        n_tiles: int,
        n_spaxels: int,
        offsets: Parameter,
        line_centre: Parameter,
        n_modes: tuple[int, int],
        A_kernel: Kernel,
        v_kernel: Kernel,
        σ_kernel: Kernel,
        σ_lsf: Parameter,
        v_bary: Parameter,
        v_syst: Parameter,
        C_v_cal: Parameter,  # MUST be 2 values i.e. shape is (2,)
        f_cal_unconstrained: Parameter,
        eps: Parameter,  # Continuum level as a fraction of line peak
        sky_level: Parameter,
    ):
        # Latent GPs for line properties
        A_raw_ = FourierGP(n_modes=n_modes, kernel=A_kernel)
        v_ = FourierGP(n_modes=n_modes, kernel=v_kernel)
        vσ_raw_ = FourierGP(n_modes=n_modes, kernel=σ_kernel)

        # Per-spaxel stuff that is measured, not fit for
        σ_lsf_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf)
        v_bary_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary)

        # Calibration / nuisance components for line
        v_cal_ = WaveCalVelocity(C_v_cal=C_v_cal, μ=line_centre)

        self.stars = StellarContinuum(n_spaxels=n_spaxels, offs=offsets)
        self.line = EmissionLine(
            μ=line_centre,
            A_raw=A_raw_,
            v=v_,
            vσ_raw=vσ_raw_,
            σ_lsf=σ_lsf_,
            v_bary=v_bary_,
            v_syst=v_syst,
            v_cal=v_cal_,
        )
        self.cont = SmoothContinuum(A_raw=A_raw_, eps=eps)
        self.flux_cal = FluxCalFactor(n_tiles=n_tiles, ifu_values=f_cal_unconstrained)
        self.sky = SkyBackground(n_tiles=n_tiles, ifu_values=sky_level)

    def __call__(self, λ, spatial_data):
        return self.flux_cal(spatial_data) * (
            self.cont(spatial_data) + self.stars(spatial_data) + self.line(λ, spatial_data)
        ) + self.sky(spatial_data)

    # def __call__(self, λ, spatial_data):
    #     return self.flux_cal(spatial_data) * (
    #         self.stars(spatial_data) + self.line(λ, spatial_data)
    #     ) + self.sky(spatial_data)


def neg_ln_posterior(model, λ, xy_data, data, u_data, mask):
    vmapped_model = jax.vmap(model, in_axes=(0, None))
    ln_like = ln_likelihood(vmapped_model, λ, xy_data, data, u_data, mask)
    print(ln_like)
    ln_prior = (
        model.line.A_raw.prior_logpdf()
        + model.line.v.prior_logpdf()
        + model.line.vσ_raw.prior_logpdf()
    )
    print(ln_prior)
    return -1 * (ln_like + ln_prior)
