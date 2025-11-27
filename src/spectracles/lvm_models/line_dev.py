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
from spectracles.model.spatial import PerIFUAndTile, PerTile

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
    # f_cal_raw: PerIFUAndTile  # Unconstrained flux calibration factor per tile and ifu
    f_cal_raw: PerTile  # Unconstrained flux calibration factor per tile

    # def __init__(self, n_tiles, ifu_values):
    def __init__(self, n_tiles, tile_values):
        # self.f_cal_raw = PerIFUAndTile(n_tiles=n_tiles, n_ifus=3, ifu_values=ifu_values)
        self.f_cal_raw = PerTile(n_tiles=n_tiles, tile_values=tile_values)

    def __call__(self, s: SpatialData) -> Array:
        return l_bounded(self.f_cal_raw(s), lower=0.0) / l_bounded(0, lower=0.0)


class StellarContinuum(SpatialModel):
    offs: PerSpaxel

    def __init__(self, n_spaxels: int, offs: Parameter):
        self.offs = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=offs)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.offs(s)


class LSFScatter(SpatialModel):
    Δσ_lsf: PerSpaxel

    def __init__(self, n_spaxels: int, delta_sigma_lsf: Parameter):
        self.Δσ_lsf = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=delta_sigma_lsf)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.Δσ_lsf(s)


class LineFluxScatter(SpatialModel):
    ΔA: PerSpaxel

    def __init__(self, n_spaxels: int, delta_A: Parameter):
        self.ΔA = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=delta_A)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.ΔA(s)


class VelocityScatter(SpatialModel):
    Δv: PerSpaxel

    def __init__(self, n_spaxels: int, delta_v: Parameter):
        self.Δv = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=delta_v)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.Δv(s)


class SmoothContinuum(SpatialModel):
    # Model components
    A_raw: SpatialModel  # Unconstrained line flux, shared with the emission line
    cont_residual_raw: SpatialModel  # "residual" GP part of the continuum
    eps_1: Parameter  # Continuum level as a fraction of line peak
    eps_2: Parameter  # Continuum level for residual GP (as a fraction of line peak if sharing kernel with A_raw, effectively scales down the variance)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.eps_1.val * self.A(s) + self.eps_2.val * self.cont_residual(s)

    def A(self, s) -> Array:
        return l_bounded(self.A_raw(s), lower=A_LOWER)

    def cont_residual(self, s) -> Array:
        return l_bounded(self.cont_residual_raw(s), lower=A_LOWER)


class SkyBackground(SpatialModel):
    # level: PerIFUAndTile  # Sky background level per tile and IFU
    level: PerTile  # Sky background level per tile

    # def __init__(self, n_tiles, ifu_values):
    def __init__(self, n_tiles, tile_values):
        # self.level = PerIFUAndTile(n_tiles=n_tiles, n_ifus=3, ifu_values=ifu_values)
        self.level = PerTile(n_tiles=n_tiles, tile_values=tile_values)

    def __call__(self, s: SpatialDataLVM) -> Array:
        return self.level(s)


class EmissionLine(SpectralSpatialModel):
    # Line centre in Angstroms
    μ: Parameter
    # Model components / line quantities
    A_raw: SpatialModel  # Unconstrained line flux
    v: SpatialModel  # Radial velocity in rest frame in km/s
    vσ_raw: SpatialModel  # Broadening velocity in km/s before constraint
    vσ_raw_mean: Parameter  # Mean of vσ_raw GP
    # Scatter in line quantities
    ΔA: SpatialModel  # Line flux scatter
    Δv: SpatialModel  # Velocity scatter
    # Measured line quantities
    σ_lsf: SpatialModel  # LSF width (std dev) in Angstroms
    v_bary: SpatialModel  # Barycentric velocity CORRECTION in km/s
    # Systematics
    v_syst: Parameter  # Systematic velocity offset in km/s
    v_cal: WaveCalVelocity  # Per-IFU Velocity calibration offset in km/s
    Δσ_lsf: SpatialModel  # LSF scatter in Angstroms

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> Array:
        μ_obs = self.μ_obs(spatial_data)
        σ2_obs = self.σ2_obs(spatial_data, μ_obs)
        peak = self.A_total(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs)
        return peak * jnp.exp(-0.5 * (λ - μ_obs) ** 2 / σ2_obs)

    def A(self, s) -> Array:
        return l_bounded(self.A_raw(s), lower=A_LOWER)

    def A_total(self, s) -> Array:
        return self.A(s) + self.ΔA(s)

    def vσ(self, s) -> Array:
        return l_bounded(self.vσ_raw(s) + self.vσ_raw_mean.val, lower=0.0)

    def v_total(self, s) -> Array:
        return self.v(s) + self.Δv(s)

    def v_obs(self, s) -> Array:
        # return self.v(s) + self.v_syst.val + self.v_cal(s) - self.v_bary(s)  # + self.Δv(s)
        return self.v_total(s) + self.v_syst.val + self.v_cal(s) - self.v_bary(s)

    def μ_obs(self, s) -> Array:
        return self.μ.val * (1 + self.v_obs(s) / C_KMS)

    def σ2_obs(self, s, μ_obs) -> Array:
        return (self.vσ(s) * μ_obs / C_KMS) ** 2 + (self.σ_lsf(s) + self.Δσ_lsf(s)) ** 2

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
        eps_1: Parameter,  # Continuum level as a fraction of line peak
        eps_2: Parameter,  # Continuum residual GP level as a fraction of line peak level
        sky_level: Parameter,
        Δσ_lsf: Parameter,
        ΔA: Parameter,
        Δv: Parameter,
        vσ_raw_mean: Parameter,
    ):
        # Latent GPs for line properties
        A_raw_ = FourierGP(n_modes=n_modes, kernel=A_kernel)
        v_ = FourierGP(n_modes=n_modes, kernel=v_kernel)
        vσ_raw_ = FourierGP(n_modes=n_modes, kernel=σ_kernel)

        # Scatter in line properties (not currently used)
        ΔA_ = LineFluxScatter(n_spaxels=n_spaxels, delta_A=ΔA)
        Δv_ = VelocityScatter(n_spaxels=n_spaxels, delta_v=Δv)

        # Latent GP for residual continuum component (shares kernel with A_raw)
        cont_residual_raw_ = FourierGP(n_modes=n_modes, kernel=A_kernel)

        # Per-spaxel stuff that is measured, not fit for
        σ_lsf_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf)
        v_bary_ = PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary)

        # Calibration / nuisance components for line
        v_cal_ = WaveCalVelocity(C_v_cal=C_v_cal, μ=line_centre)
        Δσ_lsf_ = LSFScatter(n_spaxels=n_spaxels, delta_sigma_lsf=Δσ_lsf)

        self.stars = StellarContinuum(n_spaxels=n_spaxels, offs=offsets)
        self.line = EmissionLine(
            μ=line_centre,
            A_raw=A_raw_,
            v=v_,
            ΔA=ΔA_,
            Δv=Δv_,
            vσ_raw=vσ_raw_,
            vσ_raw_mean=vσ_raw_mean,
            σ_lsf=σ_lsf_,
            v_bary=v_bary_,
            v_syst=v_syst,
            v_cal=v_cal_,
            Δσ_lsf=Δσ_lsf_,
        )
        self.cont = SmoothContinuum(
            A_raw=A_raw_,
            cont_residual_raw=cont_residual_raw_,
            eps_1=eps_1,
            eps_2=eps_2,
        )
        # self.flux_cal = FluxCalFactor(n_tiles=n_tiles, ifu_values=f_cal_unconstrained)
        # self.sky = SkyBackground(n_tiles=n_tiles, ifu_values=sky_level)
        self.flux_cal = FluxCalFactor(n_tiles=n_tiles, tile_values=f_cal_unconstrained)
        self.sky = SkyBackground(n_tiles=n_tiles, tile_values=sky_level)

    def __call__(self, λ, spatial_data):
        return self.flux_cal(spatial_data) * (
            self.cont(spatial_data) + self.stars(spatial_data) + self.line(λ, spatial_data)
        ) + self.sky(spatial_data)


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
