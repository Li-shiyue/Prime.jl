# ====== use for Clustering stage ======

function his_gf(rdn_val, rdm_val, z)
    z1, z2 = z
    return z1^rdn_val * z2^rdm_val
end

# ====== use for Inference stage ======

function mean_std_raw(col, pos)
    n = 0
    mean = 0.0
    M2 = 0.0
    @inbounds for i in pos
        x = float(col[i])
        n += 1
        δ = x - mean
        mean += δ / n
        M2 += δ * (x - mean)
    end
    if n <= 1
        return mean, 0.0
    else
        var = M2 / (n - 1)
        return mean, sqrt(max(var, 0.0))
    end
end

function inlier_positions_for_cluster_scaled(
    rdm_col, rdn_col,
    base_pos;
    nsigma::Float64 = 5.0
)
    isempty(base_pos) && return Int[]
    μm, σm = mean_std_raw(rdm_col, base_pos)
    μn, σn = mean_std_raw(rdn_col, base_pos)

    σm = (σm == 0.0) ? 1e-12 : σm
    σn = (σn == 0.0) ? 1e-12 : σn

    out = Int[]
    sizehint!(out, length(base_pos))

    @inbounds for i in base_pos
        xm = float(rdm_col[i])
        xn = float(rdn_col[i])
        if abs(xm - μm) ≤ nsigma * σm && abs(xn - μn) ≤ nsigma * σn
            push!(out, i)
        end
    end
    return out
end

function hdi(samples; cred_mass=0.8)
    s = sort(vec(samples))
    N = length(s)
    interval = round(Int, cred_mass * N)
    widths = s[(interval+1):end] .- s[1:(end-interval)]
    idx = argmin(widths)
    return (s[idx], s[idx + interval])
end

function extract_stats(chn; burnin_frac=0.1, thin=1)
    l_samples         = Array(chn[:ℓ])
    sigma_on_samples  = Array(chn[:σon])
    sigma_off_samples = Array(chn[:σoff])
    tau_samples       = Array(chn[:τ])

    bs_raw = l_samples ./ sigma_off_samples
    bf_raw = sigma_on_samples .* tau_samples

    bs_med = median(bs_raw)
    bf_med = median(bf_raw)

    N = length(bs_raw)
    b = clamp(floor(Int, N * burnin_frac), 0, N - 1)
    sel = (b+1):thin:N

    bs_post = bs_raw[sel]
    bf_post = bf_raw[sel]
    τ_post  = tau_samples[sel]

    data = Array{Float64}(undef, N, 3, 1)
    data[:, 1, 1] = bs_raw
    data[:, 2, 1] = bf_raw
    data[:, 3, 1] = tau_samples
    ch = Chains(data, [:bs, :bf, :τ])
    ess_map = ess(ch)

    return (
        bs_med=bs_med, bf_med=bf_med,
        bs_post=bs_post, bf_post=bf_post, τ_post=τ_post,
        bs_ess=DataFrame(ess_map[:bs]).ess[1],
        bf_ess=DataFrame(ess_map[:bf]).ess[1],
        τ_ess=DataFrame(ess_map[:τ]).ess[1],
        rho_med=median(l_samples),
        sigma_on_med=median(sigma_on_samples),
        sigma_off_med=median(sigma_off_samples),
        tau_med=median(tau_samples),
        bs_ci=hdi(bs_raw),
        bf_ci=hdi(bf_raw),
        tau_ci=hdi(tau_samples)
    )
end

function hist_gf(hist_data,z)
    z1,z2 = z
    Nx = size(hist_data,1)
    Ny = size(hist_data,2)
    z1_vec = [z1.^i for i = 0 : Nx-1]
    z2_vec = [z2.^i for i = 0 : Ny-1]
    z_mat = z1_vec*z2_vec'
    return sum(z_mat.*hist_data)
end

function cus_hist2(data1::Vector,data2::Vector)
    data = (data1,data2)
    max1 = maximum(data1)
    max2 = maximum(data2)
    edge1 = collect(0:1:max1+1)
    edge2 = collect(0:1:max2+1)
    h = fit(Histogram, data,(edge1.-0.5,edge2.-0.5))
    Weights = h.weights/length(data1)
    return Weights
end

function beta_kde_volume_factor(β_1;
                                nquad::Int=13, min_n::Int=30)
    β = Float64.(β_1)
    if length(β) < min_n
        Xl = β
        PF = ones(length(β))
        PF ./= sum(PF)
        return Xl, PF
    end
    Density = kde(β)
    minβ, maxβ = minimum(β), maximum(β)
    xl, wl = gausslegendre(nquad)

    Xl = (maxβ - minβ)/2 .* xl .+ (maxβ + minβ)/2
    ik = InterpKDE(Density)
    pf = pdf(ik, Xl)

    PF = pf .* wl .* (maxβ - minβ) / 2
    s = sum(PF)
    PF = (s == 0.0) ? fill(1.0/length(PF), length(PF)) : (PF ./ s)
    return Xl, PF
end

function ep_pgf_from_hist(hist_data, t)
    out = Vector{Float64}(undef, length(t))
    @inbounds for k in eachindex(t)
        out[k] = hist_gf(hist_data, t[k])
    end
    return out
end

function Evec_pgf_from_hist(hist_data, T)
    out = Vector{Float64}(undef, length(T))
    @inbounds for k in eachindex(T)
        out[k] = hist_gf(hist_data, T[k])
    end
    return out
end

function ensure_positive_definite(matrix, ϵ::Float64 = 1e-8)
    A = Symmetric(matrix)
    λmin = minimum(eigvals(A))
    if λmin < 0
        A = A + (abs(λmin) + ϵ) * I
    else
        A = A + ϵ * I
    end
    return Matrix(A)
end

#Default t/T grid
function default_t()
    t0 = collect(0.95:0.04/6:0.99)
    [(t0[i], t0[j]) for i in 1:length(t0), j in 1:length(t0)] |> vec
end

function default_T(t::Vector)
    [t[i] .* t[j] for i in 1:length(t), j in 1:length(t)] |> vec
end
