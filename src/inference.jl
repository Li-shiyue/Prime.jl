# src/inference.jl
import ..model_gf2
import ..mean_std_raw
import ..inlier_positions_for_cluster_scaled
import ..hdi
import ..extract_stats
import ..hist_gf
import ..cus_hist2
import ..beta_kde_volume_factor
import ..ep_pgf_from_hist
import ..Evec_pgf_from_hist
import ..ensure_positive_definite
import ..default_t
import ..default_T

import ..model_gf2
import ..mean_std_raw
import ..inlier_positions_for_cluster_scaled
import ..hdi
import ..extract_stats
import ..hist_gf
import ..cus_hist2
import ..beta_kde_volume_factor
import ..ep_pgf_from_hist
import ..Evec_pgf_from_hist
import ..ensure_positive_definite
import ..default_t
import ..default_T


@model function Turing_model(Xl, PF, z, y, Σ, sel)
    ℓ    ~ InverseGamma(2, 8)
    σon  ~ InverseGamma(2, 0.5)
    σoff ~ InverseGamma(2, 0.5)
    τ    ~ Gamma(2, 0.85)

    θ = [[ℓ*Xl[i]; σon; σoff; τ] for i in eachindex(Xl)]
    U = [(x->model_gf2(θ[i], x, sel)).(z) for i in eachindex(Xl)]
    μ = mean(U, weights(PF))
    y ~ MvNormal(μ, Σ)
end

# --------- Worker task: process one gene across all clusters ---------
function BayInf_gene_pgf!(gi::Int,
                          genes,
                          rdn, rdm, β,
                          clusters,
                          base_positions,
                          t, T,
                          out_ch::RemoteChannel;
                          nsigma::Float64 = 5.0,
                          sel::Int = 4,
                          n_samples::Int = 1000)

    gname = String(genes[gi])
    @views rdn_col = rdn[:, gi]
    @views rdm_col = rdm[:, gi]

    L = length(t)

    for (j, c) in enumerate(clusters)
        pos0 = base_positions[j]
        pos = inlier_positions_for_cluster_scaled(rdm_col, rdn_col, pos0; nsigma=nsigma)
        isempty(pos) && continue

        β_1 = β[pos]
        Xl, PF = beta_kde_volume_factor(β_1; nquad=13, min_n=30)

        data1 = Int.(rdn_col[pos])
        data2 = Int.(rdm_col[pos])
        hist_data = cus_hist2(data1, data2)

        ep   = ep_pgf_from_hist(hist_data, t)
        Evec = Evec_pgf_from_hist(hist_data, T)
        Emat = reshape(Evec, L, L)

        Σ = (Emat - ep * ep' + 1e-14 * I) / length(pos)
        Σ = ensure_positive_definite(Σ)

        model = Turing_model(Xl, PF, t, ep, Σ, sel)
        Random.seed!(26)
        chn = sample(model, NUTS(), n_samples; progress=false)
        st = extract_stats(chn; burnin_frac=0.1, thin=1)

        bs_str  = join(string.(st.bs_post), ';')
        bf_str  = join(string.(st.bf_post), ';')
        tau_str = join(string.(st.τ_post), ';')

        bs_ci  = string(st.bs_ci[1],  ",", st.bs_ci[2])
        bf_ci  = string(st.bf_ci[1],  ",", st.bf_ci[2])
        tau_ci = string(st.tau_ci[1], ",", st.tau_ci[2])

        line = string(
            gname, ",", c, ",",
            st.bs_med, ",", st.bf_med, ",",
            st.bs_ess, ",", st.bf_ess, ",", st.τ_ess, ",",
            "\"", bs_str, "\"", ",",
            "\"", bf_str, "\"", ",",
            "\"", tau_str, "\"", ",",
            "\"", bs_ci, "\"", ",",
            "\"", bf_ci, "\"", ",",
            "\"", tau_ci, "\"", ",",
            st.rho_med, ",", st.sigma_on_med, ",",
            st.sigma_off_med, ",", st.tau_med,
            "\n"
        )
        put!(out_ch, line)

        chn = nothing
        GC.gc()
    end

    return nothing
end


function prime_infer_to_csv(rdn,
                            rdm,
                            β,
                            cluster_labels,
                            genes,
                            outfile;
                            t = default_t(),
                            T = default_T(t),
                            nsigma::Real = 5.0,
                            sel::Int = 4,
                            n_samples::Int = 1000,
                            buffer_size::Int = 128)

    n_cells, n_genes = size(rdm)
    @assert size(rdn) == (n_cells, n_genes) 
    @assert length(β) == n_cells 
    @assert length(cluster_labels) == n_cells 
    @assert length(genes) == n_genes 

    clusters = unique(cluster_labels)
    base_positions = [findall(==(c), cluster_labels) for c in clusters]

    open(outfile, "w") do io
        write(io,
            "gene,class,bs,bf,bs_ess,bf_ess,tau_ess," *
            "bs_posterior,bf_posterior,tau_posterior," *
            "bs_ci,bf_ci,tau_ci," *
            "rho_med,sigma_on_med,sigma_off_med,tau_med\n")
    end

    out_ch = RemoteChannel(() -> Channel{String}(buffer_size))

    writer_task = @async begin
        open(outfile, "a") do io
            while true
                line = take!(out_ch)
                line === "__DONE__" && break
                write(io, line)
            end
        end
    end

    gene_indices = collect(1:n_genes)
    pmap(gene_indices) do gi
        BayInf_gene_pgf!(gi, genes, rdn, rdm, β,
                         clusters, base_positions,
                         t, T, out_ch;
                         nsigma = nsigma,
                         sel = sel,
                         n_samples = n_samples)
        nothing
    end

    put!(out_ch, "__DONE__")
    wait(writer_task)

    return nothing
end
