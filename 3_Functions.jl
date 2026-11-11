# =============================================================================
# SPECTRAL SIMILARITY FUNCTIONS
# Used by: central_spectrum_pipeline.jl
#
# Provides:
#   - parse_mz                     : parse spectrum strings to Float64 vectors
#   - match_peaks_in_spectra       : align two spectra by m/z (2-pointer merge)
#   - unweighted_entropy_distance  : entropy-based spectral distance
#   - cosine_similarity            : cosine distance (0 = identical)
#   - jaccard_distance             : Tanimoto-style distance
#   - manhattan_distance           : L1 distance
# =============================================================================

using LinearAlgebra, Statistics

const EPS = 1e-12


# =============================================================================
# PARSING
# =============================================================================

# Parse a stored spectrum string (e.g. "[100.0, 200.5, ...]") to Float64 vector.
# Uses string splitting — much faster and safer than eval(Meta.parse(...)).
function parse_mz(s::AbstractString)::Vector{Float64}
    isempty(s) && return Float64[]
    return [parse(Float64, m.match) for m in eachmatch(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", s)]
end


# =============================================================================
# PEAK MATCHING  (2-pointer merge, O(n+m), assumes both spectra sorted by m/z)
# =============================================================================
# Returns an (N x 3) matrix: [matched_mz | intensity_a | intensity_b]
# Unmatched peaks filled with 0. Intensities L1-normalised for scale invariance.

function match_peaks_in_spectra(
    spec_a::AbstractMatrix{<:Real},
    spec_b::AbstractMatrix{<:Real};
    ms2_da::Float64 = 0.005
)
    a_mz = @view spec_a[:, 1];  a_int = @view spec_a[:, 2]
    b_mz = @view spec_b[:, 1];  b_int = @view spec_b[:, 2]

    na = length(a_mz)
    nb = length(b_mz)

    # Pre-allocate output matrix (upper bound = na + nb rows)
    out = Matrix{Float32}(undef, na + nb, 3)
    k = 0

    i = 1;  j = 1
    @inbounds while i <= na && j <= nb
        delta = a_mz[i] - b_mz[j]
        if abs(delta) <= ms2_da
            k += 1
            out[k, 1] = Float32((a_mz[i] + b_mz[j]) / 2)
            out[k, 2] = Float32(a_int[i])
            out[k, 3] = Float32(b_int[j])
            i += 1;  j += 1
        elseif delta < 0          # a unmatched
            k += 1
            out[k, 1] = Float32(a_mz[i])
            out[k, 2] = Float32(a_int[i])
            out[k, 3] = 0.0f0
            i += 1
        else                       # b unmatched
            k += 1
            out[k, 1] = Float32(b_mz[j])
            out[k, 2] = 0.0f0
            out[k, 3] = Float32(b_int[j])
            j += 1
        end
    end
    @inbounds while i <= na
        k += 1
        out[k, 1] = Float32(a_mz[i]);  out[k, 2] = Float32(a_int[i]);  out[k, 3] = 0.0f0
        i += 1
    end
    @inbounds while j <= nb
        k += 1
        out[k, 1] = Float32(b_mz[j]);  out[k, 2] = 0.0f0;  out[k, 3] = Float32(b_int[j])
        j += 1
    end

    # L1-normalise each intensity column in-place
    sum_a = 0.0f0;  sum_b = 0.0f0
    @inbounds for r in 1:k
        sum_a += out[r, 2]
        sum_b += out[r, 3]
    end
    @inbounds if sum_a > 0
        inv_a = 1.0f0 / sum_a
        for r in 1:k;  out[r, 2] *= inv_a;  end
    end
    @inbounds if sum_b > 0
        inv_b = 1.0f0 / sum_b
        for r in 1:k;  out[r, 3] *= inv_b;  end
    end

    return out[1:k, :]
end


# =============================================================================
# SIMILARITY / DISTANCE METRICS
# All expect L1-normalised non-negative input vectors (as returned above).
# All return values >= 0 so that s = 1/(1+d) maps cleanly to (0, 1].
# =============================================================================

# --- Entropy distance ---
# Range [0, 2]. 0 = identical spectra.
# Jensen-Shannon divergence in spectral space.
function unweighted_entropy_distance(p::AbstractVector, q::AbstractVector)
    @inline function spectral_entropy(x::AbstractVector)
        s = 0.0
        @inbounds for v in x;  s += v;  end
        s <= 0.0 && return 0.0
        h = 0.0
        @inbounds for v in x
            v > 0.0 && (h -= (v / s) * log(v / s))
        end
        return h
    end

    # Compute entropy of (p+q) without allocating a new vector
    s_pq = 0.0
    @inbounds for i in eachindex(p);  s_pq += p[i] + q[i];  end
    h_pq = 0.0
    @inbounds for i in eachindex(p)
        v = (p[i] + q[i]) / s_pq
        v > 0.0 && (h_pq -= v * log(v))
    end

    return 2.0 * h_pq - spectral_entropy(p) - spectral_entropy(q)
end

# --- Cosine distance ---
# Returns DISTANCE (0 = identical). Named cosine_similarity for compatibility.
# Range [0, 1].
function cosine_similarity(p::AbstractVector, q::AbstractVector)
    num = 0.0;  sa = 0.0;  sb = 0.0
    @inbounds @simd for i in eachindex(p)
        num += p[i] * q[i]
        sa  += p[i] * p[i]
        sb  += q[i] * q[i]
    end
    return 1.0 - num / (sqrt(sa * sb) + EPS)
end

# --- Jaccard / Tanimoto distance ---
# Range [0, 1]. 0 = identical.
function binary_tanimoto(p::AbstractVector, q::AbstractVector)
    intersection = 0.0;  union_count = 0.0
    @inbounds @simd for i in eachindex(p)
        a = p[i] > 0.0;  b = q[i] > 0.0
        intersection += a & b
        union_count  += a | b
    end
    return 1.0 - intersection / (union_count + EPS)
end

# --- Manhattan distance ---
# With L1-normalised inputs lives in [0, 2]. 0 = identical.
function manhattan_distance(p::AbstractVector, q::AbstractVector)
    s = 0.0
    @inbounds @simd for i in eachindex(p)
        s += abs(p[i] - q[i])
    end
    return s
end





################Query
using Base.Threads
struct SpectrumEntry
    mz_sorted  :: Vector{Float64}
    int_sorted :: Vector{Float64}
end

# ClusterEntry now carries an optional centroid spectrum parsed from
# optimal_spectra.  nothing means no centroid row was found for this cluster;
# centroid linkage score will be NaN and does not contribute to the combined
# score (weight effectively re-distributed to complete + single).
struct LinkageClusterEntry
    inchikey        :: String
    precursor_type  :: String
    ion_mode        :: String
    precursor_mz    :: Float64
    name            :: String
    formula         :: String    
    n_spectra       :: Int
    spectra         :: Vector{SpectrumEntry}
    centroid        :: Union{SpectrumEntry, Nothing}
    centroid_idx    :: Union{Int, Nothing}
    row_indices     :: Vector{Int}
    centroid_accession :: String
    tightness           :: Float64
end


struct LinkageSearchIndex
    clusters         :: Vector{LinkageClusterEntry}
    precursor_mz_vec :: Vector{Float64}
end


# =============================================================================
# SECTION 2 — INDEX BUILDER
# =============================================================================
# Loads both the main library and the optimal_spectra CSV.  The optimal row for
# each (INCHIKEY, PRECURSOR_TYPE, ION_MODE) triple is embedded directly inside
# the ClusterEntry so no lookup is needed at search time.

function build_linkage_index(
    library_csv        :: String,
    optimal_spectra_csv :: String;
    min_spectra        :: Int  = 5,
    verbose            :: Bool = true
) :: LinkageSearchIndex

    # ── Load main library ────────────────────────────────────────────────────
    verbose && println("Loading library from: $library_csv")
    df = CSV.read(library_csv, DataFrame)
    df = dropmissing(df, [:INCHIKEY, :PRECURSOR_TYPE, :ION_MODE, :PRECURSOR_MZ,
                           :MZ_VALUES, :MZ_INT])
    verbose && println("  $(nrow(df)) spectra after dropping missing")

    # Pre-parse all spectra once
    parsed_mz  = Vector{Vector{Float64}}(undef, nrow(df))
    parsed_int = Vector{Vector{Float64}}(undef, nrow(df))
    @inbounds for i in 1:nrow(df)
        parsed_mz[i]  = parse_mz(string(df.MZ_VALUES[i]))
        parsed_int[i] = parse_mz(string(df.MZ_INT[i]))
    end

    # ── Load optimal spectra (centroids) ────────────────────────────────────
    verbose && println("Loading optimal spectra from: $optimal_spectra_csv")
    opt = CSV.read(optimal_spectra_csv, DataFrame)
    opt = dropmissing(opt, [:INCHIKEY, :PRECURSOR_TYPE, :ION_MODE,
                             :MZ_VALUES, :MZ_INT, :cluster_tightness])

    # Build a lookup Dict: (inchikey, precursor_type, ion_mode) => SpectrumEntry
    centroid_lookup = Dict{Tuple{String,String,String}, Tuple{SpectrumEntry,Int,Float64}}()
    for row in eachrow(opt)
        key = (string(row.INCHIKEY), string(row.PRECURSOR_TYPE), string(row.ION_MODE))
        mz   = parse_mz(string(row.MZ_VALUES))
        ints = parse_mz(string(row.MZ_INT))
        ord  = sortperm(mz)
        centroid_lookup[key] = (SpectrumEntry(mz[ord], ints[ord]), Int(row.ROW_ID), Float64(row.cluster_tightness))
    end
    verbose && println("  $(length(centroid_lookup)) centroid entries loaded")

    # ── Group and build clusters ─────────────────────────────────────────────
    grouped = groupby(df, [:INCHIKEY, :PRECURSOR_TYPE, :ION_MODE])

    clusters = LinkageClusterEntry[]
    sizehint!(clusters, length(grouped))

    skipped      = 0
    no_centroid  = 0

    for g in grouped
        nrow(g) < min_spectra && (skipped += 1; continue)

        row_idxs = parentindices(g)[1] |> collect

        entries = Vector{SpectrumEntry}(undef, length(row_idxs))
        @inbounds for (k, ridx) in enumerate(row_idxs)
            mz   = parsed_mz[ridx]
            ints = parsed_int[ridx]
            ord  = sortperm(mz)
            entries[k] = SpectrumEntry(mz[ord], ints[ord])
        end

        mean_precursor = mean(Float64.(g.PRECURSOR_MZ))

        name_val = ""
        if hasproperty(g, :NAME)
            for v in g.NAME
                if !ismissing(v) && !isempty(string(v))
                    name_val = string(v); break
                end
            end
        end
        formula_val = ""
        if hasproperty(g, :FORMULA)
            for v in g.FORMULA
                if !ismissing(v) && !isempty(string(v))
                    formula_val = string(v); break
                end
            end
        end

        key       = (string(g.INCHIKEY[1]), string(g.PRECURSOR_TYPE[1]),
                     string(g.ION_MODE[1]))
        lookup_hit = get(centroid_lookup, key, nothing)
        centroid, centroid_row_id, cluster_tightness_val = isnothing(lookup_hit) ? (nothing, -1, NaN) : lookup_hit
        isnothing(lookup_hit) && (no_centroid += 1)

        centroid_idx = isnothing(lookup_hit) ? nothing :
                       findfirst(==(centroid_row_id), row_idxs)

        centroid_accession = ""
       if centroid_idx !== nothing && hasproperty(df, :ACCESSION)
        val = df.ACCESSION[row_idxs[centroid_idx]]
        centroid_accession = ismissing(val) ? "" : string(val)
       end
        push!(clusters, LinkageClusterEntry(
            key[1], key[2], key[3],
            mean_precursor,
            name_val,
            formula_val,
            nrow(g),
            entries,
            centroid,
            centroid_idx,    
            row_idxs,
            centroid_accession,
            cluster_tightness_val,
        ))
    end

    precursor_mz_vec = [c.precursor_mz for c in clusters]

    if verbose
        println("  Index built: $(length(clusters)) clusters " *
                "($(skipped) skipped < $min_spectra spectra, " *
                "$(no_centroid) without optimal centroid)")
    end

    return LinkageSearchIndex(clusters, precursor_mz_vec)
end


# =============================================================================
# SECTION 3 — PER-SPECTRUM SIMILARITY
# =============================================================================
# Returns (entropy_sim, cosine_sim, tanimoto_sim) for one query vs. one library
# spectrum.  Returns (NaN, NaN, NaN) if fewer than min_matched_peaks matched.

function _pairwise_similarity(
    query   :: SpectrumEntry,
    lib     :: SpectrumEntry;
    ms2_da  :: Float64 = 0.005,
    min_matched_peaks :: Int = 1,
) :: Tuple{Float64, Float64, Float64}

    query_mat = hcat(query.mz_sorted, query.int_sorted)
    lib_mat   = hcat(lib.mz_sorted,   lib.int_sorted)
    matched   = match_peaks_in_spectra(query_mat, lib_mat; ms2_da=ms2_da)

    nrows = size(matched, 1)
    nrows < 1 && return NaN, NaN, NaN

    p = Float64.(matched[:, 2])
    q = Float64.(matched[:, 3])

    # Gate on number of shared peaks (both query and lib have intensity > 0)
    n_matched = sum(matched[r,2] > 0.0f0 && matched[r,3] > 0.0f0 for r in 1:nrows)
    n_matched < min_matched_peaks && return 0.0, 0.0, 0.0

    # ← REMOVED: sum_q renormalisation — p and q are already L1-normalised
    #   by match_peaks_in_spectra; renormalising here would remove the
    #   unmatched-peak penalty we want to preserve

    e_dist = unweighted_entropy_distance(p, q)
    e = 1.0 - clamp(e_dist / log(4), 0.0, 1.0)
    c = 1.0 - cosine_similarity(p, q)
    t = 1.0 - binary_tanimoto(p, q)

    return e, c, t
end



# =============================================================================
# SECTION 3b — DIAGNOSTIC FUNCTION
# =============================================================================
# Call this once when scores are all NaN to pinpoint exactly where things break.
# Picks the first candidate cluster after precursor filtering and traces every
# step: query shape, library spectrum shape, matched matrix, peak counts, scores.
#
# Usage:
#   diagnose_linkage_search(index_l, query_mz, query_int, precursor_mz;
#                            ion_mode="POSITIVE", mz_tol_da=0.05)

function diagnose_linkage_search(
    index        :: LinkageSearchIndex,
    query_mz     :: Vector{Float64},
    query_int    :: Vector{Float64},
    precursor_mz :: Float64;
    ion_mode     :: Union{String, Nothing} = nothing,
    mz_tol_da    :: Float64 = 0.005,
    ms2_da       :: Float64 = 0.01,
    n_members_to_show :: Int = 3,   # how many cluster members to inspect
)
    println("=" ^ 70)
    println("LINKAGE SEARCH DIAGNOSTICS")
    println("=" ^ 70)

    # ── 1. Query ──────────────────────────────────────────────────────────────
    println("\n[1] QUERY")
    println("  precursor_mz  : $precursor_mz")
    println("  n_fragments   : $(length(query_mz))")
    println("  mz range      : $(round(minimum(query_mz),digits=4)) – $(round(maximum(query_mz),digits=4))")
    println("  int range     : $(round(minimum(query_int),digits=1)) – $(round(maximum(query_int),digits=1))")
    println("  mz values     : $query_mz")

    ord   = sortperm(query_mz)
    query = SpectrumEntry(query_mz[ord], query_int[ord])

    # ── 2. Precursor filter ───────────────────────────────────────────────────
    println("\n[2] PRECURSOR FILTER  (tolerance ±$(mz_tol_da) Da)")
    pmz  = index.precursor_mz_vec
    cand = findall(i -> abs(pmz[i] - precursor_mz) <= mz_tol_da, 1:length(pmz))
    println("  candidates before ion mode filter : $(length(cand))")

    if !isnothing(ion_mode)
        cand = filter(i -> uppercase(index.clusters[i].ion_mode) == uppercase(ion_mode), cand)
        println("  candidates after  ion mode filter : $(length(cand))  (mode=$ion_mode)")
    end

    if isempty(cand)
        println("  !! NO CANDIDATES — precursor filter eliminated everything.")
        println("     Try increasing mz_tol_da (currently $mz_tol_da Da).")
        println("     Nearest library precursor m/z values:")
        diffs  = abs.(pmz .- precursor_mz)
        nearest = sortperm(diffs)[1:min(5, length(diffs))]
        for i in nearest
            c = index.clusters[i]
            println("       $(round(pmz[i],digits=5)) Da  Δ=$(round(diffs[i],digits=5))  $(c.inchikey)  $(c.precursor_type)  $(c.ion_mode)")
        end
        return
    end

    # ── 3. First candidate cluster ────────────────────────────────────────────
    ci  = cand[1]
    cl  = index.clusters[ci]
    println("\n[3] FIRST CANDIDATE CLUSTER")
    println("  inchikey      : $(cl.inchikey)")
    println("  name          : $(cl.name)")
    println("  precursor_type: $(cl.precursor_type)")
    println("  ion_mode      : $(cl.ion_mode)")
    println("  precursor_mz  : $(cl.precursor_mz)  (Δ=$(round(cl.precursor_mz - precursor_mz, digits=5)) Da)")
    println("  n_spectra     : $(cl.n_spectra)")
    println("  has centroid  : $(!isnothing(cl.centroid))")
    if !isnothing(cl.centroid)
        println("  centroid frags: $(length(cl.centroid.mz_sorted))")
        println("  centroid mz   : $(cl.centroid.mz_sorted)")
    end

    # ── 4. Inspect first N cluster members ────────────────────────────────────
    n_show = min(n_members_to_show, cl.n_spectra)
    println("\n[4] FIRST $n_show CLUSTER MEMBERS vs. QUERY  (ms2_da=$ms2_da)")

    for k in 1:n_show
        lib = cl.spectra[k]
        println("\n  ── Member $k ──")
        println("    lib n_fragments : $(length(lib.mz_sorted))")
        println("    lib mz range    : $(round(minimum(lib.mz_sorted),digits=4)) – $(round(maximum(lib.mz_sorted),digits=4))")

        query_mat = hcat(query.mz_sorted, query.int_sorted)
        lib_mat   = hcat(lib.mz_sorted,   lib.int_sorted)
        matched   = match_peaks_in_spectra(query_mat, lib_mat; ms2_da=ms2_da)

        println("    matched matrix  : $(size(matched, 1)) rows  eltype=$(eltype(matched))")

        n_both = 0
        n_query_only = 0
        n_lib_only   = 0
        for r in 1:size(matched, 1)
            a = matched[r, 2] > 0.0f0
            b = matched[r, 3] > 0.0f0
            if a && b;      n_both       += 1
            elseif a;       n_query_only += 1
            else;           n_lib_only   += 1
            end
        end
        println("    peaks matched   : $n_both both | $n_query_only query-only | $n_lib_only lib-only")

        if n_both == 0
            println("    !! ZERO shared peaks — checking m/z overlap manually:")
            q_range = (minimum(query.mz_sorted), maximum(query.mz_sorted))
            l_range = (minimum(lib.mz_sorted),   maximum(lib.mz_sorted))
            overlap = max(0.0, min(q_range[2], l_range[2]) - max(q_range[1], l_range[1]))
            println("       query mz range : $(round(q_range[1],digits=4)) – $(round(q_range[2],digits=4))")
            println("       lib   mz range : $(round(l_range[1],digits=4)) – $(round(l_range[2],digits=4))")
            println("       m/z overlap    : $(round(overlap, digits=4)) Da")
            println("       ms2_da window  : $ms2_da Da")
            println("       query mz       : $(query.mz_sorted)")
            if overlap < ms2_da
                println("    !! No m/z overlap at all — spectra are incompatible (wrong adduct/CE?)")
            else
                println("    !! m/z ranges overlap but no peaks matched within $ms2_da Da — try ms2_da=0.05")
            end
        else
            e, c, t = _pairwise_similarity(query, lib; ms2_da=ms2_da, min_matched_peaks=1)
            println("    similarity      : entropy=$(round(e,digits=4))  cosine=$(round(c,digits=4))  tanimoto=$(round(t,digits=4))")
        end
    end

    # ── 5. Centroid vs query ──────────────────────────────────────────────────
    if !isnothing(cl.centroid)
        println("\n[5] CENTROID vs. QUERY")
        lib = cl.centroid
        query_mat = hcat(query.mz_sorted, query.int_sorted)
        lib_mat   = hcat(lib.mz_sorted,   lib.int_sorted)
        matched   = match_peaks_in_spectra(query_mat, lib_mat; ms2_da=ms2_da)
        n_both = sum(matched[r,2] > 0.0f0 && matched[r,3] > 0.0f0 for r in 1:size(matched,1))
        println("  matched peaks (both) : $n_both")
        if n_both > 0
            e, c, t = _pairwise_similarity(query, lib; ms2_da=ms2_da, min_matched_peaks=1)
            println("  similarity : entropy=$(round(e,digits=4))  cosine=$(round(c,digits=4))  tanimoto=$(round(t,digits=4))")
        else
            println("  !! Zero shared peaks with centroid too")
        end
    end

    println("\n" * "=" ^ 70)
    println("TIP: if all peak counts are 0, the most likely causes are:")
    println("  (a) ms2_da too tight — try ms2_da=0.05 for a quick test")
    println("  (b) intensities not parsed correctly — check parse_mz output")
    println("  (c) query m/z values don't overlap with library (wrong mode/adduct)")
    println("=" ^ 70)
end


# =============================================================================
# SECTION 4 — LINKAGE SCORING FOR ONE CLUSTER
# =============================================================================
# Computes query-to-each-member similarities, then extracts:
#   complete  — minimum similarity across members  (most dissimilar = worst case)
#   single    — maximum similarity across members  (most similar  = best case)
#   centroid  — similarity to the pre-embedded optimal spectrum
#
# Note on direction: linkage in clustering uses *distance* (complete = max dist).
# Here we work in *similarity* space, so the mapping is:
#   complete linkage (max distance) ↔ min similarity   ← hardest match to beat
#   single   linkage (min distance) ↔ max similarity   ← easiest match to find
#
# Returns a NamedTuple with 9 Float64 fields (NaN where no valid match exists).

function score_linkage(
    query   :: SpectrumEntry,
    cluster :: LinkageClusterEntry;
    ms2_da  :: Float64 = 0.005,
    min_matched_peaks :: Int = 1,   # passed through to _pairwise_similarity
) :: NamedTuple

    n = cluster.n_spectra

    # Collect per-member similarities ─────────────────────────────────────────
    sims_e = Vector{Float64}(undef, n)
    sims_c = Vector{Float64}(undef, n)
    sims_t = Vector{Float64}(undef, n)

    @inbounds for k in 1:n
        sims_e[k], sims_c[k], sims_t[k] =
            _pairwise_similarity(query, cluster.spectra[k];
                                 ms2_da=ms2_da,
                                 min_matched_peaks=min_matched_peaks)
    end

    # Helper: filter NaN before taking extrema ────────────────────────────────
    valid_e = filter(!isnan, sims_e)
    valid_c = filter(!isnan, sims_c)
    valid_t = filter(!isnan, sims_t)

    # complete linkage = min similarity (= max distance) ─────────────────────
    comp_e = isempty(valid_e) ? NaN : minimum(valid_e)
    comp_c = isempty(valid_c) ? NaN : minimum(valid_c)
    comp_t = isempty(valid_t) ? NaN : minimum(valid_t)

    # single linkage = max similarity (= min distance) ────────────────────────
    # single linkage = max similarity (= min distance) ────────────────────────
    sing_e = isempty(valid_e) ? NaN : maximum(valid_e)
    sing_c = isempty(valid_c) ? NaN : maximum(valid_c)
    sing_t = isempty(valid_t) ? NaN : maximum(valid_t)

    # plain mean across all cluster members — ablation vs. linkage extremes ───
    mean_e = isempty(valid_e) ? NaN : mean(valid_e)
    mean_c = isempty(valid_c) ? NaN : mean(valid_c)
    mean_t = isempty(valid_t) ? NaN : mean(valid_t)

    # centroid linkage = similarity to optimal spectrum ───────────────────────

    # centroid linkage = similarity to optimal spectrum ───────────────────────
    if !isnothing(cluster.centroid)
        cent_e, cent_c, cent_t =
            _pairwise_similarity(query, cluster.centroid;
                                 ms2_da=ms2_da,
                                 min_matched_peaks=min_matched_peaks)
    else
        cent_e = cent_c = cent_t = NaN
    end

    return (
        complete_entropy  = comp_e,
        complete_cosine   = comp_c,
        complete_tanimoto = comp_t,
        single_entropy    = sing_e,
        single_cosine     = sing_c,
        single_tanimoto   = sing_t,
        centroid_entropy  = cent_e,
        centroid_cosine   = cent_c,
        centroid_tanimoto = cent_t,
        mean_entropy      = mean_e,
        mean_cosine       = mean_c,
        mean_tanimoto     = mean_t,
    )
end


function combined_linkage_score_bayes(
    s :: NamedTuple;
    use_complete :: Bool = false,
) :: Float64

    _nanmean3(a, b, c) = begin
        vals = filter(!isnan, [a, b, c])
        isempty(vals) ? NaN : mean(vals)
    end

    row_complete = _nanmean3(s.complete_entropy,  s.complete_cosine,  s.complete_tanimoto)
    row_single   = _nanmean3(s.single_entropy,    s.single_cosine,    s.single_tanimoto)
    row_centroid = _nanmean3(s.centroid_entropy,  s.centroid_cosine,  s.centroid_tanimoto)

    if isnan(row_single) || isnan(row_centroid)
        return NaN
    end

    if use_complete
        isnan(row_complete) && return NaN
        return row_complete * row_single * row_centroid
    else
        return row_single * row_centroid
    end
end



# =============================================================================
# SECTION 6 — MAIN SEARCH FUNCTION
# =============================================================================
# Drop-in replacement for search_library() in library_search.jl.
# Same precursor-filter → score → rank → DataFrame pattern, but uses linkage
# scoring instead of mean similarity.
#
# Extra keyword arguments vs. the original:
#   w_complete  :: Float64 — weight for complete-linkage row  (default 1/3)
#   w_single    :: Float64 — weight for single-linkage row    (default 1/3)
#   w_centroid  :: Float64 — weight for centroid-linkage row  (default 1/3)
#
# Output DataFrame columns:
#   rank, inchikey, precursor_type, ion_mode, name, n_spectra,
#   score_combined,
#   complete_entropy, complete_cosine, complete_tanimoto,
#   single_entropy,   single_cosine,   single_tanimoto,
#   centroid_entropy, centroid_cosine, centroid_tanimoto,
#   precursor_mz_lib, precursor_mz_delta

function search_library_linkage(
    index        :: LinkageSearchIndex,
    query_mz     :: Vector{Float64},
    query_int    :: Vector{Float64},
    precursor_mz :: Float64;
    top_n        :: Int                    = 10,
    mz_tol_da    :: Float64                = 0.005,
    ms2_da       :: Float64                = 0.005,
    ion_mode     :: Union{String, Nothing} = nothing,
    min_matched_peaks :: Int               = 1,
) :: DataFrame

    # Sort query once
    ord   = sortperm(query_mz)
    query = SpectrumEntry(query_mz[ord], query_int[ord])

    if length(query.mz_sorted) < 3
        @warn "Query has fewer than 3 fragments; returning empty result."
        return _empty_linkage_result_df()
    end

    # Step 1 — precursor m/z filter (vectorised)
    pmz            = index.precursor_mz_vec
    candidate_idxs = findall(i -> abs(pmz[i] - precursor_mz) <= mz_tol_da,
                              1:length(pmz))

    # Step 2 — optional ion mode filter
    if !isnothing(ion_mode)
        candidate_idxs = filter(
            i -> uppercase(index.clusters[i].ion_mode) == uppercase(ion_mode),
            candidate_idxs
        )
    end

    isempty(candidate_idxs) && return _empty_linkage_result_df()

    # Step 3 — score each candidate (threaded)
    n_cand  = length(candidate_idxs)
    linkage_scores  = Vector{NamedTuple}(undef, n_cand)
    combined_scores = Vector{Float64}(undef, n_cand)

    @threads for k in 1:n_cand
        ci = candidate_idxs[k]
        s  = score_linkage(query, index.clusters[ci];
                           ms2_da=ms2_da,
                           min_matched_peaks=min_matched_peaks)
        linkage_scores[k]  = s
        combined_scores[k] = combined_linkage_score_bayes(s)
    end

    # Step 4 — rank (replace NaN with -Inf so they sink to the bottom naturally)
    sortable = replace(combined_scores, NaN => -Inf)
    order    = sortperm(sortable; rev=true)
    n_out   = min(top_n, n_cand)
    top_idx = order[1:n_out]

    # Step 5 — build results DataFrame
    results = _empty_linkage_result_df()

    for (rank, k) in enumerate(top_idx)
        ci = candidate_idxs[k]
        c  = index.clusters[ci]
        s  = linkage_scores[k]
        push!(results, (
            rank,
            c.inchikey,
            c.precursor_type,
            c.ion_mode,
            c.name,
            c.n_spectra,
            combined_scores[k],
            s.complete_entropy,
            s.complete_cosine,
            s.complete_tanimoto,
            s.single_entropy,
            s.single_cosine,
            s.single_tanimoto,
            s.centroid_entropy,
            s.centroid_cosine,
            s.centroid_tanimoto,
            c.precursor_mz,
            c.precursor_mz - precursor_mz,
        ))
    end

    return results
end


# =============================================================================
# SECTION 7 — CONVENIENCE WRAPPERS
# =============================================================================

# Search directly from raw CSV strings (e.g. a row of Dataset)
function search_library_linkage_from_strings(
    index        :: LinkageSearchIndex,
    mz_string    :: AbstractString,
    int_string   :: AbstractString,
    precursor_mz :: Float64;
    kwargs...
) :: DataFrame
    query_mz  = parse_mz(mz_string)
    query_int = parse_mz(int_string)
    return search_library_linkage(index, query_mz, query_int, precursor_mz; kwargs...)
end

# Batch search: top-1 hit per row in a query DataFrame.
# Appends query_idx to each hit for alignment with the original table.
function batch_search_linkage_top1(
    index    :: LinkageSearchIndex,
    queries  :: DataFrame;
    mz_tol_da     :: Float64 = 0.005,
    ms2_da        :: Float64 = 0.01,
    ion_mode_col  :: Union{Symbol, Nothing} = :ION_MODE,
    verbose       :: Bool    = true,
) :: DataFrame

    n       = nrow(queries)
    results = DataFrame(query_idx = Int[])
    first_hit_appended = false

    verbose && println("Batch linkage search over $n queries...")

    for i in 1:n
        ion = isnothing(ion_mode_col) ? nothing : string(queries[i, ion_mode_col])
        hits = search_library_linkage_from_strings(
            index,
            string(queries.MZ_VALUES[i]),
            string(queries.MZ_INT[i]),
            Float64(queries.PRECURSOR_MZ[i]);
            top_n=1,
            mz_tol_da=mz_tol_da,
            ms2_da=ms2_da,
            ion_mode=ion,
        )

        if nrow(hits) > 0
            row = hits[1, :]
            if !first_hit_appended
                # Initialise results columns from first hit to avoid hard-coding
                for col in names(hits)
                    results[!, col] = eltype(hits[!, col])[]
                end
                first_hit_appended = true
            end
            push!(results, merge((query_idx=i,), NamedTuple(row)))
        end

        verbose && i % 1000 == 0 && println("  $i / $n done")
    end

    return results
end


# =============================================================================
# SECTION 8 — EMPTY RESULT HELPER
# =============================================================================

function _empty_linkage_result_df() :: DataFrame
    return DataFrame(
        rank               = Int[],
        inchikey           = String[],
        precursor_type     = String[],
        ion_mode           = String[],
        name               = String[],
        n_spectra          = Int[],
        score_combined     = Float64[],
        complete_entropy   = Float64[],
        complete_cosine    = Float64[],
        complete_tanimoto  = Float64[],
        single_entropy     = Float64[],
        single_cosine      = Float64[],
        single_tanimoto    = Float64[],
        centroid_entropy   = Float64[],
        centroid_cosine    = Float64[],
        centroid_tanimoto  = Float64[],
        precursor_mz_lib   = Float64[],
        precursor_mz_delta = Float64[],
    )
end

# =============================================================================
# SPECTRUM NOISE INJECTION
# Used by: 2b_Noise_benchmark.jl
#
# Generates a noisy, statistically independent replicate of a reference
# spectrum for library-search benchmarking. Design rationale (from supervisor
# discussion):
#   - the noisy spectrum is NOT excluded from matching against its parent
#     cluster, since the amount of noise added is uncontrolled / random
#     (i.e. not a deterministic transform of the original)
#   - fragments must be perturbed (m/z + intensity) AND randomly
#     dropped/added, not just jittered in place
#   - the m/z noise window must exceed the fragment matching tolerance
#     (0.005 Da in library_search_linkage.jl), otherwise you're not
#     generating meaningful false positives/negatives. Default here is
#     ±30 mDa, matching the supervisor's stated default.
#   - at least one peak is always added and at least one is always removed
#     (never a no-op call)
# =============================================================================

using Random

"""
    add_spectrum_noise(spec::SpectrumEntry; kwargs...) -> NamedTuple

Returns `(spectrum = SpectrumEntry, n_removed = Int, n_added = Int)`.

# Keyword arguments
- `mz_window::Float64 = 0.03`
      ± m/z jitter applied to every surviving peak, in Da (supervisor default: 30 mDa).
- `intensity_pct_change::Float64 = 0.20`
      Each surviving peak's intensity is scaled by a random factor in
      `[1 - intensity_pct_change, 1 + intensity_pct_change]` (uniform, independent per peak).
- `max_removed_peaks::Int = 5`
      Number of real fragments dropped is `rand(1:max_removed_peaks)` — always >= 1.
- `max_added_peaks::Int = 5`
      Number of false-positive fragments inserted is `rand(1:max_added_peaks)` — always >= 1.
- `added_peak_intensity_range::Tuple{Float64,Float64} = (0.01, 0.15)`
      Intensity range for inserted false-positive peaks (pre-normalisation scale).
- `min_peaks_remaining::Int = 3`
      Floor so removal never guts a small spectrum down to nothing.
- `apply_mz_jitter::Bool = true`
      Toggle the m/z jitter component on/off. Set `false` to isolate the
      effect of intensity-jitter and/or peak-removal noise only.
- `apply_intensity_jitter::Bool = true`
      Toggle the intensity jitter component on/off. Set `false` to isolate
      the effect of m/z-jitter and/or peak-removal noise only.
- `apply_removal::Bool = true`
      Toggle the peak-removal component on/off. Set `false` to isolate the
      effect of m/z-jitter and/or intensity-jitter noise only.
- `rng::AbstractRNG = Random.default_rng()`
      Pass your own RNG for reproducibility (e.g. Random.seed!(42) in the benchmark script).

# Isolated noise-type usage (for the noise-ablation benchmark)
Each component can be switched off independently so a benchmark cache can be
built for "m/z jitter only", "intensity change only", or "peak removal only",
in addition to the default "all three combined" behaviour:

```julia
# m/z jitter only
add_spectrum_noise(spec; apply_intensity_jitter=false, apply_removal=false)

# intensity change only
add_spectrum_noise(spec; apply_mz_jitter=false, apply_removal=false)

# peak removal only
add_spectrum_noise(spec; apply_mz_jitter=false, apply_intensity_jitter=false)
```
"""
function add_spectrum_noise(spec::SpectrumEntry;
        mz_window::Float64 = 0.03,
        intensity_pct_change::Float64 = 0.20,
        max_removed_peaks::Int = 5,
        added_peak_intensity_range::Tuple{Float64,Float64} = (0.01, 0.15),
        min_peaks_remaining::Int = 3,
        apply_mz_jitter::Bool = true,
        apply_intensity_jitter::Bool = true,
        apply_removal::Bool = true,
        rng::AbstractRNG = Random.default_rng())

    mz   = spec.mz_sorted
    ints = spec.int_sorted
    n    = length(mz)
    @assert length(ints) == n "mz_sorted and int_sorted must be the same length"

    # --- 1. m/z jitter: uniform noise in [-mz_window, +mz_window] -----------
    if apply_mz_jitter
        mz_noisy = mz .+ (rand(rng, n) .* 2 .- 1) .* mz_window
    else
        mz_noisy = copy(mz)
    end

    # --- 2. intensity jitter: +/- intensity_pct_change, independent per peak
    if apply_intensity_jitter
        pct       = (rand(rng, n) .* 2 .- 1) .* intensity_pct_change
        int_noisy = ints .* (1 .+ pct)
        int_noisy = max.(int_noisy, 0.0)   # guard against negative intensities
    else
        int_noisy = copy(ints)
    end

    # --- 3. randomly remove fragments (always >= 1 when enabled) -------------
    if apply_removal
        n_removed = rand(rng, 1:max_removed_peaks)
        n_removed = min(n_removed, max(n - min_peaks_remaining, 0))
        if n_removed > 0
            keep_idx  = sort(shuffle(rng, 1:n)[1:(n - n_removed)])
            mz_noisy  = mz_noisy[keep_idx]
            int_noisy = int_noisy[keep_idx]
        end
    else
        n_removed = 0
    end

    # --- 5. re-sort by m/z — match_peaks_in_spectra assumes sorted input ----
    order = sortperm(mz_noisy)

    return (
        spectrum  = SpectrumEntry(mz_noisy[order], int_noisy[order]),
        n_removed = n_removed,
    )
end

# --- 4. randomly add false-positive fragments (always >= 1) --------------
    #n_added = rand(rng, 1:max_added_peaks)
    #mz_lo, mz_hi = extrema(mz)
    #added_mz  = mz_lo .+ rand(rng, n_added) .* (mz_hi - mz_lo)
    #alo, ahi  = added_peak_intensity_range
    #added_int = alo .+ rand(rng, n_added) .* (ahi - alo)
    #mz_noisy  = vcat(mz_noisy, added_mz)
    #int_noisy = vcat(int_noisy, added_int)