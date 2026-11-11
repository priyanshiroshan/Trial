# =============================================================================
# METABOLITE NETWORKING — NEW FUNCTIONS
#
# These are additive to 3_Functions.jl (which is `include`d before this file —
# see run_metabolite_networking.jl). Nothing here touches your existing library
# search / linkage code; it reuses the low-level building blocks
# (SpectrumEntry, match_peaks_in_spectra, _pairwise_similarity) directly.
#
# What this file provides, in the order you'd call them:
#   load_delta_table         — read the curated biotransformation Δm table
#   find_ms1_candidate_edges — MS1 layer: within-sample Δm matching
#   build_feature_lookup     — helper: (mz, frag_mzs, frag_ints) per feature
#   frags_to_cnl_spectrum    — turn one feature's fragments into a CNL "spectrum"
#                              using your ORIGINAL fixed CNL bins, but sparse and
#                              intensity-aware so it plugs into match_peaks_in_spectra
#   pairwise_match_factor    — the "just give me the combined entropy+cosine+
#                              tanimoto score between these two spectra" function
#                              you asked for, with no library/centroid machinery
#   score_candidate_edges    — MS2 layer: applies pairwise_match_factor to every
#                              candidate edge, twice (raw fragments + CNL)
# =============================================================================

using DataFrames, CSV, Statistics


# =============================================================================
# DELTA TABLE
# =============================================================================

"""
    load_delta_table(path) -> DataFrame

Loads the curated biotransformation mass-shift table (transformation name, Δm in
Da, category, notes). See biotransformation_deltas.csv for the actual values —
these are standard monoisotopic mass differences for the transferred atoms/groups
(e.g. +O = 15.9949, -CH2 = -14.0157), cross-checked against the drug-metabolism
HR-MS literature, not invented numbers. Edit that CSV directly to add/remove rows;
nothing here is hard-coded.

Two rows (`methylation` and `ethanol_transesterification`) share the exact same
Δm (+14.0157) on purpose — see the notes column in the CSV. Mass alone cannot
separate them; both will show up together whenever a pair matches that Δm.
"""
function load_delta_table(path::String) :: DataFrame
    tbl = CSV.read(path, DataFrame)
    @assert all(c -> c in names(tbl), ["transformation", "delta_mass"]) "delta table must have transformation and delta_mass columns"
    return tbl
end


# =============================================================================
# MS1 LAYER — WITHIN-SAMPLE MASS-DIFFERENCE CANDIDATE EDGES
# =============================================================================

"""
    find_ms1_candidate_edges(measmass, feature_ids, delta_table; ppm_tol=5.0, min_abs_tol=0.001) -> DataFrame

MS1 layer of the pipeline. For every pair of features *within one sample*, tests
whether their mass difference matches a known biotransformation Δm from
`delta_table`, checked in *both directions* — you don't need the table to list
both +15.9949 and -15.9949, this tries measmass[i] - measmass[j] for every
ordered pair, so whichever feature turns out to be the parent is discovered
automatically rather than assumed.

Tolerance is ppm-scaled to the pair's mass (appropriate for HRMS, where mass
accuracy is quoted in ppm, not as a flat Da window), with `min_abs_tol` as a floor
so very small Δm entries in the table (like ±2.0157 for redox) don't get an
unrealistically tight window at low mass.

By design, NO retention-time filter is applied — per your note, a metabolite
isn't reliably earlier/later eluting than its parent, so RT is not used to filter
here. If you later want it, `df.RT` could be joined back onto the output using
`parent_id`/`metabolite_id` for downstream inspection.

A single feature pair can appear more than once in the output if its Δm matches
more than one transformation (e.g. the methylation / ethanol-transesterification
ambiguity) — that's intentional: this layer only generates hypotheses, it doesn't
adjudicate between them. `score_candidate_edges` (MS2 layer) is what should help
you decide which hypothesis is more likely for a given pair.

# Returns
DataFrame with columns:
`parent_id, metabolite_id, parent_mz, metabolite_mz, delta_observed,
transformation, delta_expected, ppm_error`

# Performance note
This is O(n² × m) where n = features in the sample and m = rows in the delta
table (currently 15). Fine for the hundreds-to-low-thousands of features typical
of one non-target sample; if you ever run this on tens of thousands of features
per sample, it's worth switching to a sorted-array + binary-search approach per
delta-table row instead of the brute-force double loop below — say if that
becomes a bottleneck and I'll rewrite this function.
"""
function find_ms1_candidate_edges(
    measmass    :: Vector{Float64},
    feature_ids :: AbstractVector,
    delta_table :: DataFrame;
    ppm_tol     :: Float64 = 5.0,
    min_abs_tol :: Float64 = 0.001,
) :: DataFrame

    n = length(measmass)
    @assert length(feature_ids) == n "measmass and feature_ids must be the same length"

    rows = NamedTuple[]

    @inbounds for i in 1:n, j in 1:n
        i == j && continue
        # delta_observed such that measmass[i] = measmass[j] + delta_observed
        # i.e. candidate i = metabolite, candidate j = parent, for THIS ordering.
        # The reverse ordering (j vs i) is covered on its own pass of the loop.
        delta_observed = measmass[i] - measmass[j]

        ref_mass = max(measmass[i], measmass[j])
        tol      = max(min_abs_tol, ppm_tol * 1e-6 * ref_mass)

        for r in eachrow(delta_table)
            if abs(delta_observed - r.delta_mass) <= tol
                push!(rows, (
                    parent_id      = feature_ids[j],
                    metabolite_id  = feature_ids[i],
                    parent_mz      = measmass[j],
                    metabolite_mz  = measmass[i],
                    delta_observed = delta_observed,
                    transformation = r.transformation,
                    delta_expected = r.delta_mass,
                    ppm_error      = (delta_observed - r.delta_mass) / ref_mass * 1e6,
                ))
            end
        end
    end

    isempty(rows) && return DataFrame(
        parent_id=Any[], metabolite_id=Any[], parent_mz=Float64[], metabolite_mz=Float64[],
        delta_observed=Float64[], transformation=String[], delta_expected=Float64[], ppm_error=Float64[],
    )

    return DataFrame(rows)
end


# =============================================================================
# FEATURE LOOKUP HELPER
# =============================================================================

"""
    build_feature_lookup(df; id_col=:feature_id, mz_col=:MeasMass, frag_mz_col=:MS2Comp, frag_int_col=:MS2CompInt) -> Dict

Builds the `feature_id => (mz, frag_mzs, frag_ints)` lookup that `score_candidate_edges`
needs, from one sample's CompCreate DataFrame.

⚠ ASSUMPTION TO VERIFY — I only saw `MeasMass` and `MS2Comp` in what you pasted, not
a fragment-intensity column name. This defaults to looking for `:MS2CompInt`. If your
real column is named differently (or intensities are packed into the same string as
`MS2Comp`, e.g. "mz:int" pairs rather than a separate parallel array), tell me the
actual layout and I'll change the parser rather than have you patch this blind. If no
intensity column is found by that name, this prints one warning and falls back to
uniform intensity (every fragment weighted equally) so the pipeline still runs instead
of crashing — but note this silently degrades the MS2 match factors back to something
closer to a presence/absence comparison, so it's worth fixing the column name properly
rather than leaving the fallback in place.
"""
function build_feature_lookup(
    df           :: DataFrame;
    id_col       :: Symbol = :feature_id,
    mz_col       :: Symbol = :MeasMass,
    frag_mz_col  :: Symbol = :MS2Comp,
    frag_int_col :: Symbol = :MS2CompInt,
) :: Dict

    has_intensity = hasproperty(df, frag_int_col)
    has_intensity || @warn "Column :$frag_int_col not found — using uniform fragment intensity (all 1.0) for every feature. Check the real column name for fragment intensities." maxlog=1

    lookup = Dict{Any, NamedTuple}()
    for row in eachrow(df)
        fid   = row[id_col]
        fmzs  = safe_parse_array(string(row[frag_mz_col]))
        fints = has_intensity ? safe_parse_array(string(row[frag_int_col])) : ones(length(fmzs))

        if length(fints) != length(fmzs)
            @warn "Fragment mz/intensity length mismatch for feature $fid — using uniform intensity for this feature only."
            fints = ones(length(fmzs))
        end

        lookup[fid] = (mz = Float64(row[mz_col]), frag_mzs = fmzs, frag_ints = fints)
    end
    return lookup
end


# =============================================================================
# CNL SPECTRUM BUILDER — using your ORIGINAL fixed CNL bins
# =============================================================================

"""
    frags_to_cnl_spectrum(precursor_mz, frag_mzs, frag_ints, cnl_headers; tolerance=0.01, use_intensity=true) -> SpectrumEntry

Converts one feature's MS/MS fragments into a "CNL spectrum": same idea as your
original `frags_to_cnls_vec`, and using the SAME fixed bin vocabulary
(`cnl_headers`, loaded from `CNL Headers.csv`) — just returned as a sparse
`SpectrumEntry` (only bins actually hit) instead of a dense 0/1 vector over every
header. That's what lets it drop straight into `match_peaks_in_spectra` /
`_pairwise_similarity` from 3_Functions.jl, reusing code you've already validated,
rather than needing a second bespoke comparison method for CNLs.

# The one deliberate change from the original: intensity, not just presence
`frags_to_cnls_vec` marks a bin 1 or 0 regardless of how large the underlying
fragment peak was — correct for a fixed-length ML input (your RB classifier),
but for comparing two specific CNL spectra to each other it throws away real
discriminating signal: two features that both show a "loss of 18" driven by a
dominant fragment vs a trace fragment are not equally similar. Here, when one or
more fragments land in the same bin, their intensities are summed into that bin.

Set `use_intensity=false` if you want to reproduce the original binary behaviour
exactly (every hit bin gets intensity 1.0) — e.g. for a direct side-by-side
comparison against the RB-model style fingerprint.

# Arguments
- `precursor_mz`  — this feature's MS1 mass (`MeasMass`)
- `frag_mzs`      — this feature's fragment m/z values
- `frag_ints`     — this feature's fragment intensities, same order as `frag_mzs`
- `cnl_headers`   — the fixed bin vocabulary from CNL Headers.csv
- `tolerance`     — matches your original `+0.0001` buffer convention (default 0.01 Da bin ± buffer)
- `use_intensity` — true (default) = intensity-weighted bins; false = binary bins (original behaviour)
"""
function frags_to_cnl_spectrum(
    precursor_mz :: Float64,
    frag_mzs     :: Vector{Float64},
    frag_ints    :: Vector{Float64},
    cnl_headers  :: Vector{Float64};
    tolerance     :: Float64 = 0.01,
    use_intensity :: Bool    = false,
) :: SpectrumEntry

    @assert length(frag_mzs) == length(frag_ints) "frag_mzs and frag_ints must be the same length"

    if isempty(frag_mzs)
        return SpectrumEntry(Float64[], Float64[])
    end

    cnls = precursor_mz .- frag_mzs

    bin_intensity = Dict{Float64, Float64}()   # header bin value -> accumulated intensity
    for (cnl, inten) in zip(cnls, frag_ints)
        hits = findall(h -> abs(h - cnl) <= tolerance + 0.0001, cnl_headers)
        isempty(hits) && continue
        add = use_intensity ? inten : 1.0
        for h_idx in hits
            h = cnl_headers[h_idx]
            bin_intensity[h] = get(bin_intensity, h, 0.0) + add
        end
    end

    isempty(bin_intensity) && return SpectrumEntry(Float64[], Float64[])

    mzs  = collect(keys(bin_intensity))
    ints = collect(values(bin_intensity))
    ord  = sortperm(mzs)
    return SpectrumEntry(mzs[ord], ints[ord])
end


# =============================================================================
# STAND-ALONE COMBINED MATCH FACTOR (no library, no clustering, no centroid)
# =============================================================================

"""
    pairwise_match_factor(spec_a, spec_b; ms2_da=0.005, min_matched_peaks=1) -> NamedTuple

The stand-alone combined entropy+cosine+tanimoto match factor between exactly two
spectra — this is the piece you asked for directly: "get the combined match
factor... we are not comparing to a library, we are getting match factors within
the parents and possible metabolites."

Internally this is nothing new: it's `_pairwise_similarity` (already defined in
3_Functions.jl, unmodified) plus the same simple mean-of-three combination that
`combined_linkage_score_bayes` uses per row (there called `_nanmean3`). Using the
exact same combination rule means this number sits on a directly comparable scale
to anything you've already produced with the library-search code — it's just not
multiplied against a "single-linkage" and "centroid-linkage" row the way the
library search score is, since here there's no cluster/library to link against;
there's just the one query spectrum and one candidate spectrum.

Works identically whether `spec_a`/`spec_b` are raw fragment spectra or CNL
spectra from `frags_to_cnl_spectrum` — both are just `SpectrumEntry` objects, and
`match_peaks_in_spectra`/`_pairwise_similarity` don't care which axis they represent.

# Returns
`(entropy, cosine, tanimoto, combined, n_matched)` — `combined` is the mean of
whichever of the three metrics are not NaN; `n_matched` is how many peaks/bins the
comparison actually shared, so you can filter out match factors computed from a
suspiciously small number of matches (e.g. `n_matched < 3`).
"""
function pairwise_match_factor(
    spec_a :: SpectrumEntry,
    spec_b :: SpectrumEntry;
    ms2_da :: Float64 = 0.005,
    min_matched_peaks :: Int = 1,
) :: NamedTuple

    if isempty(spec_a.mz_sorted) || isempty(spec_b.mz_sorted)
        return (entropy=NaN, cosine=NaN, tanimoto=NaN, combined=NaN, n_matched=0)
    end

    e, c, t = _pairwise_similarity(spec_a, spec_b; ms2_da=ms2_da, min_matched_peaks=min_matched_peaks)

    a_mat   = hcat(spec_a.mz_sorted, spec_a.int_sorted)
    b_mat   = hcat(spec_b.mz_sorted, spec_b.int_sorted)
    matched = match_peaks_in_spectra(a_mat, b_mat; ms2_da=ms2_da)
    n_matched = sum(matched[r,2] > 0.0f0 && matched[r,3] > 0.0f0 for r in 1:size(matched,1))

    vals     = filter(!isnan, [e, c, t])
    combined = isempty(vals) ? NaN : mean(vals)

    return (entropy=e, cosine=c, tanimoto=t, combined=combined, n_matched=n_matched)
end


# =============================================================================
# MS2 LAYER — SCORE EVERY CANDIDATE MS1 EDGE
# =============================================================================

"""
    score_candidate_edges(edges, feature_lookup, cnl_headers; ms2_da=0.01, cnl_tol=0.01, use_intensity=true) -> DataFrame

MS2 confirmation layer. For every candidate MS1 edge (the output of
`find_ms1_candidate_edges`), computes TWO independent pieces of spectral evidence
— deliberately not just one, because they're expected to succeed in different
cases:

  - `mz_*` columns  — combined match factor on the RAW fragment spectra (parent
    vs metabolite). This works well when the transformation leaves most fragments
    untouched (oxidation, reduction, methylation), since the raw peaks barely
    shift and align directly.

  - `cnl_*` columns — combined match factor on each feature's CNL spectrum
    (`frags_to_cnl_spectrum`, using your original fixed bins) instead of raw
    fragments. This is the one that matters when the transformation shifts MOST
    fragments by the same Δm (dealkylation, conjugation, and your
    ethanol-transesterification case) — the raw peaks move, but the underlying
    neutral losses for any fragment not containing the modified site stay
    numerically identical between parent and metabolite, so this stays sensitive
    exactly where `mz_*` is expected to fail.

Neither column is treated as "the" answer — both are appended so you can define
your own confirmation rule downstream, e.g. keep an edge if
`cnl_combined > 0.5 || mz_combined > 0.5`, or weight them by transformation type
(favour `cnl_combined` for the dealkylation/conjugation rows in the delta table,
favour `mz_combined` for oxidation/reduction rows).

`feature_lookup` is the output of `build_feature_lookup`.
"""
function score_candidate_edges(
    edges          :: DataFrame,
    feature_lookup :: Dict,
    cnl_headers    :: Vector{Float64};
    ms2_da         :: Float64 = 0.01,
    cnl_tol        :: Float64 = 0.01,
    use_intensity  :: Bool    = false,
) :: DataFrame

    n = nrow(edges)

    mz_entropy   = Vector{Float64}(undef, n); mz_cosine   = Vector{Float64}(undef, n)
    mz_tanimoto  = Vector{Float64}(undef, n); mz_combined = Vector{Float64}(undef, n)
    cnl_entropy  = Vector{Float64}(undef, n); cnl_cosine  = Vector{Float64}(undef, n)
    cnl_tanimoto = Vector{Float64}(undef, n); cnl_combined = Vector{Float64}(undef, n)
    n_matched_mz  = Vector{Int}(undef, n)
    n_matched_cnl = Vector{Int}(undef, n)

    @showprogress for k in 1:n
        row = edges[k, :]
        p = feature_lookup[row.parent_id]
        m = feature_lookup[row.metabolite_id]

        # --- raw fragment match factor ---
        ord_p  = sortperm(p.frag_mzs)
        ord_m  = sortperm(m.frag_mzs)
        spec_p = SpectrumEntry(p.frag_mzs[ord_p], p.frag_ints[ord_p])
        spec_m = SpectrumEntry(m.frag_mzs[ord_m], m.frag_ints[ord_m])

        mzres = pairwise_match_factor(spec_p, spec_m; ms2_da=ms2_da)
        mz_entropy[k]  = mzres.entropy
        mz_cosine[k]   = mzres.cosine
        mz_tanimoto[k] = mzres.tanimoto
        mz_combined[k] = mzres.combined
        n_matched_mz[k] = mzres.n_matched

        # --- CNL match factor (original fixed bins, intensity-aware) ---
        cnl_p = frags_to_cnl_spectrum(p.mz, p.frag_mzs, p.frag_ints, cnl_headers;
                                       tolerance=cnl_tol, use_intensity=use_intensity)
        cnl_m = frags_to_cnl_spectrum(m.mz, m.frag_mzs, m.frag_ints, cnl_headers;
                                       tolerance=cnl_tol, use_intensity=use_intensity)

        cnlres = pairwise_match_factor(cnl_p, cnl_m; ms2_da=ms2_da)
        cnl_entropy[k]  = cnlres.entropy
        cnl_cosine[k]   = cnlres.cosine
        cnl_tanimoto[k] = cnlres.tanimoto
        cnl_combined[k] = cnlres.combined
        n_matched_cnl[k] = cnlres.n_matched
    end

    result = copy(edges)
    result.mz_entropy    = mz_entropy
    result.mz_cosine     = mz_cosine
    result.mz_tanimoto   = mz_tanimoto
    result.mz_combined   = mz_combined
    result.n_matched_mz  = n_matched_mz
    result.cnl_entropy   = cnl_entropy
    result.cnl_cosine    = cnl_cosine
    result.cnl_tanimoto  = cnl_tanimoto
    result.cnl_combined  = cnl_combined
    result.n_matched_cnl = n_matched_cnl

    return result
end


# =============================================================================
# IDENTITY ANCHORING — no library, just known dosed-compound masses
# =============================================================================

"""
    identify_anchor_features(measmass, feature_ids, known_compounds; ppm_tol=10.0) -> DataFrame

Matches MS1 features directly against a small table of compounds you KNOW were
dosed (exact ion mass only — no spectral matching needed, since these aren't
unknowns). This is what gives the network real starting identities instead of
every node being an anonymous feature_id.

`known_compounds` (see known_compounds.csv) needs columns `name` and
`exact_mass`, where exact_mass is the ION mass you expect in YOUR ionization
mode (e.g. [M+H]+ in positive mode) — not the neutral monoisotopic mass. The
values I put in the template are standard [M+H]+ masses computed from formula,
but verify against your own reference standard/CoA before trusting them,
especially if you're using a salt form.

Returns one row per (feature, known compound) match within ppm_tol. Normally
zero or one hit per feature — if you get more than one, check whether two of
your dosed compounds are close enough in mass to be ambiguous at this ppm_tol.
"""
function identify_anchor_features(
    measmass        :: Vector{Float64},
    feature_ids     :: AbstractVector,
    known_compounds :: DataFrame;
    ppm_tol         :: Float64 = 10.0,
) :: DataFrame

    rows = NamedTuple[]
    for (i, mz) in enumerate(measmass)
        for r in eachrow(known_compounds)
            tol = ppm_tol * 1e-6 * r.exact_mass
            if abs(mz - r.exact_mass) <= tol
                push!(rows, (
                    feature_id = feature_ids[i],
                    mz         = mz,
                    name       = r.name,
                    exact_mass = r.exact_mass,
                    ppm_error  = (mz - r.exact_mass) / r.exact_mass * 1e6,
                ))
            end
        end
    end

    isempty(rows) && return DataFrame(feature_id=Any[], mz=Float64[], name=String[],
                                       exact_mass=Float64[], ppm_error=Float64[])
    return DataFrame(rows)
end


# =============================================================================
# NETWORK CONSTRUCTION — threshold + breadth-first walk from anchors
# =============================================================================

"""
    build_confirmed_network(scored_edges, anchors, feature_lookup;
                             score_col=:cnl_combined, min_score=0.5,
                             n_matched_col=:n_matched_cnl, min_n_matched=3) -> (nodes, edges)

Turns the big scored candidate-edge table into an actual network you can
inspect, in two steps:

  1. THRESHOLD — keep only edges whose confirmation score clears your cutoff
     (default: cnl_combined >= 0.5, based on >= 3 matched CNL bins). Both
     numbers are yours to tune once you've eyeballed a real score distribution
     — there's nothing principled about 0.5 other than "the midpoint," start
     there and adjust after looking at a histogram of scored_edges.cnl_combined.

  2. WALK — starting from the known anchor features (from
     `identify_anchor_features`), breadth-first search outward along the
     surviving edges, labelling every reachable feature with a `generation`:
     0 = anchor/dosed parent drug, 1 = direct metabolite, 2 =
     metabolite-of-a-metabolite, and so on. This is what actually assigns
     "parent" vs "metabolite" roles rather than leaving it as an untested
     mass-difference hypothesis.

Returns two DataFrames sized for direct import into Cytoscape/Gephi:
  - `nodes`: feature_id, mz, name (only set for anchors), generation, is_anchor
  - `edges`: the confirmed subset of scored_edges, restricted to nodes that
    were actually reached from an anchor

Anything NOT reachable from an anchor at your score threshold is deliberately
left out of both tables — those are either false-positive MS1 coincidences, or
real relationships your threshold is currently too strict to confirm. Inspect
`scored_edges` directly (unfiltered) if you want to see unlinked/orphan
clusters too — e.g. two unidentified features that match each other well but
never connect back to a known parent, which is its own interesting case
(could be metabolites of an unlabelled/impure component in the mix).
"""
function build_confirmed_network(
    scored_edges   :: DataFrame,
    anchors        :: DataFrame,
    feature_lookup :: Dict;
    score_col      :: Symbol  = :cnl_combined,
    min_score      :: Float64 = 0.5,
    n_matched_col  :: Symbol  = :n_matched_cnl,
    min_n_matched  :: Int     = 3,
)
    confirmed = filter(
        r -> !ismissing(r[score_col]) && !isnan(r[score_col]) &&
             r[score_col] >= min_score && r[n_matched_col] >= min_n_matched,
        scored_edges,
    )

    # adjacency: parent_id -> Vector of metabolite_ids it points to
    adjacency = Dict{Any, Vector{Any}}()
    for r in eachrow(confirmed)
        push!(get!(adjacency, r.parent_id, Any[]), r.metabolite_id)
    end

    generation = Dict{Any, Int}()
    for aid in anchors.feature_id
        generation[aid] = 0
    end

    queue = collect(anchors.feature_id)
    while !isempty(queue)
        current = popfirst!(queue)
        for child in get(adjacency, current, Any[])
            if !haskey(generation, child) || generation[child] > generation[current] + 1
                generation[child] = generation[current] + 1
                push!(queue, child)
            end
        end
    end

    node_ids      = collect(keys(generation))
    anchor_lookup = Dict(a.feature_id => a.name for a in eachrow(anchors))

    nodes = DataFrame(
        feature_id = node_ids,
        mz         = [feature_lookup[id].mz for id in node_ids],
        generation = [generation[id] for id in node_ids],
        is_anchor  = [haskey(anchor_lookup, id) for id in node_ids],
        name       = [get(anchor_lookup, id, missing) for id in node_ids],
    )

    edges_out = filter(r -> r.parent_id in node_ids && r.metabolite_id in node_ids, confirmed)

    return nodes, edges_out
end
