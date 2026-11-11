using DataFrames, CSV, JSON3
using ProgressMeter, StatsPlots, Plots
using ScikitLearn, PyCall, Statistics
using LinearAlgebra



"""
Convert MS/MS fragment list to a CNL fingerprint bit-vector.
Matches each CNL to the nearest header bin within tolerance (0.01 + 0.0001 buffer).
"""
function frags_to_cnls_vec(precursor_mz, frag_mzs, cnl_headers; tolerance=0.01)
    cnls   = precursor_mz .- frag_mzs
    cnl_fp = zeros(length(cnl_headers))
    for cnl in cnls
        cnl_fp[abs.(cnl_headers .- cnl) .<= tolerance + 0.0001] .= 1
    end
    return cnl_fp
end
"""Parse a JSON-formatted string of floats, returning Float64[] on failure."""
function safe_parse_array(x)
    try
        return JSON3.read(replace(x, " " => ""), Vector{Float64})
    catch
        return Float64[]
    end
end

#Import Data
all_CompCreate = "CompCreate data/"
all_files = readdir(all_CompCreate)
file_1 = CSV.read(joinpath("CompCreate data/", all_files[1]), DataFrame)

#CNL generation 
cnl_headers = CSV.read("CNL Headers.csv", DataFrame)[:, 1]


for i in eachindex(all_files)

    cnl_vecs = @showprogress [
        frags_to_cnls_vec(file_1.MeasMass[i], safe_parse_array(file_1.MS2Comp[i]), cnl_headers)
        for i in axes(file_1, 1)
    ]

    



